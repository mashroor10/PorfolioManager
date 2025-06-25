import sys
from SignalStrategy import SignalStrategy
from TradingStrategy import TradingStrategy
import pandas as pd
import math
import numpy as np
import time
import concurrent.futures
from backtesting import Backtest
from CreateCombinedData import getCombinedDataDict
from Database import PostgresManager
import os
from dotenv import load_dotenv
import gc
import psutil

load_dotenv()
pg = PostgresManager(
    host=os.getenv('host'),
    port=os.getenv('port'),
    dbname=os.getenv('dbname'),
    user=os.getenv('user'),
    password=os.getenv('password')
)
from sqlalchemy import text


def run_backtest_worker(ticker, df, strategy_name):
    """Process-isolated backtest function with minimal memory footprint"""
    try:
        # Reduce memory usage immediately
        df = df.astype({
            'Open': 'float32',
            'High': 'float32',
            'Low': 'float32',
            'Close': 'float32',
            'Volume': 'float32',
            'signal': 'int8'
        })
        
        # Dynamic import inside worker
        from backtesting import Backtest
        
        # Import strategy based on name
        if strategy_name == "EbitdaValueStrategy":
            from SignalStrategy import SignalStrategy
            strategy_class = SignalStrategy
        else:
            # Add other strategy mappings as needed
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        bt = Backtest(df, strategy_class, cash=10_000, commission=.002)
        stats = bt.optimize(
            stopLoss=range(1, 5, 1),
            takeProfit=range(12, 25, 1),
            maximize='Sharpe Ratio',
            return_heatmap=False,
            max_tries=50,
        )
        
        # Return only essential statistics
        return {
            'Sharpe Ratio': stats['Sharpe Ratio'],
            'Return [%]': stats['Return [%]'],
            'Max. Drawdown [%]': stats['Max. Drawdown [%]'],
            '_equity_curve': stats['_equity_curve'][['Equity']].copy()
        }
    except Exception as e:
        print(f"❌ Error in {ticker} backtest: {str(e)}")
        return None
    finally:
        # Explicit cleanup
        del bt, stats, df
        gc.collect()    

class EbitdaValueStrategy(TradingStrategy):
    def createProcessedData(self):
        self.filterRawData()
        """Preprocess raw data into working format"""
        column_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        for ticker in self.rawData.keys():
            df = self.rawData[ticker].copy()
            # Drop the symbol column if it exists
            if 'symbol' in df.columns:
                df.drop(columns=['symbol'], inplace=True)
            # Rename columns using the mapping
            df.rename(columns=column_map, inplace=True)
            # If High or Low columns don't exist, create them as a copy of Close
            if 'High' not in df.columns:
                df['High'] = df['Close']
            if 'Low' not in df.columns:
                df['Low'] = df['Close']
            self.rawData[ticker] = df
            
            min_date = min(df.index.min() for df in self.rawData.values())
            print(f"Minimum date after processing: {min_date}")
            
    def createValueData(self):
        # 1. Create EBITDA DataFrame
        value_df = pd.DataFrame()
        rawData = self.rawData.copy()
        hasEbitda = 0
        # Collect EBITDA values for each ticker
        for ticker, df in rawData.items():
            if 'ebitdaValue' in df.columns:
                hasEbitda += 1
                # Ensure we have a datetime index
                temp = df[['ebitdaValue']].copy()
                temp.index = pd.to_datetime(temp.index)
                
                # Rename column to ticker name
                temp = temp.rename(columns={'ebitdaValue': ticker})
                
                if value_df.empty:
                    value_df = temp
                else:
                    value_df = value_df.join(temp, how='outer')
                
                
        
        #Make the Nan values 0
        value_df.fillna(0, inplace=True)
        
        self.value_df = value_df
    
    def getTopBottomTickers(self):
        value_df = self.value_df.copy()

        # 3. Rank values row-wise (date-wise)
        rank_df = value_df.rank(axis=1, method='min', ascending=False)
        rank_df.dropna(axis=1, how='all', inplace=True)  # Drop columns with all NaN values

        self.rank_df = rank_df
        
        # 4. Identify top/bottom tickers
        top_bottom_tickers = set()
        
        # Iterate through each date
        for date, row in rank_df.iterrows():
            # Get top 50 and bottom 50 tickers for this date
            sorted_tickers = row.sort_values().index
            #removing the tickers that have 0 as the value
            sorted_tickers = [ticker for ticker in sorted_tickers if row[ticker] > 0]
            window = self.window
            if len(sorted_tickers) < self.window * 2:
                window = math.floor(self.window/2)
                continue
            top_tickers = sorted_tickers[:self.window]
            bottom_tickers = sorted_tickers[-self.window:]
            
            # Add to our set
            top_bottom_tickers.update(top_tickers)
            top_bottom_tickers.update(bottom_tickers)
            
        return top_bottom_tickers
            
    def filterRawData(self):
        # getting the lowest minimum date of all the dataframes in rawData
        min_date = min(df.index.min() for df in self.rawData.values())
        print(f"Minimum date before filtration: {min_date}")
        
        self.createValueData()
        top_bottom_tickers = self.getTopBottomTickers()
        
        # 5. Filter rawData to keep only top/bottom tickers
        filtered_rawData = {
            ticker: df 
            for ticker, df in self.rawData.items() 
            if ticker in top_bottom_tickers
        }
        
        self.rawData = filtered_rawData
        
        min_date = min(df.index.min() for df in self.rawData.values())
        print(f"Minimum date after filtration: {min_date}")
        
        # Only keep columns in rank_df that are in filtered_rawData
        filtered_columns = [col for col in self.rank_df.columns if col in filtered_rawData.keys()]
        self.rank_df = self.rank_df[filtered_columns]
        # Create the rankedDict {date: {ticker: rank}}
        rankedDict = {}
        for date, row in self.rank_df.iterrows():
            date_ranks = {}
            for ticker, rank_value in row.items():
                date_ranks[ticker] = int(rank_value)  # Convert to integer rank
            rankedDict[date] = date_ranks
            
        self.rankedDict = rankedDict
        

        print(f"Original tickers: {len(self.rawData)}")
        print(f"Filtered tickers: {len(filtered_rawData)}")
        
    def mergedData(self):
        min_date = min(df.index.min() for df in self.rawData.values())
        print(f"Minimum date before margedData: {min_date}")
        
        for ticker in self.rawData.keys():
            df = self.rawData[ticker].copy()
            # Drop the symbol column if it exists
            if 'symbol' in df.columns:
                df.drop(columns=['symbol'], inplace=True)
            # Always add prefix for consistency
            df = df.add_prefix(f"{ticker}_")
            if self.workingData.empty:
                self.workingData = df.copy()
            else:
                self.workingData = self.workingData.join(df, how='outer')
            # If {ticker}_symbol column exists, drop it
            symbol_col = f"{ticker}_symbol"
            if symbol_col in self.workingData.columns:
                self.workingData.drop(columns=[symbol_col], inplace=True)
        print(f"Minimum date of workingData after mergedData: {self.workingData.index.min()}")
        min_date = min(df.index.min() for df in self.rawData.values())
        print(f"Minimum date after mergedData: {min_date}")
            
        # Ensure the workingData DataFrame is sorted by date index
        self.workingData.sort_index(inplace=True)
        
        #only keep the dates are are in self.dataDict.keys()
        
    
    def getRanks(self, ticker):
        """Get ranks for a specific ticker based on the rankedDict"""
        # Create a new Series with the same index as self.workingData
        signal_series = pd.Series(0, index=self.workingData.index)
        #print the mimimum date of the signal_series
        print(f"Minimum date of signal_series: {signal_series.index.min()}")
        
        # Iterate through the date in the index of signal_series
        for date in signal_series.index:
            if date in self.rankedDict:
                rank = self.rankedDict[date].get(ticker, 0)
                if 1 <= rank <= self.window:
                    signal_series[date] = 1
                elif len(self.rankedDict[date]) - self.window < rank <= len(self.rankedDict[date]):
                    signal_series[date] = -1
                else:
                    signal_series[date] = 0
            else:
                signal_series[date] = 0
        return signal_series          
    
    def calculateMetrics(self):
        """Calculate strategy-specific metrics and signals"""
        #the signal for each ticker will be in the column {ticker}_signal in the self.mergedData DataFrame
        #for each ticker, if the value for that date in rankedData is between 1 and 10, then the signal is 1
        #for each ticker if the value for that date in rankedData is between number of Columns in self.rankedDf and 
        #number of Columns in self.rankedDf - 10, then the signal is -1
        min_date = min(df.index.min() for df in self.rawData.values())
        print(f"Minimum date before calculateMetrics: {min_date}")
        
        self.mergedData()
        print(f"Minimum date of workingData before calculateMetrics: {self.workingData.index.min()}")
        for ticker in self.rawData.keys():
            # Create a new column for the signal
            self.workingData[f"{ticker}_signal"] = 0
            self.workingData[f"{ticker}_signal"] = self.getRanks(ticker)
            #shift the symbol signal by 1 to avoid lookahead bias
            self.workingData[f"{ticker}_signal"] = self.workingData[f"{ticker}_signal"].shift(1)
        
        min_date = min(df.index.min() for df in self.rawData.values())
        print(f"Minimum date after filtration: {min_date}")
        #print the mimimum date of the workingData DataFrame
        #calculate the minimum date of the workingData DataFrame
        print(f"Minimum date of workingData after calculateMetrics: {self.workingData.index.min()}")
            
            
            #for each row put in the ran
    
    def addSignalsToDataFrames(self):
        """Add generated signals to individual asset DataFrames"""
        # First ensure all indexes are unique
        self.workingData = self.workingData[~self.workingData.index.duplicated(keep='last')]
        
        for ticker, df in self.rawData.items():
            # Ensure current DF has unique index
            df_clean = df[~df.index.duplicated(keep='last')]
            
            # Get intersection of dates
            common_dates = df_clean.index.intersection(self.workingData.index)
            
            # Get signals only for existing common dates
            if not common_dates.empty and f'{ticker}_signal' in self.workingData.columns:
                signal_series = self.workingData.loc[common_dates, f'{ticker}_signal']
                
                # Add to a copy to avoid SettingWithCopyWarning
                df_clean = df_clean.copy()
                df_clean['signal'] = signal_series
                
                # Drop NA signals and update rawData
                df_clean.dropna(subset=['signal'], inplace=True)
                self.rawData[ticker] = df_clean
                print(f"✅ Signals added to {ticker} DataFrame.")
            else:
                print(f"⚠️ No signals available for {ticker}")
                
        # Verify dates after processing
        min_date = min(df.index.min() for df in self.rawData.values() if not df.empty)
        print(f"Minimum date after adding signals: {min_date}")
        
    def calculateRollingSharpe(self, returns, window=90, risk_free_rate=0.0):
        excess_returns = returns - risk_free_rate
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = excess_returns.rolling(window).std()
        return (rolling_mean / rolling_std) * np.sqrt(252)  # 
        
    def calculateBacktestMetrics(self):
        all_returns = {}
        
        for ticker in self.tickerResults:
            tickerEquityCurve = self.tickerResults[ticker]['_equity_curve'] ## is a dataframe
            tickerEquityCurve['Return']= tickerEquityCurve['Equity'].pct_change().fillna(0)
            # tickerEquityCurve['rolling_sharpe_90'] = self.calculateRollingSharpe(
            #                                     tickerEquityCurve['Return'],
            #                                     window=90,
            #                                 )
            
            all_returns[ticker] = tickerEquityCurve['Return']
            
        returns_df = pd.DataFrame(all_returns)

        # Step 3: Fill missing values with 0 (for dates before IPO)
        returns_df.fillna(0, inplace=True)

        # Step 4: Calculate strategy returns (sum across tickers)
        strategy_daily_returns = returns_df.sum(axis=1)
        print("the daily returns of the strategy")
        print(strategy_daily_returns)
        strategy_rolling_sharpe = self.calculateRollingSharpe(
            strategy_daily_returns,
            window=90,
            risk_free_rate=0.0
        )

        # Step 5: Store results in the object
        self.strategy_daily_returns = strategy_daily_returns
        self.strategy_rolling_sharpe = strategy_rolling_sharpe
        
        pg.upload_dataframe(
            strategy_daily_returns.to_frame(name='strategy_daily_returns'),
            'EBITDA_StrategyDailyReturns',
            if_exists='replace'
        )
            
    
    def executeBacktests(self):
        min_date = min(df.index.min() for df in self.rawData.values())
        print(f"Minimum date before backtest: {min_date}")
        """Execute backtests for all assets"""
        startTime = time.perf_counter()
        for ticker, df in self.rawData.items():
            if 'signal' in df.columns:
                #minimum date of the df DataFrame
                if df.empty or df['signal'].isnull().all():
                    print(f"❌ No valid data for backtest on {ticker}. Skipping...")
                    continue
                print(f"Mimimum date of {ticker} DataFrame: {df.index.min()}")
                bt = Backtest(df, SignalStrategy, cash=10_000, commission=.002)
                stats, heatmap = bt.optimize(
                    stopLoss = range(1, 5, 1),
                    takeProfit = range(12, 25, 1),
                    maximize='Sharpe Ratio',
                    return_heatmap=True,
                    max_tries= 50,
                )
                self.backTests[ticker] = bt
                self.tickerResults[ticker] = stats
                print(f"✅ Backtest executed for {ticker}.")
        endTime = time.perf_counter()
        self.BackTestTime = endTime - startTime
        print(f"Total backtest execution time: {self.BackTestTime:.2f} seconds")
        
    def executeMultiProcessedBacktests(self):
        min_date = min(df.index.min() for df in self.rawData.values())
        print(f"Minimum date before backtest: {min_date}")
        
        start_time = time.perf_counter()
        
        # Create list of tickers to process
        tickers_to_process = []
        for ticker, df in self.rawData.items():
            if 'signal' not in df.columns or df.empty or df['signal'].isnull().all():
                print(f"⚠️ Skipping {ticker} - no signals")
                continue
            tickers_to_process.append(ticker)
        
        # Calculate safe number of workers based on available memory
        available_mem = psutil.virtual_memory().available / (1024 ** 3)  # GB
        df_size = sum(df.memory_usage().sum() for df in self.rawData.values()) / (1024 ** 3)
        safe_workers = max(1, min(
            len(tickers_to_process),
            os.cpu_count() // 2,  # Use half of logical cores
            int(available_mem / (df_size * 1.5))  # Memory-based limit
        ))
        print(f"Using {safe_workers} workers (memory safe: {available_mem:.2f}GB available, "
              f"data size: {df_size:.2f}GB)")
        
        # Process in batches to avoid memory overload
        batch_size = safe_workers * 2
        results = {}
        
        for i in range(0, len(tickers_to_process), batch_size):
            batch = tickers_to_process[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(tickers_to_process)-1)//batch_size + 1}: "
                  f"{len(batch)} tickers")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers) as executor:
                futures = {}
                for ticker in batch:
                    # Pass minimal data to worker
                    df = self.rawData[ticker][['Open', 'High', 'Low', 'Close', 'Volume', 'signal']].copy()
                    future = executor.submit(
                        run_backtest_worker, 
                        ticker, 
                        df,
                        self.__class__.__name__  # Pass strategy name
                    )
                    futures[future] = ticker
                
                for future in concurrent.futures.as_completed(futures):
                    ticker = futures[future]
                    try:
                        stats = future.result()
                        if stats is not None:
                            results[ticker] = stats
                            print(f"✅ Backtest completed for {ticker}")
                    except Exception as e:
                        print(f"❌ Backtest failed for {ticker}: {str(e)}")
            
            # Explicit cleanup between batches
            del futures
            gc.collect()
        
        # Store results
        for ticker, stats in results.items():
            self.tickerResults[ticker] = stats
        
        end_time = time.perf_counter()
        self.BacktestTime = end_time - start_time
        print(f"Total multiprocessed backtest time: {self.BacktestTime:.2f}s")

    # Worker function defined at module level for pickling
    



combinedDataDict = getCombinedDataDict()      
print(combinedDataDict.keys())
valStr = EbitdaValueStrategy(combinedDataDict)
valStr.createProcessedData()
valStr.calculateMetrics()
valStr.addSignalsToDataFrames()
valStr.executeMultiProcessedBacktests()
valStr.calculateBacktestMetrics()