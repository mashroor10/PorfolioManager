import pandas as pd
from backtesting import Backtest, Strategy
from TradingStrategy import TradingStrategy
from SignalStrategy import SignalStrategy

class MeanReversionStrategy(TradingStrategy):
    def __init__(self, data, window=30):
        super().__init__(data, window)
        self.createProcessedData()
        
    def createProcessedData(self):
        """Preprocess raw data into working format"""
        # Rename columns to uppercase first letter for backtesting compatibility
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
            
        self.mergeData()
        self.processed = True
        
    def mergeData(self):
        """Merge individual asset data into a single working DataFrame"""
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
    
    def calculateMetrics(self):
        """
        Calculate mean reversion metrics such as spread, z-score, and Bollinger Bands.
        This method is generalized to handle any number of tickers in rawData.
        """
        if not self.processed:
            self.createProcessedData()
            
        tickers = list(self.rawData.keys())
        n = len(tickers)
        # Calculate normalized close for each ticker
        for ticker in tickers:
            self.workingData[f'{ticker}_normalizedClose'] = (
                self.workingData[f'{ticker}_Close'] - self.workingData[f'{ticker}_Close'].rolling(window=self.window).mean()
            ) / self.workingData[f'{ticker}_Close'].rolling(window=self.window).std()

        # Calculate pairwise normalized close ratios and z-scores
        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = tickers[i], tickers[j]
                ratio_col = f'{t1}_{t2}_normalizedCloseRatio'
                mean_col = f'{t1}_{t2}_rolling_mean'
                std_col = f'{t1}_{t2}_rolling_std'
                z_col = f'{t1}_{t2}_z_score'
                # Ratio
                self.workingData[ratio_col] = (
                    self.workingData[f'{t1}_normalizedClose'] / self.workingData[f'{t2}_normalizedClose']
                )
                # Rolling mean/std
                self.workingData[mean_col] = self.workingData[ratio_col].rolling(window=self.window).mean()
                self.workingData[std_col] = self.workingData[ratio_col].rolling(window=self.window).std()
                # Z-score
                self.workingData[z_col] = (
                    self.workingData[ratio_col] - self.workingData[mean_col]
                ) / self.workingData[std_col]
                self.workingData[z_col].fillna(0, inplace=True)

                # Generate signals for both tickers based on z-score
                for ticker_signal, direction in [(t1, ('sell', 'buy')), (t2, ('buy', 'sell'))]:
                    signal_col = f'{ticker_signal}_signal'
                    if signal_col not in self.workingData.columns:
                        self.workingData[signal_col] = 'hold'
                    self.workingData.loc[self.workingData[z_col] > 1.5, signal_col] = direction[0]
                    self.workingData.loc[self.workingData[z_col] < -1.5, signal_col] = direction[1]
                    self.workingData.loc[
                        (self.workingData[z_col] >= -0.9) & (self.workingData[z_col] <= 0.9), signal_col
                    ] = 'close'
                    # Shifting the signal by 1 to avoid lookahead bias
                    self.workingData[signal_col] = self.workingData[signal_col].shift(1)
        
        # Drop rows with NaN values in any of the calculated columns
        self.workingData.dropna(inplace=True)
        
    def addSignalsToDataFrames(self):
        """
        Add signals to each ticker's DataFrame based on the calculated metrics.
        This method is generalized to handle any number of tickers in workingData.
        """
        tickers = list(self.rawData.keys())
        for ticker in tickers:
            signal_col = f'{ticker}_signal'
            if signal_col not in self.workingData.columns:
                self.workingData[signal_col] = 'hold'
            # Add the signal to the rawData DataFrame for the appropriate ticker
            self.rawData[ticker][signal_col] = self.workingData[signal_col]
            # Remove rows with NaN signals for this ticker in rawData
            self.rawData[ticker].dropna(subset=[signal_col], inplace=True)
            # Replace the buy/sell/close or hold signals with 1/-1/0
            self.rawData[ticker][signal_col] = self.rawData[ticker][signal_col].replace({
                'buy': 1,
                'sell': -1,
                'close': 0,
                'hold': 0
            })
            # Rename the signal column to 'signal' for consistency
            self.rawData[ticker].rename(columns={signal_col: 'signal'}, inplace=True)
            
    def executeBacktests(self):
        """Execute backtests for all assets using optimized parameters"""
        self.tickerResults = {}
        self.backTests = {}
        for ticker in self.rawData.keys():
            print(f"Running backtest for {ticker}...")
            df = self.rawData[ticker].copy()
            
            bt = Backtest(
                df,
                SignalStrategy,
                cash=10_000,
                commission=.002
            )
            
            stats, heatmap = bt.optimize(
                stopLoss = range(1, 5, 1),
                takeProfit = range(12, 20, 1),
                maximize='Sharpe Ratio',
                return_heatmap=True,
                max_tries=50,
            )
            self.backTests[ticker] = bt
            self.tickerResults[ticker] = stats