# some_file.py
from Database import PostgresManager
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import text
# Load environment variables
load_dotenv()
# Initialize PostgreSQL manager
pg = PostgresManager(
    host=os.getenv('host'),
    port=os.getenv('port'),
    dbname=os.getenv('dbname'),
    user=os.getenv('user'),
    password=os.getenv('password')
)

def getCombinedDataDict():
    
# 1. Get valid symbols from CompanyProfiles
    query = """
    SELECT symbol FROM "CompanyProfiles"
    WHERE "marketCap" IS NOT NULL AND "marketCap" > 0
    """
    symbols_df = pd.read_sql_query(text(query), pg.engine)
    tickers = symbols_df['symbol'].tolist()

    print(f"✅ Retrieved {len(tickers)} tickers with market cap > 0")

    # 2. Get valid EODData tables that exist in the database
    existing_EOD_tables = pg.get_tickers_from_EOD_tables()
    valid_EOD_tickers = [t for t in tickers if t in existing_EOD_tables]
    existing_Fundamentals_tables = pg.get_tickers_from_Fundamentals_tables()
    valid_tickers = [t for t in valid_EOD_tickers if t in existing_Fundamentals_tables]
    print(f"✅ Found {len(existing_EOD_tables)} existing EODData tables")
    print(f"✅ Found {len(existing_Fundamentals_tables)} existing FundamentalsData tables")
    print(f"✅ {len(valid_EOD_tickers)} tickers with available EODData")
    print(f"✅ {len(existing_Fundamentals_tables)} tickers with available FundamentalsData")
    print(f"✅ {len(valid_tickers)} tickers with available FundamentalsData")

    print(f"✅ {len(valid_tickers)} tickers with available EODData")



    columnsOfInterest = ['filingDate_income', 'ebitda', 'ebit', 'grossProfit', 'revenue', 'researchAndDevelopmentExpenses',
                        'costAndExpenses', 'totalCurrentLiabilities', 'weightedAverageShsOut', 'epsDiluted', 'totalLiabilities','totalAssets']

    combinedDataDict = {}
    existing_tables = pg.get_tickers_from_combined_tables()

    for ticker in valid_tickers:
        try:
            dataframeName = f"{ticker}_CombinedData"
            # Check if the combined data already exists
            
            fundamentals_df = pg.getTickerFundamentalsData(ticker)
            if fundamentals_df.empty:
                print(f"❌ No fundamentals data for {ticker}")
                continue
            
            # Filter columns of interest
            filtered_df = fundamentals_df[columnsOfInterest]
            
            # Convert 'filingDate_income' to datetime
            filtered_df['filingDate_income'] = pd.to_datetime(filtered_df['filingDate_income'], errors='coerce')
            
            # Drop rows with NaN values in 'filingDate_income'
            filtered_df.dropna(subset=['filingDate_income'], inplace=True)
            
            # Set 'filingDate_income' as index
            #filtered_df.set_index('filingDate_income', inplace=True)
            
            #getting the EOD data for the same ticker
            eod_df = pg.getTickerEODData(ticker)
            if eod_df.empty:
                print(f"❌ No EOD data for {ticker}")
                continue
            # Merge EOD data with fundamentals on date index
            # forward fill missing values when combining the two dataframes
            combined_df = eod_df.join(filtered_df, how='outer', lsuffix='_eod', rsuffix='_fundamentals')
            combined_df.ffill(inplace=True)
            combined_df.dropna(inplace=True)  # Drop rows with any NaN values after merging
            if combined_df.empty:
                print(f"❌ Combined data is empty for {ticker}")
                continue
            #the market cap is the 'weightedAverageShsOut' * 'close'
            combined_df['marketCap'] = combined_df['weightedAverageShsOut'] * combined_df['close']
            combined_df['bookValue'] = combined_df['totalAssets'] - combined_df['totalLiabilities']
            combined_df['debtToEquity'] = combined_df['totalLiabilities'] / ((combined_df['totalAssets'] - combined_df['totalLiabilities']))
            combined_df['bookToMarket'] = combined_df['bookValue'] / combined_df['marketCap']
            combined_df['enterpriseValue'] = combined_df['marketCap'] + combined_df['totalLiabilities'] - combined_df['totalAssets']
            combined_df['ebitValue'] = combined_df['ebitda'] / combined_df['enterpriseValue']
            combined_df['ebitdaValue'] = combined_df['ebitda'] / combined_df['marketCap']
            combined_df['profitValue'] = combined_df['grossProfit'] / combined_df['marketCap']
            combinedDataDict[ticker] = combined_df.copy()
            combined_df.reset_index(inplace=True)  # Reset index to have 'date' as a column
            
            pg.upload_dataframe(combined_df, dataframeName, if_exists='replace')
            print(f"✅ Processed {ticker} successfully with {len(combined_df)} rows of combined data")
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            
    return combinedDataDict