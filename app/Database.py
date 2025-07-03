import pandas as pd
import re
from sqlalchemy import create_engine, inspect

class PostgresManager:
    def __init__(self, host, port, dbname, user, password):
        self.db_params = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password
        }
        self.engine_str = (
            f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        )
        self.engine = create_engine(self.engine_str)
        print("✅ PostgreSQL connection initialized.")

    def upload_dataframe(self, df: pd.DataFrame, table_name: str, if_exists='replace'):
        """
        Uploads a DataFrame to PostgreSQL.
        - if_exists: 'replace', 'append', or 'fail'
        """
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False, method='multi')
            print(f"✅ Data uploaded to table '{table_name}'.")
        except Exception as e:
            print(f"❌ Failed to upload to '{table_name}': {e}")
    
    def getTicker30MinData(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves 30-minute interval data for a given ticker from PostgreSQL
        """
        table_name = f"{ticker.upper()}_30MinData"
        try:
            query = f"SELECT * FROM \"{table_name}\""
            df = pd.read_sql_query(query, self.engine)
            
            # Convert 'date' column to datetime and set as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                print(f"✅ Retrieved {len(df)} rows of 30-min data for {ticker}")
            else:
                print(f"❌ 'date' column missing in table {table_name}")
            return df
        except Exception as e:
            print(f"❌ Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame()
    def getTickerEODData(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves EOD data for a given ticker from PostgreSQL
        """
        table_name = f"{ticker.upper()}_EOD_Data"
        try:
            query = f"SELECT * FROM \"{table_name}\""
            df = pd.read_sql_query(query, self.engine)
            
            # Convert 'date' column to datetime and set as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                print(f"✅ Retrieved {len(df)} rows of EOD data for {ticker}")
            else:
                print(f"❌ 'date' column missing in table {table_name}")
            return df
        except Exception as e:
            print(f"❌ Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame()
    
    def getTickerFundamentalsData(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves fundamentals data for a given ticker from PostgreSQL
        """
        table_name = f"{ticker.upper()}_FundamentalsData"
        try:
            query = f"SELECT * FROM \"{table_name}\""
            df = pd.read_sql_query(query, self.engine)
            
            # Convert 'date' column to datetime and set as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                print(f"✅ Retrieved {len(df)} rows of fundamentals data for {ticker}")
            else:
                print(f"❌ 'date' column missing in table {table_name}")
            return df
        except Exception as e:
            print(f"❌ Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame()
    def getTickerCombinedData(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves combined data for a given ticker from PostgreSQL
        """
        table_name = f"{ticker.upper()}_CombinedData"
        try:
            query = f"SELECT * FROM \"{table_name}\""
            df = pd.read_sql_query(query, self.engine)
            
            # Convert 'date' column to datetime and set as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                print(f"✅ Retrieved {len(df)} rows of combined data for {ticker}")
            else:
                print(f"❌ 'date' column missing in table {table_name}")
            return df
        except Exception as e:
            print(f"❌ Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame()
        
    def get_tickers_from_30min_tables(self):
        """
        Extracts all tickers from tables that match the {ticker}_30MinData format.
        """
        try:
            inspector = inspect(self.engine)
            all_tables = inspector.get_table_names()
            pattern = re.compile(r'^(.*)_30MinData$', re.IGNORECASE)
            tickers = [match.group(1).upper() for table in all_tables if (match := pattern.match(table))]
            return tickers
        except Exception as e:
            print(f"❌ Failed to inspect tables: {e}")
            return []
    def get_tickers_from_EOD_tables(self):
        """
        Extracts all tickers from tables that match the {ticker}_30MinData format.
        """
        try:
            inspector = inspect(self.engine)
            all_tables = inspector.get_table_names()
            pattern = re.compile(r'^(.*)_EOD_Data$', re.IGNORECASE)
            tickers = [match.group(1).upper() for table in all_tables if (match := pattern.match(table))]
            return tickers
        except Exception as e:
            print(f"❌ Failed to inspect tables: {e}")
            return []
        
    def get_tickers_from_Fundamentals_tables(self):
        """
        Extracts all tickers from tables that match the {ticker}_Fundamentals format.
        """
        try:
            inspector = inspect(self.engine)
            all_tables = inspector.get_table_names()
            pattern = re.compile(r'^(.*)_FundamentalsData$', re.IGNORECASE)
            tickers = [match.group(1).upper() for table in all_tables if (match := pattern.match(table))]
            return tickers
        except Exception as e:
            print(f"❌ Failed to inspect tables: {e}")
            return []
    
    def get_tickers_from_combined_tables(self):
        try:
            inspector = inspect(self.engine)
            all_tables = inspector.get_table_names()
            pattern = re.compile(r'^(.*)_CombinedData$', re.IGNORECASE)
            tickers = [match.group(1).upper() for table in all_tables if (match := pattern.match(table))]
            return tickers
        except Exception as e:
            print(f"❌ Failed to inspect tables: {e}")
            return []
        
    def getLastDateInfo(self, ticker: str):
        """
        Returns a dictionary with the last available date for each data type for the given ticker.
        Example:
        {
            "30MinData": "2024-06-10 15:30:00",
            "EOD_Data": "2024-06-10",
            "FundamentalsData": "2024-03-31",
            "CombinedData": "2024-06-10"
        }
        If a table does not exist or is empty, the value will be None.
        """
        result = {}
        table_types = {
            "EOD_Data": f"{ticker.upper()}_EOD_Data",
            "FundamentalsData": f"{ticker.upper()}_FundamentalsData",
            "CombinedData": f"{ticker.upper()}_CombinedData"
        }
        for key, table_name in table_types.items():
            try:
                query = f'SELECT MAX("date") as last_date FROM "{table_name}"'
                df = pd.read_sql_query(query, self.engine)
                last_date = df["last_date"].iloc[0] if not df.empty else None
                result[key] = str(last_date) if pd.notnull(last_date) else None
            except Exception:
                result[key] = None
        return result
    
    def extendEODData(self, ticker: str, new_data: pd.DataFrame):
        """
        Extends the EOD data for a given ticker with new data.
        Assumes new_data has 'date' column in datetime format.
        """
        table_name = f"{ticker.upper()}_EOD_Data"
        try:
            new_data.to_sql(table_name, self.engine, if_exists='append', index=False, method='multi')
            print(f"✅ Extended EOD data for {ticker} with {len(new_data)} new rows.")
        except Exception as e:
            print(f"❌ Failed to extend EOD data for {ticker}: {e}")
            
    def extendFundamentalsData(self, ticker: str, new_data: pd.DataFrame):
        """
        Extends the fundamentals data for a given ticker with new data.
        Assumes new_data has 'date' column in datetime format.
        """
        table_name = f"{ticker.upper()}_FundamentalsData"
        try:
            new_data.to_sql(table_name, self.engine, if_exists='append', index=False, method='multi')
            print(f"✅ Extended fundamentals data for {ticker} with {len(new_data)} new rows.")
        except Exception as e:
            print(f"❌ Failed to extend fundamentals data for {ticker}: {e}")
            
    def extendCombinedData(self, ticker: str, new_data: pd.DataFrame):
        """
        Extends the combined data for a given ticker with new data.
        Assumes new_data has 'date' column in datetime format.
        """
        table_name = f"{ticker.upper()}_CombinedData"
        try:
            new_data.to_sql(table_name, self.engine, if_exists='append', index=False, method='multi')
            print(f"✅ Extended combined data for {ticker} with {len(new_data)} new rows.")
        except Exception as e:
            print(f"❌ Failed to extend combined data for {ticker}: {e}")
        
        