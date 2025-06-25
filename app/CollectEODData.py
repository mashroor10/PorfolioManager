import requests
import time
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from Database import PostgresManager

load_dotenv()

def getEODData(apiKey, ticker, end):
    def get_ipo_date(ticker, api_key):
        url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json()
                ipo_date = data[0].get('ipoDate')
                if ipo_date:
                    return ipo_date
            except (KeyError, IndexError, ValueError):
                print(f"⚠️ Failed to parse IPO date for {ticker}")
        else:
            print(f"⚠️ Failed to fetch IPO info. Status code: {response.status_code}")
        return None

    # Initialize rate limiter state
    rate_limiter = {
        'start_time': time.time(),
        'call_count': 0
    }
    
    def rate_limited_get(url, params):
        """Make API calls while respecting the rate limit"""
        nonlocal rate_limiter
        
        # Calculate time since last reset
        elapsed = time.time() - rate_limiter['start_time']
        
        # Handle rate limiting
        if rate_limiter['call_count'] >= 290:  # Using 290 for safety buffer
            if elapsed < 60:
                # Calculate precise wait time
                wait_time = 60.01 - elapsed  # Add 10ms buffer
                print(f"⏳ API limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            
            # Reset counter after waiting
            rate_limiter['start_time'] = time.time()
            rate_limiter['call_count'] = 0
            print("♻️ Rate limit counter reset")
        
        # Make the API call
        rate_limiter['call_count'] += 1
        print(f"📞 API call #{rate_limiter['call_count']} (Elapsed: {elapsed:.2f}s)")
        finalurl =  f"{url}?{'&'.join([f'{key}={value}' for key, value in params.items()])}"
        print(f"🔗 Request URL: {finalurl}")
        
        return requests.get(finalurl)
    
    def pullData(ticker, apiKey, start, end, period = 'daily'):
        #putting a switch case for 30mins and daily data url
        if period == '30min':
            base_url = "https://financialmodelingprep.com/stable/historical-chart/30min"
        elif period == 'daily':
            base_url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
        else:
            raise ValueError("Invalid period. Use '30mins' or 'daily'.")
        
        
        params = {
            'to': end,
            'from': start,
            'symbol': ticker,
            'apikey': apiKey,
        }
        
        response = rate_limited_get(base_url, params=params)
        
        # Check for successful response
        if response.status_code != 200:
            print(f"⚠️ Error {response.status_code} for {start} to {end}")
        print(f"📥 Downloaded data for {ticker} from {start} to {end} ({period})")
            
        data = response.json()
        #converting the data to a pandas dataframe
        df = pd.DataFrame(data)
        #getting the earliest and latest date from the data
        if len(df) == 0:
            print(f"No data found for {ticker} from {start} to {end}.")
            return pd.DataFrame(), start
        earliestDate = df['date'].min()
        latestDate = df['date'].max()
        
        print(f"📅 Earliest date: {earliestDate}, Latest date: {latestDate}")
        
        #dropping the high and low columns if they exist
        if 'high' in df.columns and 'low' in df.columns:
            df = df.drop(columns=['high', 'low'])
        #drop 'change', 'changePercent' and 'vwap' columns if they exist
        if 'change' in df.columns and 'changePercent' in df.columns and 'vwap' in df.columns:
            df = df.drop(columns=['change', 'changePercent', 'vwap'])
            print("Dropped 'change', 'changePercent', and 'vwap' columns.")
            print("the length of the dataframe is: ", len(df))
        
        #getting the earliest Date and removing the time part
        if period == '30min':
            earliestDate = earliestDate.split(' ')[0]
        #converting the date to a datetime object
        earliestDate = datetime.strptime(earliestDate, '%Y-%m-%d').strftime('%Y-%m-%d')
        
        #returning the dataframe and the earliest date
        return df, earliestDate
    
    
    
    #the data is stored in a pandas dataframe
    df = pd.DataFrame()
    
    
    
    #converting the start and end dates to datetime strings with date and hour format %Y-%m-%d
    # Get IPO date for the ticker and use it as the start date
    ipoDate = get_ipo_date(ticker, apiKey)
    if ipoDate is None:
        raise ValueError(f"Could not retrieve IPO date for {ticker}")
        return None
    
    start = datetime.strptime(ipoDate, '%Y-%m-%d')
    end = datetime.strptime(end, '%Y-%m-%d')
    
    #remove the hour component from the start and end dates
    start = start.strftime('%Y-%m-%d')
    end = end.strftime('%Y-%m-%d')
    
    print(f"Getting data for {ticker} from {start} to {end}")
    
    currentEnd = []
        
        
        
    #the current end is the end with %Y-%m-%d format
    currentEnd = end
    while start not in currentEnd:
        #get the data for the current date
        currentData, earliestDate = pullData(ticker, apiKey, start=start, end=currentEnd, period = 'daily')
        
        # If earliestDate is the same as currentEnd, set currentEnd to 15:30:00 of the day before earliestDate
        if earliestDate in currentEnd:
            prev_day = datetime.strptime(earliestDate, '%Y-%m-%d') - timedelta(days=1)
            currentEnd = prev_day.strftime('%Y-%m-%d')
        else:
            currentEnd = earliestDate
        print("the current end is: ", currentEnd)
        df = pd.concat([df, currentData], ignore_index=True)
        
    #sort the dataframe by date
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    #remove duplicates based on date
    df = df.drop_duplicates(subset='date')
    #push it to the postgres database
    return df
    
ticker = "SPY"
api_key = os.getenv("api_key")

data = getEODData(api_key, ticker,end = "2025-06-06")
print(data.tail())

pg = PostgresManager(
    host=os.getenv('host'),
    port=os.getenv('port'),
    dbname=os.getenv('dbname'),
    user=os.getenv('user'),
    password=os.getenv('password')
)

pg.upload_dataframe(data, f"{ticker}_EOD_Data", if_exists='replace')