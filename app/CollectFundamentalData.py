import os
import requests
import pandas as pd
#import env
from dotenv import load_dotenv
load_dotenv()
def get_quarterly_data(url_base, ticker, api_key, period):
    params = {
        'symbol': ticker,
        'period': period,
        'limit': 20,
        'apikey': api_key,
    }
    full_url = f"{url_base}?{'&'.join([f'{key}={value}' for key, value in params.items()])}"
    response = requests.get(full_url)
    if response.status_code == 200:
        try:
            df = pd.DataFrame(response.json())
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['period'] = period
            return df
        except ValueError:
            print(f"JSON decoding failed for {url_base} - {period}")
            return pd.DataFrame()
    else:
        print(f"Request failed for {url_base} - {period}")
        return pd.DataFrame()


def GetAllFundamentalData(ticker, api_key):
    income_url = "https://financialmodelingprep.com/stable/income-statement"
    balance_url = "https://financialmodelingprep.com/stable/balance-sheet-statement"
    cashflow_url = "https://financialmodelingprep.com/stable/cash-flow-statement"
    
    all_income, all_balance, all_cash = [], [], []

    for period in ['Q1', 'Q2', 'Q3', 'Q4']:
        income_df = get_quarterly_data(income_url, ticker, api_key, period)
        balance_df = get_quarterly_data(balance_url, ticker, api_key, period)
        cashflow_df = get_quarterly_data(cashflow_url, ticker, api_key, period)

        if not (income_df.empty or balance_df.empty or cashflow_df.empty):
            # Align columns before merge
            for df in [income_df, balance_df, cashflow_df]:
                df.set_index('date', inplace=True)

            # Inner merge to keep only matching dates across all statements
            merged = income_df.join(balance_df, how='inner', lsuffix='_income', rsuffix='_balance')
            merged = merged.join(cashflow_df, how='inner', rsuffix='_cashflow')

            merged.reset_index(inplace=True)
            merged['period'] = period
            all_income.append(merged)

    if all_income:
        final_df = pd.concat(all_income, ignore_index=True)
        final_df.sort_values(by='date', inplace=True)

        return final_df
    else:
        print("No combined data found for given periods.")
        return pd.DataFrame()

# Example usage
ticker = "AAPL"
api_key = os.getenv('api_key')  # Replace with your actual API key
df = GetAllFundamentalData(ticker, api_key)
print(df.head())