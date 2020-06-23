import requests
import pandas as pd
import os
from sys import path


def data_download(data=pd.DataFrame, start='2010-01-01', end='2020-06-22'):
    data_path = os.path.join(path[0], 'mkt_data/')
    os.makedirs(data_path, exist_ok=True)
    for ticker in data.ticker:
        start = start.replace("-", "")
        end = end.replace("-", "")
        download_url = f'https://stooq.com//q/d/l/?s={ticker}&d1={start}&d2={end}&i=d'
        req = requests.get(download_url)
        url_content = req.content
        csv_file = open(f'{data_path}{ticker}.csv', 'wb')
        csv_file.write(url_content)
        csv_file.close()
        print(f'{ticker} downloaded')


if __name__ == '__main__':
    # symbol_ticker is just a file with list of name and shortcut stocks which are in portfolio
    portfolio_symbols = pd.read_csv('symbol_ticker.csv')
    start_date = "2010-01-01"
    end_date = "2020-06-22"
    data_download(portfolio_symbols, start_date, end_date)
