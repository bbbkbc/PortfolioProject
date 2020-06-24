import math
import pandas as pd
import datetime
import numpy as np


# load needed data
trade_history = pd.read_csv('trade_history.csv', index_col=0)
symbol_ticker = pd.read_csv('symbol_ticker.csv', index_col=0)
# quick check how data looks like
# print(trade_history.head())
# print(symbol_ticker.head())


def data_preparation(df_trades=trade_history, df_st=symbol_ticker):
    # change format of date_time then split date_time column to date and time
    # and drop date_time column
    df_trades['date_time'] = pd.to_datetime(df_trades['date_time'])
    df_trades["date"] = df_trades['date_time'].dt.date
    df_trades['time'] = df_trades['date_time'].dt.time
    df_trades = df_trades.drop(columns=['date_time'])
    # set position of columns
    df_trades = df_trades[['date', 'time', 'symbol',
                           'ticker', 'site', 'num_of_share',
                           'stock_price', 'value']]
    # set new column with commission values,
    # in this example rules about commission are quiet simple
    # min commission equal 3, commision_lvl equal 0,0039 from value of transaction
    df_trades['commission'] = round(df_trades['value'] * 0.0039, ndigits=2)
    # if commission is below 3 then set new value which is 3
    df_trades.loc[df_trades['commission'] < 3, 'commission'] = 3
    # swap ticker values which are the same as in symbol column into correct
    # for this purpose i use file which store this values
    for i in df_trades.index:
        st_mask = df_st.symbol == df_trades.symbol.iloc[i]
        df_trades.at[i, 'ticker'] = df_st.loc[st_mask].iat[0, 1]

    return df_trades


def portfolio_preparation(df_trades=pd.DataFrame, evaluation_day="2020-06-10"):
    # create new empty dataframe
    df_pf = pd.DataFrame()
    # add unique ticker for each stock which is in trade history
    df_pf['ticker'] = df_trades['ticker'].drop_duplicates()
    # below is a loop which help to build a portfolio
    # add a mask which will help to filter data
    # for each stock in portfolio depends on their site
    for t in df_pf.ticker:
        buy_mask = (df_trades['site'] == 'K') & (df_trades['ticker'] == t)
        sell_mask = (df_trades['site'] == 'S') & (df_trades['ticker'] == t)
        df_pf.loc[df_pf['ticker'] == t, 'mean_buy'] = df[buy_mask].stock_price.mean()
        df_pf.loc[df_pf['ticker'] == t, 'buy_comm'] = df[buy_mask].commission.sum()
        df_pf.loc[df_pf['ticker'] == t, 'buy_share_sum'] = df[buy_mask].num_of_share.sum()
        df_pf.loc[df_pf['ticker'] == t, 'mean_sell'] = df[sell_mask].stock_price.mean()
        df_pf.loc[df_pf['ticker'] == t, 'sell_comm'] = df[sell_mask].commission.sum()
        df_pf.loc[df_pf['ticker'] == t, 'sell_share_sum'] = df[sell_mask].num_of_share.sum()
    # create new columns with contain value of living position
    # settled pnl as well portfolio value based on buy_price
    df_pf['shares_actual'] = df_pf['buy_share_sum'] - df_pf['sell_share_sum']
    df_pf['pnl_closed'] = df_pf['sell_share_sum'] * (df_pf['mean_sell'] - df_pf['mean_buy'])
    df_pf['value_at_open'] = df_pf['shares_actual'] * df_pf['mean_buy']
    # loop below will add to df_pf new column with closing price of assets which are in portfolio
    for ticker in df_pf.ticker:
        # get data from database to every stock in portfolio
        mkt_data = pd.read_csv(f'./mkt_data/{ticker}.csv')
        # get closing price from mkt data and set this price into specified row/column in df_pf
        date_mask = mkt_data['Date'] == evaluation_day
        df_pf.loc[df_pf['ticker'] == ticker, 'mkt_close_price'] = mkt_data[date_mask].iat[0, 4]
    # adding new columns with pnl_live, and value_now which is showing worth of shares
    # in portfolio
    df_pf['pnl_live'] = df_pf['shares_actual'] * (df_pf['mkt_close_price'] - df_pf['mean_buy'])
    df_pf['value_now'] = df_pf['value_at_open'] + df_pf['pnl_live']
    return df_pf


def portfolio_analysis(df_pf=pd.DataFrame):
    # for start, some simple sum up
    v_0 = df_pf.value_at_open.sum()
    v_1 = df_pf.value_now.sum()
    v_2 = df_pf.pnl_closed.sum()
    v_3 = df_pf.pnl_live.sum()
    v_4 = sum(df_pf.buy_comm + df_pf.sell_comm)
    print(f'PNL live: {v_3:.2f}, PNL settled: {v_2:.2f}, PNL total: {v_3 + v_2:.2f}')
    print(f'Total transaction costs: {v_4:.2f}')
    print(f'Portfolio value at open: {v_0:.2f}')
    print(f'Portfolio value now: {v_1:.2f}')
    print(f'Open position performance: {((v_1/v_0 - 1) * 100):.2f}%')
    print(f'Total performance after costs: {(((v_1 + v_2 + v_4) / v_0 - 1) * 100):.2f}%')


if __name__ == '__main__':
    df = data_preparation(trade_history, symbol_ticker)
    eval_day = "2020-06-23"
    portfolio = portfolio_preparation(df, eval_day)
    portfolio_analysis(portfolio)
