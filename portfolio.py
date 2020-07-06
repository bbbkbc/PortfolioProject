import math
import pandas as pd
import datetime
from datetime import datetime as dt
import holidays
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Pastel1_7, Dark2_8
import historical_data as hd


# quick check how data looks like
# print(trade_history.head())
# print(symbol_ticker.head())


def data_preparation(df_trades, df_st, portfolio_date="1990-01-01"):
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
    # min commission equal 3, commission_lvl equal 0,0039 from value of transaction
    df_trades['commission'] = round(df_trades['value'] * 0.0039, ndigits=2)
    # if commission is below 3 then set new value which is 3
    df_trades.loc[df_trades['commission'] < 3, 'commission'] = 3
    # swap ticker values which are the same as in symbol column into correct
    # for this purpose i use file which store this values
    for i in df_trades.index:
        st_mask = df_trades.symbol.iloc[i] == df_st.symbol
        df_trades.loc[i, 'ticker'] = df_st[st_mask].iat[0, 1]
    # this functionality allow to filter portfolio by date
    if portfolio_date != "1990-01-01":
        portfolio_date = pd.to_datetime(portfolio_date).date()
        df_trades = df_trades[df_trades['date'] <= portfolio_date]
    return df_trades


def portfolio_preparation(df_trades, sym_tik, evaluation_day="2020-06-10"):
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
        df_pf.loc[df_pf['ticker'] == t, 'mean_buy'] = df_trades[buy_mask].stock_price.mean()
        df_pf.loc[df_pf['ticker'] == t, 'buy_comm'] = df_trades[buy_mask].commission.sum()
        df_pf.loc[df_pf['ticker'] == t, 'buy_share_sum'] = df_trades[buy_mask].num_of_share.sum()
        df_pf.loc[df_pf['ticker'] == t, 'mean_sell'] = df_trades[sell_mask].stock_price.mean()
        df_pf.loc[df_pf['ticker'] == t, 'sell_comm'] = df_trades[sell_mask].commission.sum()
        df_pf.loc[df_pf['ticker'] == t, 'sell_share_sum'] = df_trades[sell_mask].num_of_share.sum()
    # create new columns with contain value of living position
    # settled pnl as well portfolio value based on buy_price
    df_pf['shares_actual'] = df_pf['buy_share_sum'] - df_pf['sell_share_sum']
    df_pf['pnl_closed'] = df_pf['sell_share_sum'] * (df_pf['mean_sell'] - df_pf['mean_buy'])
    df_pf['value_at_open'] = df_pf['shares_actual'] * df_pf['mean_buy']
    # loop below will add to df_pf new column with closing price of assets which are in portfolio

    # check if database is updated if not update
    mkt = pd.read_csv(f'./mkt_data/TPE.csv')
    last_date = pd.to_datetime(mkt.iloc[-1, 0]).date()
    e_day = pd.to_datetime(evaluation_day)
    if e_day > last_date:
        hd.data_download(sym_tik, end=evaluation_day)

    ed = evaluation_day
    for ticker in df_pf.ticker:
        # get data from database to every stock in portfolio
        mkt_data = pd.read_csv(f'./mkt_data/{ticker}.csv')
        # get closing price from mkt data and set this price into specified row/column in df_pf
        date_mask = mkt_data['Date'] == evaluation_day
        # check if each ticker has in this day any data if not, set last available date
        if len(mkt_data[date_mask]) == 0:
            # print(f'{ticker} empty data {evaluation_day}')
            evaluation_day = mkt_data[mkt_data['Date'] <= str(evaluation_day)].iloc[-1, 0]
            date_mask = mkt_data['Date'] == evaluation_day
            # print(f'Set a new date {evaluation_day}')
        df_pf.loc[df_pf['ticker'] == ticker, 'mkt_close_price'] = mkt_data[date_mask].iat[0, 4]
        evaluation_day = ed

    # adding new columns with pnl_live, and value_now which is showing worth of shares in portfolio
    df_pf['pnl_live'] = df_pf['shares_actual'] * (df_pf['mkt_close_price'] - df_pf['mean_buy'])
    df_pf['value_now'] = df_pf['value_at_open'] + df_pf['pnl_live']
    return df_pf


def portfolio_analysis(df_pf, pparam=False, v_param=False):
    """
    :param df_pf: as input should be data frame taken from portfolio preparation function
    :param pparam: if want to print out and check output set as TRUE default is False
    :param v_param: if want to change returned data into v params change on True
    :return: if v_param False function return list of strings with calculated summary data,
    :parameters if v_param True:
    v_0 = df_pf.value_at_open.sum()
    v_1 = df_pf.value_now.sum()
    v_2 = df_pf.pnl_closed.sum()
    v_3 = df_pf.pnl_live.sum()
    v_4 = sum(df_pf.buy_comm + df_pf.sell_comm)
    """
    # for start, simple summarize
    v_0 = df_pf.value_at_open.sum()
    v_1 = df_pf.value_now.sum()
    v_2 = df_pf.pnl_closed.sum()
    v_3 = df_pf.pnl_live.sum()
    v_4 = sum(df_pf.buy_comm + df_pf.sell_comm)
    if v_param:
        return [v_0, v_1, v_2, v_3, v_4]
    # summarize of portfolio:
    pnl_sum = f'PNL live: {v_3:.2f}, PNL settled: {v_2:.2f}, PNL total: {v_3 + v_2:.2f}'
    costs = f'Total transaction costs: {v_4:.2f}'
    value_open = f'Portfolio value at open: {v_0:.2f}'
    value_now = f'Portfolio value now: {v_1:.2f}'
    open_per = f'Open position performance: {((v_1 / v_0 - 1) * 100):.2f}%'
    total_per = f'Total performance after costs: {(((v_1 + v_2 + v_4) / v_0 - 1) * 100):.2f}%'
    if pparam:
        print(pnl_sum)
        print(costs)
        print(value_open)
        print(value_now)
        print(open_per)
        print(total_per)
    return [pnl_sum, costs, value_open, value_now, open_per, total_per]


def visualization(df_pf, p_composition='donut', p=None):
    portfolio_analysis(df_pf, p)
    data = df_pf[['ticker', 'value_at_open', 'pnl_live', 'pnl_closed']]
    data = data.drop(data[data.value_at_open == 0].index)
    data = data.fillna(0)
    # plotting portfolio composition for existing exposure in different styles
    y_pos = np.arange(len(data.ticker))
    if p_composition == 'donut':
        my_circle = plt.Circle((0, 0), 0.50, color='white')
        plt.pie(data.value_at_open,
                labels=data.ticker,
                colors=Pastel1_7.hex_colors,
                autopct='%1.1f%%',
                pctdistance=0.9)
        p = plt.gcf()
        p.gca().add_artist(my_circle)
        plt.show()
    elif p_composition == 'barplot':
        plt.bar(y_pos, data.value_at_open, color=Dark2_8.hex_colors)
        plt.xticks(y_pos, data.ticker)
        plt.show()
    # this one present initial value for living exposure with stacked settled pnl and live pnl
    elif p_composition == 'stacked':
        data["bars"] = data.value_at_open + data.pnl_closed
        # Create brown bars
        plt.bar(y_pos, data.value_at_open,
                color='#6cd4d2', edgecolor='white')
        # Create green bars (middle), on top of the firs ones
        plt.bar(y_pos, data.pnl_closed,
                bottom=data.value_at_open, color='#3547cc', edgecolor='white')
        # Create green bars (top)
        plt.bar(y_pos, data.pnl_live,
                bottom=data.bars, color='#335d80', edgecolor='white')
        plt.xticks(y_pos, data.ticker)
        plt.show()
    return data


def pnl_analysis(trade_history, symbol_ticker, start="2020-04-24", end="2020-07-02", show_chart=False, benchmark=False):
    pl_holidays = holidays.PL()
    st = pd.to_datetime(start).date()
    ed = pd.to_datetime(end).date()
    lst_bd = []
    pkl_check = pd.read_pickle('pf_pnl.pkl')
    pkl_last_date = pkl_check.date.max()
    if pkl_last_date >= ed:
        start_date_mask = pkl_check.date >= st
        end_date_mask = pkl_check.date <= ed
        df_pnl = pkl_check[start_date_mask & end_date_mask]
    else:
        for i in range(int((ed - st).days) + 1):
            bd = st + datetime.timedelta(i)
            if bd in pl_holidays or bd.weekday() > 4:
                continue
            lst_bd.append(bd)
        date_lst = []
        pnl_lst = []
        val_open_lst = []
        val_now_lst = []
        for x in lst_bd:
            pf_data = portfolio_preparation(data_preparation(trade_history, symbol_ticker, x), symbol_ticker, x)
            pnl_l = pf_data.pnl_live.sum()
            pnl_c = pf_data.pnl_closed.sum()
            val_open = pf_data.value_at_open.sum()
            val_now = pf_data.value_now.sum()
            # print(f'DATE:{x} | PNL TOTAL: {pnl_l + pnl_c:.2f}')
            date_lst.append(pd.to_datetime(x).date())
            pnl_lst.append(pnl_l + pnl_c)
            val_open_lst.append(val_open)
            val_now_lst.append(val_now)
        df_pnl = pd.DataFrame(list(zip(date_lst, pnl_lst, val_open_lst, val_now_lst)),
                              columns=['date', 'pnl_total', 'val_open_lst', 'val_now_lst'])
        pd.to_pickle(df_pnl, 'pf_pnl.pkl')
    # condition below will modify df_pnl and return dataframe which compare daily retrun on portfolio and wig20
    if benchmark:
        df_pnl['%_change_cumulative'] = (((df_pnl.val_open_lst + df_pnl.pnl_total) / df_pnl.val_open_lst) - 1) * 100
        df_pnl['%_1_day_shift'] = abs(df_pnl['%_change_cumulative'].shift(1).fillna(0))
        df_pnl['%_daily_change'] = df_pnl['%_change_cumulative'] - df_pnl['%_1_day_shift']
        benchmark_symbol = 'WIG20'
        benchmark_data = pd.read_csv(f'./mkt_data/{benchmark_symbol}.csv')
        benchmark_data = benchmark_data[['Date', 'Close']]
        # benchmark_data['Date'] = pd.to_datetime(benchmark_data['Date'])
        benchmark_data['Date'] = pd.to_datetime(benchmark_data['Date']).dt.date
        last_date = pd.to_datetime(benchmark_data.iloc[-1, 0]).date()
        if ed > last_date:
            hd.data_download(benchmark_symbol, end=ed)
        benchmark_data = benchmark_data.loc[benchmark_data.Date >= (st - datetime.timedelta(1))]
        benchmark_data['1d_shift'] = benchmark_data['Close'].shift(1).fillna(0)
        benchmark_data['%1d_change'] = ((benchmark_data['Close'] / benchmark_data['1d_shift']) - 1) * 100
        benchmark_data = benchmark_data.loc[(benchmark_data.Date >= st) & (benchmark_data.Date <= ed)]
        benchmark_data['%_change_cumulative'] = benchmark_data['%1d_change'].cumsum()
        return [df_pnl, benchmark_data]
    if show_chart:
        plt.plot(df_pnl.date, df_pnl.pnl_total, color='#8f9805')
        plt.show()
    return df_pnl


if __name__ == '__main__':
    # load needed data
    trade_history = pd.read_csv('trade_history.csv', index_col=0)
    symbol_ticker = pd.read_csv('symbol_ticker.csv', index_col=0)
    hist_portfolio = "2020-04-25"
    df = data_preparation(trade_history, symbol_ticker)
    eval_day = "2020-04-25"
    portfolio = portfolio_preparation(df, symbol_ticker, eval_day)
    data_pnl = pnl_analysis(trade_history, symbol_ticker, end='2020-07-03', show_chart=False, benchmark=True)
    print(data_pnl[0])
    print(data_pnl[1])
    vis_d = visualization(portfolio, p_composition=None, p=False)

    # pnl_analysis(trade_history, symbol_ticker, show_chart=True)
