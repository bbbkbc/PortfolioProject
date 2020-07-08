import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import holidays
import datetime
import plotly.graph_objects as go
from portfolio import data_preparation as data_prep
from portfolio import portfolio_preparation as pf_prep


class RiskAnalysis:
    history_df = pd.read_csv('trade_history.csv', index_col=0)
    symbol_df = pd.read_csv('symbol_ticker.csv', index_col=0)

    def __init__(self, start_date='2020-04-24', end_date='2020-07-06', eval_date='2020-05-15',
                 number_days_var=15, show_plot=False, histogram_ticker='TPE'):
        self.start_date = start_date
        self.end_date = end_date
        self.eval_date = eval_date
        self.df_data = data_prep(self.history_df, self.symbol_df, self.eval_date)
        self.df_pf = pf_prep(self.df_data, self.symbol_df, self.eval_date)
        self.number_days_var = number_days_var
        self.show_plot = show_plot
        self.histogram_ticker = histogram_ticker

    def preprocessing(self):
        portfolio = self.df_pf
        initial_investment = portfolio.value_at_open.sum()
        portfolio = portfolio[['ticker', 'value_now']]
        portfolio['weights'] = portfolio['value_now'] / initial_investment
        portfolio = portfolio[portfolio['value_now'] != 0]
        portfolio.reset_index(inplace=True)
        return [portfolio[['ticker', 'weights']], initial_investment]

    def get_data(self):
        """
        Function takes data from database, and create one dataframe with tickers as columns, and date as index
        :return: dataframe, every single nan values are replaced by previous one
        """
        tickers = self.preprocessing()[0].ticker
        df = pd.DataFrame()
        for ticker in tickers:
            closing_data = pd.read_csv(f'./mkt_data/{ticker}.csv')[['Date', 'Close']]
            end_mask = closing_data['Date'] <= self.eval_date
            closing_data = closing_data[end_mask]
            # history in range 30days before eval day
            closing_data = closing_data.iloc[-30:]
            closing_data.reset_index(inplace=True)
            closing_data['Date'] = pd.to_datetime(closing_data['Date'])
            closing_data.set_index('Date', inplace=True)
            df[ticker] = round(closing_data.Close, ndigits=4)
        df = df.fillna(method='ffill')
        return df

    def portfolio_var(self):
        # from the closing prices, calculate periodic returns
        returns = self.get_data().pct_change()
        # create covariance matrix based on returns
        cov_matrix = returns.cov()
        # calculate mean returns for each stock in portfolio
        average_ret = returns.mean()
        # normalize individual means against investment weights
        weights = np.array(self.preprocessing()[0].weights)
        portfolio_mean = average_ret.dot(weights)
        # Calculate portfolio standard deviation
        portfolio_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
        # Calculate mean of investment
        mean_investment = (1 + portfolio_mean) * self.preprocessing()[1]
        # Calculate standard deviation of investment
        stdev_investment = self.preprocessing()[1] * portfolio_stdev
        # Select our confidence interval
        confidence_level = 0.05
        cutoff = norm.ppf(confidence_level, mean_investment, stdev_investment)
        var_1d = self.preprocessing()[1] - cutoff
        return [var_1d, returns, portfolio_mean, portfolio_stdev]

    def n_day_var(self):
        var_array = []
        num_days = int(self.number_days_var)
        var_1 = self.portfolio_var()[0]
        print_lst = []
        for x in range(1, num_days + 1):
            var_array.append(np.round(var_1 * np.sqrt(x), 2))
            print_lst.append(f'{str(x)} day VaR @ 95 % confidence: {str(np.round(var_1 * np.sqrt(x), 2))}')

        if self.show_plot:
            [print(x) for x in print_lst]
            plt.xlabel('Day')
            plt.ylabel('Max portfolio loss (PLN)')
            plt.title(f'Max portfolio loss (VaR) over {self.number_days_var}-day period')
            plt.plot(var_array, 'r')
            plt.show()
        return var_array

    def histograms_var(self):
        tik = self.histogram_ticker
        var_components = self.portfolio_var()
        returns = var_components[1]
        port_mean = var_components[2]
        port_stdev = var_components[3]
        returns[tik].hist(bins=20, density=True, histtype='stepfilled', alpha=0.4)
        x = np.linspace(port_mean - 3 * port_stdev, port_mean + 3 * port_stdev, 100)
        plt.plot(x, norm.pdf(x, port_mean, port_stdev), 'r')
        plt.title(f'{tik} returns vs normal distribution')
        plt.show()

    def var_3d_surface(self):
        # iteration for only working date in range of start and end date
        holidays_PL = holidays.PL()
        start_dt = pd.to_datetime(self.start_date).date()
        end_dt = pd.to_datetime(self.end_date).date()
        date_lst = []
        var_3d_frame = pd.DataFrame(columns=[f'{n + 1}_day' for n in range(self.number_days_var)])
        for i in range(int((end_dt - start_dt).days) + 1):
            bd = start_dt + datetime.timedelta(i)
            if bd in holidays_PL or bd.weekday() > 4:
                continue
            date_lst.append(bd)
        var_3d_frame['date'] = pd.to_datetime(date_lst)
        var_3d_frame.set_index('date', inplace=True)
        for x in date_lst:
            reload = RiskAnalysis(eval_date=str(x), number_days_var=self.number_days_var, show_plot=False)
            var_3d_frame.loc[x] = reload.n_day_var()
        fig = go.Figure(data=[go.Surface(z=var_3d_frame.values)])
        fig.update_layout(title='VaR surface', autosize=False,
                          width=800, height=800,)
        return [var_3d_frame, fig]

    def print_date(self):
        print(f'start: {self.start_date}, end: {self.end_date}, eval: {self.eval_date}')
        print(f'data frame:\n {self.preprocessing()[0]}')
        print(f'init invest:\n {self.preprocessing()[1]}')
        print(f'ticker date:\n {self.get_data()}')
        print(f'1D VAR: \n {self.portfolio_var()}')
        print(f'date: {self.var_3d_surface()}')


if __name__ == '__main__':
    date = RiskAnalysis(start_date='2020-05-01', end_date='2020-05-31', eval_date='2020-05-31',
                        number_days_var=20, show_plot=False)
    date.var_3d_surface()[1].show()

