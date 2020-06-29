import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from portfolio import pnl_analysis
from portfolio import portfolio_preparation
from portfolio import data_preparation

th = pd.read_csv('trade_history.csv', index_col=0)
st = pd.read_csv('symbol_ticker.csv', index_col=0)
# date = '2020-06-23'
# df_trade = data_preparation(th, st, date)
# pf_table = portfolio_preparation(df_trade, st, date)
pf_data = pnl_analysis(trade_history=th, symbol_ticker=st)

app = dash.Dash()
app.layout = html.Div(children=[
    html.H1('Total PnL performance'),
    dcc.Graph(id='Total Pnl',
              figure={'data': [{'x': pf_data.date,
                                'y': pf_data.pnl_total,
                                'type': 'line',
                                'name': 'PNL'
                                }],
                      'layout': {'title': 'Total PNL overtime'}})
])

if __name__ == '__main__':
    app.run_server(debug=True)


