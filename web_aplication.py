import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import datetime
from portfolio import portfolio_preparation as pp
from portfolio import data_preparation as dp
from portfolio import pnl_analysis

th = pd.read_csv('trade_history.csv', index_col=0)
st = pd.read_csv('symbol_ticker.csv', index_col=0)
# pf_data = pnl_analysis(trade_history=th, symbol_ticker=st)
# pf_data.to_pickle('pf_pnl.pkl')
pf_data = pd.read_pickle('pf_pnl.pkl')

app = dash.Dash(external_stylesheets=[dbc.themes.CERULEAN])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "14rem",
    "padding": "1rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "15rem",
    "margin-right": "0rem",
    "padding": "4rem 2rem",
}

sidebar = html.Div(
    [
        html.H2("Portfolio Management"),
        html.Hr(),
        html.P("This app is to help you make better investment decisions"),
        dbc.Nav(
            [
                dbc.NavLink("Summary", href="/page-1", id="page-1-link"),
                dbc.NavLink("Portfolio Composition", href="/page-2", id="page-2-link"),
                dbc.NavLink("PNL", href="/page-3", id="page-3-link"),
                dbc.NavLink("Stock Charts", href="/page-4", id="page-4-link"),
                dbc.NavLink("Transactions", href="/page-5", id="page-5-link"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# page 1 content - in this section app will show summary about portfolio

# page 2 content - here are data related with portfolio composition
page_2_layout = html.Div(
    children=[
        dbc.Row([
            dbc.Col(
                html.H4('Set evaluation date:'), width='auto', className='CERULEAN'
            ),
            dbc.Col([
                dcc.DatePickerSingle(
                    id='calendar',
                    clearable=True,
                    with_portal=True,
                    date=datetime.datetime(2020, 4, 24).date(),
                    display_format='Y-MM-DD'),
            ]),
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.Div(id='portfolio-pie'),
            ]),
            dbc.Col([
                html.Div(id='portfolio-bar'),
            ]),
        ]),
    ])


@app.callback(Output(component_id='portfolio-pie', component_property='children'),
              [Input(component_id='calendar', component_property='date')])
def portfolio_pie(date):
    date_value = pd.to_datetime(date).date()
    first_day = pd.to_datetime(th.date_time.min()).date()
    today = datetime.datetime.now().date()
    if first_day <= date_value < today:
        result = dbc.Alert(f'You picked {date}', color='info')
    else:
        # if date is out of range set date_value as a first_day in transaction history
        date_value = first_day
        result = dbc.Alert(f"Please set the date which is in range between {first_day} and {today}",
                           color="warning")
    dt_day = str(date_value)
    data_prep = dp(th, st, dt_day)
    data_pp = pp(data_prep, st, dt_day)
    data = data_pp[['ticker', 'value_at_open', 'pnl_live', 'pnl_closed']]
    data = data.drop(data[data.value_at_open == 0].index)
    data = data.fillna(0)
    fig = go.Figure(data=[go.Pie(labels=data.ticker, values=data.value_at_open)])
    fig.update_layout(title_text='Portfolio Structure by Value')
    return html.Div([dcc.Graph(figure=fig), result])


@app.callback(Output(component_id='portfolio-bar', component_property='children'),
              [Input(component_id='calendar', component_property='date')])
def portfolio_bar(date):
    date_value = pd.to_datetime(date).date()
    first_day = pd.to_datetime(th.date_time.min()).date()
    today = datetime.datetime.now().date()
    if first_day <= date_value < today:
        result = dbc.Alert(f'You picked {date}', color='info')
    else:
        # if date is out of range set date_value as a first_day in transaction history
        date_value = first_day
        result = dbc.Alert(f"Please set the date which is in range between {first_day} and {today}",
                           color="warning")
    dt_day = str(date_value)
    data_prep = dp(th, st, dt_day)
    data_pp = pp(data_prep, st, dt_day)
    data = data_pp[['ticker', 'value_at_open', 'pnl_live', 'pnl_closed']]
    data = data.fillna(0)
    x = data.ticker
    y_1 = data.pnl_closed
    y_2 = data.pnl_live
    return html.Div([
        dcc.Graph(
            id='portfolio-pnl',
            figure={
                'data': [
                    {'x': x, 'y': y_1, 'type': 'bar', 'name': 'Pnl Closed'},
                    {'x': x, 'y': y_2, 'type': 'bar', 'name': 'Pnl Live'}
                ],
                'layout': {
                    'barmode': 'stack',
                    'title': 'Portfolio Structure by PNL',
                }
            },
        ),
        result
    ])


# page 3 content
page_3_layout = html.Div(children=[
    html.H1('Total PnL performance'),
    dcc.Graph(id='Total Pnl',
              figure={'data': [{'x': pf_data.date,
                                'y': pf_data.pnl_total,
                                'type': 'line',
                                'name': 'PNL'
                                }],
                      'layout': {'title': 'Total PNL overtime'}})
])
# page 4 content
page_4_layout = html.Div(children=[
    html.Div(children='symbol to graph:'),
    dcc.Input(id='symbol', value='MBK', type='text'),
    html.Div(children='set starting date:'),
    dcc.Input(id='start_date', value='2020-03-03', type='text'),
    html.Div(id='output_graph'),
])


@app.callback(
    Output(component_id='output_graph', component_property='children'),
    [Input(component_id='symbol', component_property='value'),
     Input(component_id='start_date', component_property='value')])
def graph(symbol, start_date):
    stock = symbol
    start = start_date
    df = pd.read_csv(f'mkt_data/{stock}.csv', index_col=0)
    df = df[start:]
    candlestick = go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close)
    fig = go.Figure(data=[candlestick])
    fig.update_layout(title=stock, xaxis_rangeslider_visible=False)
    return dcc.Graph(figure=fig)


# page 4 content
page_5_layout = html.Div([
    dbc.Row(dbc.Col(dbc.Alert('Here you can add a new transation to your portfolio', color='info'),
                    width={'size': 6, 'offset': 3})),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.Div("New Transaction"),
            html.Br(),
            html.Div('Date:'),
            dbc.Input(id='date_time', placeholder='2020-06-24 14:05:29', type='text'),
            html.Div('Symbol:'),
            dbc.Input(id='symbol', placeholder='PLAY', type='text'),
            html.Div('Shares number:'),
            dbc.Input(id='num_of_share', placeholder='200', type='text'),
            html.Div('Price:'),
            dbc.Input(id='stock_price', placeholder='29.54', type='text'),
            html.Div('Value:'),
            dbc.Input(id='value', placeholder='800', type='text'),
            html.Br(),
            dbc.Button('ADD TRADE', id='new-trade', n_clicks=0, color='primary', className='CERULEAN'),
        ],
            width=3),
        dbc.Col(html.Div("One of three columns")),
    ])
])


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 6)],
    [Input("url", "pathname")], )
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 6)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        return html.P('Summary')
    elif pathname == "/page-2":
        return page_2_layout
    elif pathname == "/page-3":
        return page_3_layout
    elif pathname == "/page-4":
        return page_4_layout
    elif pathname == "/page-5":
        return page_5_layout

    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server(port=8888)
