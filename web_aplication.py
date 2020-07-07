import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import datetime
from portfolio import portfolio_preparation as pp
from portfolio import data_preparation as dp
from portfolio import portfolio_analysis
from portfolio import pnl_analysis
from dash.exceptions import PreventUpdate

th = pd.read_csv('trade_history.csv', index_col=0)
st = pd.read_csv('symbol_ticker.csv', index_col=0)
pf_data = pnl_analysis(trade_history=th, symbol_ticker=st, end='2020-07-06')

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
page_1_layout = html.Div(
    children=[
        dbc.Row([
            dbc.Col(
                html.H4('Set date:'), width='auto', className='CERULEAN'
            ),
            dbc.Col([
                dcc.DatePickerSingle(
                    id='calendar',
                    clearable=True,
                    with_portal=True,
                    date=datetime.datetime(2020, 6, 30).date(),
                    display_format='Y-MM-DD'),
            ], width=5),
            dbc.Col(html.H4('Delta range:'), width='auto', className='CERULEAN'),
            dbc.Col([
                dcc.DatePickerRange(
                    id='delta-range',
                    minimum_nights=1,
                    clearable=True,
                    with_portal=True,
                    start_date=datetime.datetime(2020, 4, 24).date(),
                    end_date=datetime.datetime(2020, 7, 2).date(),
                    display_format='Y-MM-DD'
                ),
            ]),
        ]),
        html.Br(),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Portfolio Summary:", className="card-title"),
                    dbc.CardBody(id='sum-data'),
                ],
                    color='primary', inverse=True, ),
            ]),
            dbc.Col([
                dbc.Table(id='delta-data',
                          bordered=True,
                          hover=True)
            ]),
        ]),
        dbc.Row([
            dbc.Col(id='val-chart')
        ]),
    ])


@app.callback(Output(component_id='sum-data', component_property='children'),
              [Input(component_id='calendar', component_property='date')])
def portfolio_sum(date):
    df = dp(th, st, date)
    portfolio = pp(df, st, date)
    p_summary = portfolio_analysis(portfolio)

    return html.Div([html.P(p_summary[0]),
                     html.P(p_summary[1]),
                     html.P(p_summary[2]),
                     html.P(p_summary[3]),
                     html.P(p_summary[4]),
                     html.P(p_summary[5]),
                     ])


@app.callback(Output(component_id='delta-data', component_property='children'),
              [Input(component_id='delta-range', component_property='start_date'),
               Input(component_id='delta-range', component_property='end_date')])
def portfolio_delta(start_date, end_date):
    df_start = dp(th, st, start_date)
    portfolio_start = pp(df_start, st, start_date)
    ps_start = portfolio_analysis(portfolio_start, v_param=True)
    df_end = dp(th, st, end_date)
    portfolio_end = pp(df_end, st, end_date)
    ps_end = portfolio_analysis(portfolio_end, v_param=True)

    table_header = [html.Thead(html.Tr([html.Th("Indicator"),
                                        html.Th(f"Value at {start_date}"),
                                        html.Th(f"Value at {end_date}"),
                                        html.Th(f"Delta")]))]
    row1 = html.Tr([html.Td("PNL LIVE"),
                    html.Td(f"{ps_start[3]:.2f}"),
                    html.Td(f"{ps_end[3]:.2f}"),
                    html.Td(f"{ps_end[3] - ps_start[3]:.2f}")])
    row2 = html.Tr([html.Td("PNL CLOSED"),
                    html.Td(f"{ps_start[2]:.2f}"),
                    html.Td(f"{ps_end[2]:.2f}"),
                    html.Td(f"{ps_end[2] - ps_start[2]:.2f}")])
    row3 = html.Tr([html.Td("PNL TOTAL"),
                    html.Td(f"{ps_start[3] + ps_start[2]:.2f}"),
                    html.Td(f"{ps_end[3] + ps_end[2]:.2f}"),
                    html.Td(f"{(ps_end[3] + ps_end[2]) - (ps_start[3] + ps_start[2]):.2f}")])
    row4 = html.Tr([html.Td("TOTAL COST"),
                    html.Td(f"{ps_start[4]:.2f}"),
                    html.Td(f"{ps_end[4]:.2f}"),
                    html.Td(f"{ps_end[4] - ps_start[4]:.2f}")])
    row5 = html.Tr([html.Td("P VAL OPEN"),
                    html.Td(f"{ps_start[0]:.2f}"),
                    html.Td(f"{ps_end[0]:.2f}"),
                    html.Td(f"{ps_end[0] - ps_start[0]:.2f}")])
    row6 = html.Tr([html.Td("P VAL NOW"),
                    html.Td(f"{ps_start[1]:.2f}"),
                    html.Td(f"{ps_end[1]:.2f}"),
                    html.Td(f"{ps_end[1] - ps_start[1]:.2f}")])
    row7 = html.Tr([html.Td("OPEN RETURN"),
                    html.Td(f"{((ps_start[1] / ps_start[0] - 1) * 100):.2f}%"),
                    html.Td(f"{((ps_end[1] / ps_end[0] - 1) * 100):.2f}%"),
                    html.Td(f"{(((ps_end[1] / ps_end[0] - 1) - (ps_start[1] / ps_start[0] - 1)) * 100):.2f}%")])
    row8 = html.Tr([html.Td("TOTAL RETURN"),
                    html.Td(f"{(((ps_start[1] + ps_start[2] + ps_start[4]) / ps_start[0] - 1) * 100):.2f}%"),
                    html.Td(f"{(((ps_end[1] + ps_end[2] + ps_end[4]) / ps_end[0] - 1) * 100):.2f}%"),
                    html.Td(f"""{((((ps_end[1] + ps_end[2] + ps_end[4]) / ps_end[0] - 1) -
                                   ((ps_start[1] + ps_start[2] + ps_start[4]) / ps_start[0] - 1)) * 100):.2f}%""")])
    table_body = [html.Tbody([row1, row2, row3, row4, row5, row6, row7, row8])]
    return table_header + table_body


@app.callback(Output(component_id='val-chart', component_property='children'),
              [Input(component_id='delta-range', component_property='start_date'),
               Input(component_id='delta-range', component_property='end_date')])
def value_chart(start_date, end_date):
    st_date = pd.to_datetime(start_date).date()
    ed_date = pd.to_datetime(end_date).date()
    df_val = pnl_analysis(th, st, start=st_date, end=ed_date)
    df_val = df_val[['date', 'val_open_lst', 'val_now_lst', 'pnl_total']]
    graphs = dcc.Graph(
        config={'displaylogo': False},
        figure={'data': [
            {'x': df_val.date, 'y': df_val.val_open_lst, 'type': 'line', 'name': 'Value at Open'},
            {'x': df_val.date, 'y': df_val.val_now_lst, 'type': 'line', 'name': 'Value Now'},
            {'x': df_val.date, 'y': df_val.pnl_total, 'type': 'bar', 'name': 'PNL TOTAL'},
        ],
            'layout': {'title': f'Value change in range from {start_date} to {end_date}',
                       'height': 600}
        },
    )
    return graphs


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
    dcc.DatePickerRange(
        id='delta-range',
        minimum_nights=1,
        clearable=True,
        with_portal=True,
        start_date=datetime.datetime(2020, 4, 24).date(),
        end_date=datetime.datetime(2020, 7, 2).date(),
        display_format='Y-MM-DD'
    ),
    html.Hr(),
    html.Br(),
    html.H2('Total PnL performance'),
    dcc.Graph(id='Total Pnl',
              figure={'data': [{'x': pf_data.date,
                                'y': pf_data.pnl_total,
                                'type': 'line',
                                'name': 'PNL'
                                }],
                      'layout': {'title': 'Total PNL in Nominal Values'}}),
    html.Hr(),
    html.Br(),
    html.Div(id='benchmark-compare'),
])


@app.callback(Output(component_id='benchmark-compare', component_property='children'),
              [Input(component_id='delta-range', component_property='start_date'),
               Input(component_id='delta-range', component_property='end_date')])
def benchmark(start, end):
    prime_data = pnl_analysis(th, st, start=start, end=end, benchmark=True)
    portfolio_data = prime_data[0]
    benchmark_data = prime_data[1]
    return html.Div([
        dcc.Graph(
            id='benchmark-w20-port',
            figure={'data': [{'x': benchmark_data.Date, 'y': benchmark_data['%_change_cumulative'],
                              'type': 'line', 'name': 'WIG20 CUMULATIVE %'},
                             {'x': portfolio_data.date, 'y': portfolio_data['%_change_cumulative'],
                              'type': 'line', 'name': 'PORTFOLIO CUMULATIVE %'},
                             {'x': benchmark_data.Date, 'y': benchmark_data['%1d_change'],
                              'type': 'bar', 'name': 'WIG20 DAILY'},
                             {'x': portfolio_data.date, 'y': portfolio_data['%_daily_change'],
                              'type': 'bar', 'name': 'PORTFOLIO DAILY'}],
                    'layout': {'title': 'Return Portfolio vs Wig20',
                               'height': 600,
                               'legend': {'orientation': 'h', 'y': 1.02},
                               }
                    }),
    ])


# page 4 content
page_4_layout = html.Div(children=[
    html.Div(children='Symbol to graph:'),
    dcc.Input(id='symbol', value='MBK', type='text'),
    html.Div(children='Set starting date:'),
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
    fig.update_layout(title=stock, xaxis_rangeslider_visible=False, height=600)
    return dcc.Graph(figure=fig)


page_5_layout = html.Div([
    dbc.Row(dbc.Col(
        dbc.Alert('Here you can add a new transaction to your portfolio', color='info'),
        width={'size': 6, 'offset': 3})),
    html.Br(),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Div('NEW TRANSACTION'),
            html.Br(),
            html.Div('SITE (K/S)'),
            dcc.Input(id='input-0-state', value='K', type='text'),
            html.Div('Date:'),
            dcc.Input(id='input-1-state', value='2020-06-24 14:05:29', type='text'),
            html.Div('Symbol:'),
            dcc.Input(id='input-2-state', value='PLAY', type='text'),
            html.Div('Shares:'),
            dcc.Input(id='input-3-state', value='200', type='text'),
            html.Div('Price'),
            dcc.Input(id='input-4-state', value='29.54', type='text'),
            html.Div('Cash'),
            dcc.Input(id='input-5-state', value='800', type='text'),
            html.Div([
                html.Br(),
                dbc.Button('ADD TRADE', id='button-state', n_clicks=0, color='primary', className='CERULEAN')
            ]),
        ], width=3),
        dbc.Col([
            html.Div(id='output-state')]),
    ]),
])


@app.callback(Output('output-state', 'children'),
              [Input('button-state', 'n_clicks')],
              [State('input-0-state', 'value'),
               State('input-1-state', 'value'),
               State('input-2-state', 'value'),
               State('input-3-state', 'value'),
               State('input-4-state', 'value'),
               State('input-5-state', 'value')])
def update_output(n_clicks, input0, input1, input2, input3, input4, input5):
    df = pd.read_csv('trade_history.csv', index_col=0)
    if n_clicks == 0:
        raise PreventUpdate
    else:
        # df = pd.read_csv('trade_history.csv', index_col=0)
        # columns['date_time', 'symbol', 'ticker', 'site', 'num_of_share', 'stock_price', 'value']
        df.loc[-1] = [f'{input1}', f'{input2}', f'{input2}', f'{input0}', int(input3), float(input4), float(input5)]
        df.index = df.index + 1
        df = df.sort_index()
        df.to_csv('trade_history.csv')
        # (n_clicks, input1, input2, input3, input4, input5)
        return html.Div([
            dbc.Alert(f'Transaction added to the database, you added {n_clicks} new transactions', color='primary'),
            dbc.Table.from_dataframe(df.head(7), bordered=True, dark=True)
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
        return page_1_layout
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
