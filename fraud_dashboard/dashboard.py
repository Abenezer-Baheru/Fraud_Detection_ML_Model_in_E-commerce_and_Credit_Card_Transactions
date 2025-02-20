from dash import dcc, html
import dash
import plotly.express as px
import requests
import pandas as pd
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Fetch summary data from Flask API
summary_data = requests.get('http://flask-api:5000/api/transactions/summary').json()
fraud_trends = requests.get('http://flask-api:5000/api/trends/fraud').json()
fraud_geography = requests.get('http://flask-api:5000/api/geography/fraud').json()
fraud_devices = requests.get('http://flask-api:5000/api/devices/fraud').json()
fraud_browsers = requests.get('http://flask-api:5000/api/browsers/fraud').json()

# Create summary boxes
summary_boxes = html.Div([
    dbc.Row([
        dbc.Col(html.Div([
            html.H4('Total Transactions'),
            html.P(summary_data['total_transactions'])
        ])),
        dbc.Col(html.Div([
            html.H4('Fraud Cases'),
            html.P(summary_data['fraud_cases'])
        ])),
        dbc.Col(html.Div([
            html.H4('Fraud Percentage'),
            html.P(f"{summary_data['fraud_percentage']:.2f}%")
        ]))
    ])
])

# Create line chart for fraud trends
fraud_trends_df = pd.DataFrame(fraud_trends)
line_chart = dcc.Graph(
    figure=px.line(fraud_trends_df, x='day_of_week', y='is_fraud', title='Fraud Cases Over Time')
)

# Create bar chart for fraud geography
fraud_geography_df = pd.DataFrame(fraud_geography)
geo_chart = dcc.Graph(
    figure=px.bar(fraud_geography_df, x='country', y='is_fraud', title='Fraud Cases by Country')
)

# Create bar chart for fraud devices
fraud_devices_df = pd.DataFrame(fraud_devices)
device_chart = dcc.Graph(
    figure=px.bar(fraud_devices_df, x='transaction_velocity', y='is_fraud', title='Fraud Cases by Transaction Velocity')
)

# Create bar chart for fraud browsers
fraud_browsers_df = pd.DataFrame(fraud_browsers)
browser_chart = dcc.Graph(
    figure=px.bar(fraud_browsers_df, x='hour_of_day', y='is_fraud', title='Fraud Cases by Hour of Day')
)

# Layout for the dashboard
app.layout = html.Div([
    html.H1('Fraud Detection Dashboard'),
    summary_boxes,
    line_chart,
    geo_chart,
    device_chart,
    browser_chart
])

if __name__ == '__main__':
    app.run_server(debug=True)