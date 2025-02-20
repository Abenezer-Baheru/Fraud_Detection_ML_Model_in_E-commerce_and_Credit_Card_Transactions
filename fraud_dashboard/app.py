from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Load fraud data from CSV file
fraud_data = pd.read_csv('data/e-commerce_processed_data.csv')

@app.route('/api/transactions/summary', methods=['GET'])
def transactions_summary():
    total_transactions = len(fraud_data)
    fraud_cases = fraud_data[fraud_data['is_fraud'] == 1]
    fraud_count = len(fraud_cases)
    fraud_percentage = (fraud_count / total_transactions) * 100

    return jsonify({
        'total_transactions': total_transactions,
        'fraud_cases': fraud_count,
        'fraud_percentage': fraud_percentage
    })

@app.route('/api/trends/fraud', methods=['GET'])
def fraud_trends():
    fraud_cases_over_time = fraud_data.groupby('day_of_week')['is_fraud'].sum().reset_index()

    return jsonify(fraud_cases_over_time.to_dict(orient='records'))

@app.route('/api/geography/fraud', methods=['GET'])
def fraud_geography():
    fraud_by_location = fraud_data.groupby('country')['is_fraud'].sum().reset_index()

    return jsonify(fraud_by_location.to_dict(orient='records'))

@app.route('/api/devices/fraud', methods=['GET'])
def fraud_devices():
    fraud_by_device = fraud_data.groupby('transaction_velocity')['is_fraud'].sum().reset_index()

    return jsonify(fraud_by_device.to_dict(orient='records'))

@app.route('/api/browsers/fraud', methods=['GET'])
def fraud_browsers():
    fraud_by_browser = fraud_data.groupby('hour_of_day')['is_fraud'].sum().reset_index()

    return jsonify(fraud_by_browser.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)