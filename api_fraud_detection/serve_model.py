from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

# Load the trained model
model = joblib.load("gb_e-commerce_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='fraud_detection.log', level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    
    # Log the request data
    logging.info(f"Incoming request data: {data}")
    
    try:
        # Convert JSON data to DataFrame
        input_data = pd.DataFrame(data, index=[0])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Log the prediction
        logging.info(f"Prediction: {int(prediction[0])}")
        
        # Return prediction as JSON
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        # Log any errors
        logging.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)