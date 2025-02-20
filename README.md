Got it! Let's update the `README.md` to reflect the current structure and usage. Here's the updated version:

# Fraud Detection ML Model in E-commerce and Credit Card Transactions

## Project Overview
This project focuses on enhancing fraud detection for e-commerce transactions and bank credit transactions. The goal is to clean the dataset, perform feature engineering, analyze the data, build and evaluate machine learning models, and deploy the models for real-time fraud detection.

## Folder Structure
The project directory is structured as follows:
```
├── notebooks/
    ├── EDA_and_Preprocessing/
        ├── creditcard_EDA_and_Preprocessing.ipynb
        ├── e-commerce_EDA_and_Preprocessing.ipynb
    ├── models/
        ├── creditcard_model_withClassBalance.ipynb
        ├── e-commerce_model_withClassBalance.ipynb
        ├── e-commerce_model_withoutClassBalance.ipynb
    ├── __init__.py
    └── README.md
├── src/
    ├── data/
        ├── creditcard.csv
        ├── Fraud_Data.csv
        ├── IpAddress_to_Country.csv
        ├── creditcard_processed_data.csv
        ├── e-commerce_processed_data.csv
    ├── __init__.py
├── tests/
    ├── test_creditcard_EDA_and_Preprocessing.py
    ├── __init__.py
├── api_fraud_detection/
    ├── Dockerfile
    ├── gb_e-commerce_model.pkl
    ├── requirements.txt
    ├── serve_model.py
├── fraud_dashboard/
    ├── app.py
    ├── dashboard.py
    ├── docker-compose.yml
    ├── Dockerfile
    ├── requirements.txt
    ├── __init__.py
    └── README.md
├── .gitignore
├── requirements.txt
├── README.md
```

## Setup and Installation

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/Abenezer-Baheru/Fraud_Detection_ML_Model_in_E-commerce_and_Credit_Card_Transactions
    ```

2. Navigate to the project directory:
    ```bash
    cd Fraud_Detection_ML_Model_in_E-commerce_and_Credit_Card_Transactions
    ```

3. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Analysis and Preprocessing

### Running the Notebooks

1. Open the `notebooks/EDA_and_Preprocessing/creditcard_EDA_and_Preprocessing.ipynb` notebook in Jupyter:
    ```bash
    jupyter notebook notebooks/EDA_and_Preprocessing/creditcard_EDA_and_Preprocessing.ipynb
    ```

2. Execute the cells in the notebook step-by-step. The notebook performs the analysis and preprocessing tasks.

## Running the Unit Tests

1. Navigate to the `tests` directory:
    ```bash
    cd tests
    ```

2. Run the unit tests:
    ```bash
    python -m unittest test_creditcard_EDA_and_Preprocessing.py
    ```

   The unit tests will verify the functionality of the functions in the `creditcard_EDA_and_Preprocessing.py` script.

## Building and Deploying the Fraud Detection Model

1. **Model Building**:
   - Use scripts and notebooks provided in the `notebooks/models/` directory for model building and evaluation.
   - Train models such as Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Multi-Layer Perceptron (MLP), CNN, RNN, and LSTM.

2. **Model Explainability**:
   - Use SHAP and LIME for model interpretability, ensuring transparency and trust in model predictions.

3. **Model Deployment**:
   - Use the `api_fraud_detection/` directory to deploy the model using Flask and Docker.
   - Follow the instructions in `serve_model.py`, `Dockerfile`, and `requirements.txt` for deployment.

4. **Running the Flask Application**:
    ```bash
    cd api_fraud_detection
    python serve_model.py
    ```

5. **Building and Running Docker Container**:
    ```bash
    cd api_fraud_detection
    docker build -t fraud-detection-model .
    docker run -p 5000:5000 fraud-detection-model
    ```

## Building a Dashboard with Flask and Dash

1. **Set Up Your Flask Application**:
   - Create a `app.py` file for your Flask application.
   - Define the main route and run the server.

2. **Integrate Dash into Flask**:
   - Create a `dashboard.py` file to set up the Dash app.
   - Integrate Dash with Flask by using the Flask server as the backend for Dash.

3. **Set Up Docker Compose**:
   - Create a `docker-compose.yml` file to manage multi-container Docker applications.

4. **Build and Run Docker Container**:
   ```bash
   cd fraud_dashboard
   docker-compose up --build
   ```

5. **Navigate to the Dashboard**:
   - Open your browser and navigate to `http://127.0.0.1:8050/dashboard/` to see the Dash dashboard.

## Project Components

### Data Analysis and Preprocessing
- **Handle Missing Values**: Impute or drop missing values.
- **Data Cleaning**: Remove duplicates and correct data types.
- **Exploratory Data Analysis (EDA)**: Perform univariate and bivariate analysis.
- **Merge Datasets for Geolocation Analysis**: Convert IP addresses to integer format and merge with geolocation data.
- **Feature Engineering**: Create transaction frequency, velocity, and time-based features.
- **Normalization and Scaling**: Normalize and scale numerical features.
- **Encode Categorical Features**: Encode categorical features using label encoding.

### Visualizations
- Plot distributions, bivariate analysis, fraud rates, and correlation heatmaps.

### Model Building and Evaluation
- Train various models and evaluate their performance.
- Use SMOTE for class balancing.
- Implement model explainability using SHAP and LIME.

### Model Deployment
- Deploy the model using Flask and Docker.
- Create a RESTful API for real-time fraud detection.

### Dashboard and Visualization
- Build an interactive dashboard using Dash and Flask for visualizing fraud insights.

### Authors
- **Abenezer Baheru** - *Initial work* - [GitHub Profile](https://github.com/Abenezer-Baheru)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.