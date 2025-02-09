# Credit Card Fraud Detection - EDA and Preprocessing

## Project Overview
This project focuses on performing Exploratory Data Analysis (EDA) and preprocessing for credit card fraud detection. The goal is to clean the dataset, perform feature engineering, analyze the data, and prepare it for model building.

## Folder Structure
The project directory is structured as follows:
├── notebooks/ │ ├── creditcard_EDA_and_Preprocessing.ipynb │ ├── e-commerce_EDA_and_Preprocessing.ipynb │ ├── IP_Address_EDA_and_Preprocessing.ipynb │ ├── creditcard_model.ipynb │ ├── e-commerce_model.ipynb │ ├── init.py │ └── README.md└── scripts/ │ ├── creditcard_EDA_and_Preprocessing.py │ ├── e-commerce_EDA_and_Preprocessing.py │ ├── init.py │ └── README.md├── src/ │ ├── data/ │ │ ├── creditcard.csv│ │ ├── Fraud_Data.csv │ │ ├── IpAddress_to_Country.csv │ ├── init.py ├── tests/ │ ├── test_creditcard_EDA_and_Preprocessing.py │ ├── test_e-commerce_EDA_and_Preprocessing.py │ ├── init.py ├── .gitignore ├── requirements.txt├── README.md


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

### Running the Scripts

1. Navigate to the `scripts` directory:
    ```bash
    cd scripts
    ```

2. Run the `creditcard_EDA_and_Preprocessing.py` script:
    ```bash
    python creditcard_EDA_and_Preprocessing.py
    ```

   This script will load the dataset, clean the data, perform feature engineering, preprocess the data, and save the processed data to the specified path.

### Running the Notebook

1. Open the `notebooks/creditcard_EDA_and_Preprocessing.ipynb` notebook in Jupyter:
    ```bash
    jupyter notebook notebooks/creditcard_EDA_and_Preprocessing.ipynb
    ```

2. Execute the cells in the notebook step-by-step. The notebook calls the functions from the `creditcard_EDA_and_Preprocessing.py` script and performs the analysis and preprocessing tasks.

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

### Authors
- **Abenezer Baheru** - *Initial work* - [GitHub Profile](https://github.com/Abenezer-Baheru)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.