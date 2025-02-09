import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Load the dataset from the specified file path.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def clean_data(data):
    """
    Clean the dataset by checking for duplicates and correcting data types.
    """
    try:
        # Remove duplicates
        data = data.drop_duplicates()
        logging.info("Duplicates removed.")

        # Correct data types
        data['Time'] = data['Time'].astype('float64')
        data['Amount'] = data['Amount'].astype('float64')
        data['Class'] = data['Class'].astype('int64')

        # Ensure all V columns are float64
        for col in data.columns:
            if col.startswith('V'):
                data[col] = data[col].astype('float64')

        logging.info("Data types corrected.")
        return data
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        return None

def feature_engineering(data):
    """
    Perform feature engineering by creating new features.
    """
    try:
        # Convert 'Time' to datetime
        data['Time'] = pd.to_datetime(data['Time'], unit='s', origin='unix')

        # Extract the hour from the 'Time' column
        data['Transaction_Hour'] = data['Time'].dt.hour

        # Extract the day of the week from the 'Time' column
        data['Transaction_DayOfWeek'] = data['Time'].dt.dayofweek

        # Define bins with a 500-unit interval for Amount
        min_value = int(data['Amount'].min())
        max_value = int(data['Amount'].max()) + 1
        bins = list(range(min_value, max_value, 500)) + [max_value]
        labels = [f"{i}-{i+500}" for i in bins[:-2]] + [f"{bins[-2]}+"]
        data['Amount_Range'] = pd.cut(data['Amount'], bins=bins, labels=labels, include_lowest=True)

        logging.info("Feature engineering completed.")
        return data
    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        return None

def preprocess_data(data):
    """
    Normalize, scale, and encode the dataset.
    """
    try:
        # Normalize and scale numerical columns
        numerical_columns = ['Amount'] + [f'V{i}' for i in range(1, 29)] + ['Transaction_Hour']
        numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        data[numerical_columns] = numerical_transformer.fit_transform(data[numerical_columns])

        # Encode categorical columns
        categorical_columns = ['Transaction_DayOfWeek', 'Amount_Range']
        label_encoder = LabelEncoder()
        for column in categorical_columns:
            data[column] = label_encoder.fit_transform(data[column])

        logging.info("Data preprocessed.")
        return data
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        return None

def plot_distributions(data):
    """
    Plot distributions for all columns.
    """
    try:
        # Create subplots for remaining variables
        fig, axes = plt.subplots(7, 4, figsize=(20, 25))

        # List of columns for remaining variables
        columns_to_analyze = [f'V{i}' for i in range(1, 29)]

        # Plot each column
        for i, col in enumerate(columns_to_analyze):
            row, col_pos = divmod(i, 4)
            sns.histplot(data[col], kde=True, ax=axes[row, col_pos])
            axes[row, col_pos].set_title(f'Distribution of {col}')
            axes[row, col_pos].set_xlabel(col)
            axes[row, col_pos].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

        logging.info("Distributions plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting distributions: {e}")

def plot_bivariate_analysis(data):
    """
    Plot bivariate analysis for specified columns.
    """
    try:
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        # Boxplot for Amount and Class
        sns.boxplot(x='Class', y='Amount', data=data, ax=axes[0])
        axes[0].set_title('Boxplot of Amount by Class')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Amount')

        # Boxplot for Time (Hour) and Class
        sns.boxplot(x='Class', y=data['Time'].dt.hour, data=data, ax=axes[1])
        axes[1].set_title('Boxplot of Time (Hour) by Class')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Time (Hour)')

        plt.tight_layout()
        plt.show()

        logging.info("Bivariate analysis plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting bivariate analysis: {e}")

def plot_fraud_rates(data):
    """
    Plot fraud rates and total transactions by Amount range and Time (Hour).
    """
    try:
        # Calculate the total number of transactions and fraudulent transactions by Amount range
        total_transactions_by_amount_range = data['Amount_Range'].value_counts()
        fraud_transactions_by_amount_range = data[data['Class'] == 1]['Amount_Range'].value_counts()

        # Calculate the fraud rate for each Amount range
        fraud_rate_by_amount_range = (fraud_transactions_by_amount_range / total_transactions_by_amount_range) * 100

        # Calculate the total number of transactions and fraudulent transactions by Hour
        total_transactions_by_hour = data['Hour'].value_counts()
        fraud_transactions_by_hour = data[data['Class'] == 1]['Hour'].value_counts()

        # Calculate the fraud rate for each Hour
        fraud_rate_by_hour = (fraud_transactions_by_hour / total_transactions_by_hour) * 100

        # Create DataFrames for visualization
        amount_range_data = pd.DataFrame({
            'Amount Range': fraud_rate_by_amount_range.index,
            'Fraud Rate (%)': fraud_rate_by_amount_range.values,
            'Total Transactions': total_transactions_by_amount_range[fraud_rate_by_amount_range.index].values
        })

        hour_data = pd.DataFrame({
            'Hour': fraud_rate_by_hour.index,
            'Fraud Rate (%)': fraud_rate_by_hour.values,
            'Total Transactions': total_transactions_by_hour[fraud_rate_by_hour.index].values
        })

        # Create the figure and axes for subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Plot the fraud rate and total transactions by Amount range
        sns.barplot(x='Amount Range', y='Fraud Rate (%)', data=amount_range_data, palette='viridis', ax=axes[0])
        axes[0].set_title('Fraud Rate and Total Transactions by Amount Range')
        axes[0].set_xlabel('Amount Range')
        axes[0].set_ylabel('Fraud Rate (%)')
        ax2 = axes[0].twinx()
        sns.lineplot(x='Amount Range', y='Total Transactions', data=amount_range_data, color='red', marker='o', ax=ax2)
        ax2.set_ylabel('Total Transactions')
        axes[0].tick_params(axis='x', rotation=90)

        # Plot the fraud rate and total transactions by Hour
        sns.barplot(x='Hour', y='Fraud Rate (%)', data=hour_data, palette='viridis', ax=axes[1])
        axes[1].set_title('Fraud Rate and Total Transactions by Hour')
        axes[1].set_xlabel('Time (Hour)')
        axes[1].set_ylabel('Fraud Rate (%)')
        ax2 = axes[1].twinx()
        sns.lineplot(x='Hour', y='Total Transactions', data=hour_data, color='red', marker='o', ax=ax2)
        ax2.set_ylabel('Total Transactions')

        plt.tight_layout()
        plt.show()

        logging.info("Fraud rates plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting fraud rates: {e}")

def plot_correlation_heatmap(data):
    """
    Plot a heatmap to visualize the correlations between all columns.
    """
    try:
        # Compute the correlation matrix
        corr_matrix = data.corr()

        # Create a heatmap
        plt.figure(figsize=(20, 15))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title('Heatmap of Correlations Between All Columns')
        plt.show()

        logging.info("Correlation heatmap plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting correlation heatmap: {e}")

def save_data(data, file_path):
    """
    Save the processed data to the specified file path.
    """
    try:
        data.to_csv(file_path, index=False)
        logging.info(f"Final processed data has been saved to '{file_path}'.")
    except Exception as e:
        logging.error(f"Error saving data: {e}")

def main():
    """
    Main function to perform EDA and preprocessing.
    """
    file_path = '../../src/data/creditcard.csv'
    processed_file_path = '../../src/data/creditcard_final_processed_data.csv'

    # Load data
    data = load_data(file_path)
    if data is None:
        return

    # Clean data
    data = clean_data(data)
    if data is None:
        return

    # Feature engineering
    data = feature_engineering(data)
    if data is None:
        return

    # Preprocess data
    data = preprocess_data(data)
    if data is None:
        return

    # Plot distributions
    plot_distributions(data)

    # Plot bivariate analysis
    plot_bivariate_analysis(data)

    # Plot fraud rates
    plot_fraud_rates(data)

    # Plot correlation heatmap
    plot_correlation_heatmap(data)

    # Save processed data
    save_data(data, processed_file_path)

if __name__ == "__main__":
    main()