import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EDA_and_Preprocessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load the dataset from the specified file path."""
        try:
            self.data = pd.read_csv(self.file_path)
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")

    def check_data_info(self):
        """Check the size and missing values of the data."""
        try:
            print("Size of the data:")
            print(self.data.shape)

            print("\nHead of the data:")
            print(self.data.head())

            print("\nMissing values in each column:")
            print(self.data.isnull().sum())
        except Exception as e:
            logging.error(f"Error checking data information: {e}")

    def clean_data(self):
        """Remove duplicates and correct data types."""
        try:
            # Remove duplicates
            self.data = self.data.drop_duplicates()
            logging.info("Duplicates removed.")

            # Correct data types
            self.data['Time'] = self.data['Time'].astype('float64')
            self.data['Amount'] = self.data['Amount'].astype('float64')
            self.data['Class'] = self.data['Class'].astype('int64')

            # Ensure all V columns are float64
            for col in self.data.columns:
                if col.startswith('V'):
                    self.data[col] = self.data[col].astype('float64')

            logging.info("Data types corrected.")
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")

    def identify_outliers(self):
        """Identify outliers in the dataset using the IQR method."""
        try:
            Q1 = self.data.quantile(0.25)
            Q3 = self.data.quantile(0.75)
            IQR = Q3 - Q1

            outliers = ((self.data < (Q1 - 1.5 * IQR)) | (self.data > (Q3 + 1.5 * IQR))).sum()
            total_entries = self.data.count()
            outlier_percentage = (outliers / total_entries) * 100

            outlier_summary = pd.DataFrame({
                'Number of Outliers': outliers,
                'Percentage of Outliers (%)': outlier_percentage
            })

            print(outlier_summary)
        except Exception as e:
            logging.error(f"Error identifying outliers: {e}")

    def remove_outliers(self, columns):
        """Remove outliers based on the IQR method for specified columns."""
        try:
            for col in columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
            logging.info("Outliers removed.")
        except Exception as e:
            logging.error(f"Error removing outliers: {e}")

    def convert_time(self):
        """Convert 'Time' column to datetime and extract useful features."""
        try:
            self.data['Time'] = pd.to_datetime(self.data['Time'], unit='s', origin='unix')
            logging.info("Time column converted to datetime.")
        except Exception as e:
            logging.error(f"Error converting time: {e}")

    def plot_distributions(self):
        """Plot distributions for Time, Amount, and Class."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))

            sns.histplot(self.data['Time'].dt.hour, kde=True, ax=axes[0])
            axes[0].set_title('Distribution of Time (Hour)')
            axes[0].set_xlabel('Hour')
            axes[0].set_ylabel('Frequency')

            sns.histplot(self.data['Amount'], kde=True, ax=axes[1])
            axes[1].set_title('Distribution of Amount')
            axes[1].set_xlabel('Amount')
            axes[1].set_ylabel('Frequency')
            axes[1].set_ylim(0, 13566)

            class_counts = self.data['Class'].value_counts()
            class_percentages = class_counts / class_counts.sum() * 100
            axes[2].pie(class_percentages, labels=class_percentages.index, autopct='%1.1f%%', startangle=90, counterclock=False, colors=['#66b3ff','#ff6666'])
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            axes[2].add_artist(centre_circle)
            axes[2].set_title('Percentage of Class Values')

            plt.tight_layout()
            plt.show()
            logging.info("Distributions plotted successfully.")
        except Exception as e:
            logging.error(f"Error plotting distributions: {e}")

    def plot_remaining_variables(self):
        """Create subplots for remaining variables."""
        try:
            fig, axes = plt.subplots(7, 4, figsize=(20, 25))
            columns_to_analyze = [f'V{i}' for i in range(1, 29)]

            for i, col in enumerate(columns_to_analyze):
                row, col_pos = divmod(i, 4)
                sns.histplot(self.data[col], kde=True, ax=axes[row, col_pos])
                axes[row, col_pos].set_title(f'Distribution of {col}')
                axes[row, col_pos].set_xlabel(col)
                axes[row, col_pos].set_ylabel('Frequency')

            plt.tight_layout()
            plt.show()
            logging.info("Remaining variable distributions plotted successfully.")
        except Exception as e:
            logging.error(f"Error plotting remaining variable distributions: {e}")

    def plot_boxplots(self):
        """Create boxplots for Amount and Class, and Time (Hour) and Class."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))

            sns.boxplot(x='Class', y='Amount', data=self.data, ax=axes[0])
            axes[0].set_title('Boxplot of Amount by Class')
            axes[0].set_xlabel('Class')
            axes[0].set_ylabel('Amount')

            sns.boxplot(x='Class', y=self.data['Time'].dt.hour, data=self.data, ax=axes[1])
            axes[1].set_title('Boxplot of Time (Hour) by Class')
            axes[1].set_xlabel('Class')
            axes[1].set_ylabel('Time (Hour)')

            plt.tight_layout()
            plt.show()
            logging.info("Boxplots plotted successfully.")
        except Exception as e:
            logging.error(f"Error plotting boxplots: {e}")

    def plot_bivariate_analysis(self):
        """Create subplots for bivariate analysis."""
        try:
            fig, axes = plt.subplots(7, 4, figsize=(20, 40))
            columns_to_analyze = [f'V{i}' for i in range(1, 29)]

            for i, col in enumerate(columns_to_analyze):
                row, col_pos = divmod(i, 4)
                sns.boxplot(x='Class', y=col, data=self.data, ax=axes[row, col_pos])
                axes[row, col_pos].set_title(f'Boxplot of {col} by Class')
                axes[row, col_pos].set_xlabel('Class')
                axes[row, col_pos].set_ylabel(col)

            plt.tight_layout()
            plt.show()
            logging.info("Bivariate analysis plotted successfully.")
        except Exception as e:
            logging.error(f"Error plotting bivariate analysis: {e}")

    def analyze_fraud_rate(self):
        """Analyze fraud rate by Amount range and Hour."""
        try:
            self.data['Amount'] = pd.to_numeric(self.data['Amount'], errors='coerce')
            self.data = self.data.dropna(subset=['Amount'])

            min_value = int(self.data['Amount'].min())
            max_value = int(self.data['Amount'].max()) + 1
            bins = list(range(min_value, max_value, 500)) + [max_value]
            labels = [f"{i}-{i+500}" for i in bins[:-2]] + [f"{bins[-2]}+"]
            self.data['Amount_range'] = pd.cut(self.data['Amount'], bins=bins, labels=labels, include_lowest=True)

            self.data['Time'] = pd.to_datetime(self.data['Time'], unit='s', origin='unix')
            self.data['Hour'] = self.data['Time'].dt.hour

            total_transactions_by_amount_range = self.data['Amount_range'].value_counts()
            fraud_transactions_by_amount_range = self.data[self.data['Class'] == 1]['Amount_range'].value_counts()
            fraud_rate_by_amount_range = (fraud_transactions_by_amount_range / total_transactions_by_amount_range) * 100

            total_transactions_by_hour = self.data['Hour'].value_counts()
            fraud_transactions_by_hour = self.data[self.data['Class'] == 1]['Hour'].value_counts()
            fraud_rate_by_hour = (fraud_transactions_by_hour / total_transactions_by_hour) * 100

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

            fig, axes = plt.subplots(1, 2, figsize=(18, 6))

            sns.barplot(x='Amount Range', y='Fraud Rate (%)', data=amount_range_data, palette='viridis', ax=axes[0])
            axes[0].set_title('Fraud Rate and Total Transactions by Amount Range')
            axes[0].set_xlabel('Amount Range')
            axes[0].set_ylabel('Fraud Rate (%)')
            ax2 = axes[0].twinx()
            sns.lineplot(x='Amount Range', y='Total Transactions', data=amount_range_data, color='red', marker='o', ax=ax2)
            ax2.set_ylabel('Total Transactions')
            axes[0].tick_params(axis='x', rotation=90)

            sns.barplot(x='Hour', y='Fraud Rate (%)', data=hour_data, palette='viridis', ax=axes[1])
            axes[1].set_title('Fraud Rate and Total Transactions by Hour')
            axes[1].set_xlabel('Time (Hour)')
            axes[1].set_ylabel('Fraud Rate (%)')
            ax2 = axes[1].twinx()
            sns.lineplot(x='Hour', y='Total Transactions', data=hour_data, color='red', marker='o', ax=ax2)
            ax2.set_ylabel('Total Transactions')

            plt.tight_layout()
            plt.show()

            print("\nTable: Fraud Rate and Total Transactions by Amount Range")
            print(amount_range_data)
            print("\nTable: Fraud Rate and Total Transactions by Hour")
            print(hour_data)
            logging.info("Fraud rate analysis completed successfully.")
        except Exception as e:
            logging.error(f"Error analyzing fraud rate: {e}")

    def compute_correlation_matrix(self):
        """Compute and plot the correlation matrix."""
        try:
            corr_matrix = self.data.corr()

            plt.figure(figsize=(20, 15))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
            plt.title('Heatmap of Correlations Between All Columns')
            plt.show()
            logging.info("Correlation matrix computed and plotted successfully.")
        except Exception as e:
            logging.error(f"Error computing correlation matrix: {e}")

    def feature_engineering(self):
        """Perform feature engineering on the dataset."""
        try:
            self.data['Time'] = pd.to_datetime(self.data['Time'], unit='s', origin='unix')
            self.data['Transaction_Hour'] = self.data['Time'].dt.hour
            self.data['Transaction_DayOfWeek'] = self.data['Time'].dt.dayofweek

            min_value = int(self.data['Amount'].min())
            max_value = int(self.data['Amount'].max()) + 1
            bins = list(range(min_value, max_value, 500)) + [max_value]
            labels = [f"{i}-{i+500}" for i in bins[:-2]] + [f"{bins[-2]}+"]
            self.data['Amount_Range'] = pd.cut(self.data['Amount'], bins=bins, labels=labels, include_lowest=True)

            logging.info("Feature engineering completed successfully.")
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")

    def preprocess_data(self):
        """Preprocess the data by scaling numerical columns and encoding categorical columns."""
        try:
            numerical_columns = ['Amount'] + [f'V{i}' for i in range(1, 29)] + ['Transaction_Hour']
            numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            self.data[numerical_columns] = numerical_transformer.fit_transform(self.data[numerical_columns])

            categorical_columns = ['Transaction_DayOfWeek', 'Amount_Range']
            label_encoder = LabelEncoder()
            for column in categorical_columns:
                self.data[column] = label_encoder.fit_transform(self.data[column])

            logging.info("Data preprocessed successfully.")
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")

    def save_data(self, output_path):
        """Save the processed data to the specified file path."""
        try:
            self.data.to_csv(output_path, index=False)
            logging.info(f"Final processed data has been saved to '{output_path}'.")
        except Exception as e:
            logging.error(f"Error saving data: {e}")

def main():
    file_path = '../../src/data/creditcard.csv'
    output_path = '../../src/data/creditcard_processed_data.csv'
    eda = EDA_and_Preprocessing(file_path)

    eda.load_data()
    eda.check_data_info()
    eda.clean_data()
    eda.identify_outliers()
    eda.remove_outliers(columns=['Amount', 'Class'])
    eda.convert_time()
    eda.plot_distributions()
    eda.plot_remaining_variables()
    eda.plot_boxplots()
    eda.plot_bivariate_analysis()
    eda.analyze_fraud_rate()
    eda.compute_correlation_matrix()
    eda.feature_engineering()
    eda.preprocess_data()
    eda.save_data(output_path)

if __name__ == "__main__":
    main()