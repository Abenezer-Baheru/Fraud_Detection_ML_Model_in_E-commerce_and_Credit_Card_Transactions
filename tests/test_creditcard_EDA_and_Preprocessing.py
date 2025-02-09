import unittest
import pandas as pd
import numpy as np
from scripts.creditcard_EDA_and_Preprocessing import (
    load_data, clean_data, feature_engineering, preprocess_data, 
    plot_distributions, plot_bivariate_analysis, plot_fraud_rates, 
    plot_correlation_heatmap, save_data
)

class TestCreditcardEDAandPreprocessing(unittest.TestCase):

    def setUp(self):
        """
        Setup a sample dataframe to use for testing.
        """
        data = {
            'Time': [0, 86400, 172800],
            'Amount': [1.0, 2.0, 3.0],
            'Class': [0, 1, 0],
            'V1': [0.1, 0.2, 0.3],
            'V2': [0.4, 0.5, 0.6],
            'V3': [0.7, 0.8, 0.9],
            # Include V4 to V28 for completeness
            **{f'V{i}': [0.1*i, 0.2*i, 0.3*i] for i in range(4, 29)}
        }
        self.df = pd.DataFrame(data)

    def test_load_data(self):
        """
        Test loading data from a CSV file.
        """
        data = load_data('../../src/data/creditcard.csv')
        self.assertIsNotNone(data)

    def test_clean_data(self):
        """
        Test cleaning data.
        """
        cleaned_data = clean_data(self.df)
        self.assertEqual(cleaned_data.shape, self.df.shape)
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)

    def test_feature_engineering(self):
        """
        Test feature engineering.
        """
        engineered_data = feature_engineering(self.df)
        self.assertIn('Transaction_Hour', engineered_data.columns)
        self.assertIn('Transaction_DayOfWeek', engineered_data.columns)
        self.assertIn('Amount_Range', engineered_data.columns)

    def test_preprocess_data(self):
        """
        Test preprocessing data.
        """
        preprocessed_data = preprocess_data(self.df)
        for col in ['Amount'] + [f'V{i}' for i in range(1, 29)] + ['Transaction_Hour']:
            self.assertAlmostEqual(preprocessed_data[col].mean(), 0, places=5)
            self.assertAlmostEqual(preprocessed_data[col].std(), 1, places=5)

    def test_save_data(self):
        """
        Test saving data to a CSV file.
        """
        save_data(self.df, '../../src/data/test_creditcard_processed_data.csv')
        saved_data = pd.read_csv('../../src/data/test_creditcard_processed_data.csv')
        self.assertEqual(self.df.shape, saved_data.shape)

    def test_plot_distributions(self):
        """
        Test plot distributions.
        """
        try:
            plot_distributions(self.df)
        except Exception as e:
            self.fail(f"plot_distributions() raised {e}")

    def test_plot_bivariate_analysis(self):
        """
        Test plot bivariate analysis.
        """
        try:
            plot_bivariate_analysis(self.df)
        except Exception as e:
            self.fail(f"plot_bivariate_analysis() raised {e}")

    def test_plot_fraud_rates(self):
        """
        Test plot fraud rates.
        """
        try:
            plot_fraud_rates(self.df)
        except Exception as e:
            self.fail(f"plot_fraud_rates() raised {e}")

    def test_plot_correlation_heatmap(self):
        """
        Test plot correlation heatmap.
        """
        try:
            plot_correlation_heatmap(self.df)
        except Exception as e:
            self.fail(f"plot_correlation_heatmap() raised {e}")

if __name__ == '__main__':
    unittest.main()