"""
Unit tests for data preprocessing module
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import MixedDataPreprocessor, load_and_preprocess_telecom_data


class TestMixedDataPreprocessor(unittest.TestCase):
    """Test cases for MixedDataPreprocessor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = MixedDataPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'customerID': ['C001', 'C002', 'C003'],
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.0, 75.0, 25.0],
            'TotalCharges': ['600.0', '1800.0', ' '],  # Note: string with empty value
            'Contract': ['Month-to-month', 'Two year', 'One year'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'PhoneService': ['Yes', 'Yes', 'No'],
            'Churn': ['No', 'No', 'Yes']
        })

    def test_identify_feature_types(self):
        """Test automatic feature type identification"""
        numerical_cols, categorical_cols = self.preprocessor.identify_feature_types(
            self.sample_data, exclude_cols=['customerID', 'Churn']
        )
        
        # Check if numerical columns are correctly identified
        expected_numerical = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
        self.assertTrue(all(col in numerical_cols for col in expected_numerical))
        
        # Check if categorical columns are correctly identified
        expected_categorical = ['gender', 'Contract', 'InternetService', 'PhoneService']
        self.assertTrue(all(col in categorical_cols for col in expected_categorical))

    def test_handle_mixed_types(self):
        """Test handling of mixed data types (string numbers)"""
        df_cleaned = self.preprocessor.handle_mixed_types(self.sample_data, 'Churn')
        
        # Check if TotalCharges was converted to numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(df_cleaned['TotalCharges']))
        
        # Check if empty string was converted to NaN
        self.assertTrue(df_cleaned['TotalCharges'].isna().any())

    def test_create_engineered_features(self):
        """Test feature engineering functionality"""
        # First handle mixed types
        df_cleaned = self.preprocessor.handle_mixed_types(self.sample_data)
        
        # Handle missing TotalCharges
        mask = df_cleaned['TotalCharges'].isna()
        df_cleaned.loc[mask, 'TotalCharges'] = df_cleaned.loc[mask, 'MonthlyCharges'] * df_cleaned.loc[mask, 'tenure']
        
        # Create engineered features
        df_featured = self.preprocessor.create_engineered_features(df_cleaned)
        
        # Check if new features were created
        expected_features = ['tenure_group', 'avg_monthly_charge', 'total_services', 
                           'has_internet', 'has_phone']
        
        for feature in expected_features:
            self.assertIn(feature, df_featured.columns, f"Feature {feature} not created")

    def test_fit_transform(self):
        """Test complete preprocessing pipeline"""
        X_processed, feature_names = self.preprocessor.fit_transform(self.sample_data, 'Churn')
        
        # Check output shape
        self.assertEqual(X_processed.shape[0], len(self.sample_data))
        self.assertGreater(X_processed.shape[1], 0)
        
        # Check feature names
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)

    def test_get_feature_importance_mapping(self):
        """Test feature importance mapping"""
        mapping = self.preprocessor.get_feature_importance_mapping()
        
        self.assertIsInstance(mapping, dict)
        self.assertGreater(len(mapping), 0)
        
        # Check if important features are included
        self.assertIn('tenure', mapping)
        self.assertIn('MonthlyCharges', mapping)


class TestDataPreprocessingFunctions(unittest.TestCase):
    """Test cases for module-level functions"""

    def test_load_and_preprocess_telecom_data(self):
        """Test data loading and preprocessing function"""
        # Create temporary CSV file
        sample_data = pd.DataFrame({
            'customerID': ['C001', 'C002'],
            'gender': ['Male', 'Female'],
            'SeniorCitizen': [0, 1],
            'tenure': [12, 24],
            'MonthlyCharges': [50.0, 75.0],
            'TotalCharges': ['600.0', '1800.0'],
            'Contract': ['Month-to-month', 'Two year'],
            'InternetService': ['DSL', 'Fiber optic'],
            'PhoneService': ['Yes', 'Yes'],
            'Churn': ['No', 'Yes']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            X, y, features, preprocessor = load_and_preprocess_telecom_data(temp_file)
            
            # Check outputs
            self.assertIsNotNone(X)
            self.assertIsNotNone(y)
            self.assertIsInstance(features, list)
            self.assertIsInstance(preprocessor, MixedDataPreprocessor)
            
            # Check shapes
            self.assertEqual(X.shape[0], len(sample_data))
            self.assertEqual(len(y), len(sample_data))
            
            # Check target variable conversion
            self.assertTrue(all(val in [0, 1] for val in y))
            
        finally:
            os.unlink(temp_file)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def setUp(self):
        self.preprocessor = MixedDataPreprocessor()

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(Exception):
            self.preprocessor.fit_transform(empty_df)

    def test_missing_required_columns(self):
        """Test handling of missing required columns"""
        incomplete_data = pd.DataFrame({
            'customerID': ['C001'],
            'gender': ['Male']
        })
        
        # Should handle gracefully without required telecom columns
        X, features = self.preprocessor.fit_transform(incomplete_data)
        self.assertIsNotNone(X)

    def test_all_missing_values(self):
        """Test handling of columns with all missing values"""
        data_with_nulls = pd.DataFrame({
            'customerID': ['C001', 'C002'],
            'feature1': [np.nan, np.nan],
            'feature2': ['A', 'B']
        })
        
        X, features = self.preprocessor.fit_transform(data_with_nulls)
        self.assertIsNotNone(X)


if __name__ == '__main__':
    unittest.main()
