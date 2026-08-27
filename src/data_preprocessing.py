"""
Data Preprocessing Module for Customer Segmentation

This module contains functions to handle data cleaning, preprocessing,
and feature engineering for mixed data types (numerical and categorical).

Key Challenges Addressed:
1. Handling missing values in mixed data types
2. Converting string numbers to numeric types
3. Creating meaningful engineered features
4. Scaling numerical features while preserving categorical information
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class MixedDataPreprocessor:
    """
    A comprehensive preprocessor for handling mixed data types
    in customer segmentation tasks.
    """
    
    def __init__(self):
        self.numerical_features = None
        self.categorical_features = None
        self.preprocessor = None
        self.feature_names = None
        
    def identify_feature_types(self, df, exclude_cols=None):
        """
        Automatically identify numerical and categorical features
        
        Args:
            df: pandas DataFrame
            exclude_cols: list of columns to exclude from processing
            
        Returns:
            tuple: (numerical_features, categorical_features)
        """
        if exclude_cols is None:
            exclude_cols = []
            
        # Numerical features (including binary encoded as int)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Categorical features (object type)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        self.numerical_features = numerical_cols
        self.categorical_features = categorical_cols
        
        return numerical_cols, categorical_cols
    
    def handle_mixed_types(self, df, target_col=None):
        """
        Handle mixed data types issues like string numbers
        
        Args:
            df: pandas DataFrame
            target_col: name of target column to preserve
            
        Returns:
            pandas DataFrame with corrected data types
        """
        df_clean = df.copy()
        
        # Check for string numbers in object columns
        for col in df_clean.select_dtypes(include=['object']).columns:
            if col == target_col:
                continue
                
            # Try to convert to numeric if it looks like numbers
            if df_clean[col].str.strip().str.replace('.', '', 1).str.isdigit().any():
                try:
                    # Handle empty strings and spaces
                    df_clean[col] = df_clean[col].replace(' ', np.nan)
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    print(f"Converted {col} from object to numeric")
                except:
                    pass
                    
        return df_clean
    
    def create_engineered_features(self, df):
        """
        Create meaningful engineered features for telecom customer segmentation
        
        Args:
            df: pandas DataFrame
            
        Returns:
            pandas DataFrame with additional features
        """
        df_featured = df.copy()
        
        # Tenure-based features
        if 'tenure' in df.columns:
            df_featured['tenure_group'] = pd.cut(
                df_featured['tenure'], 
                bins=[0, 12, 24, 36, 48, 100], 
                labels=['New', 'Growing', 'Established', 'Mature', 'Veteran']
            )
            df_featured['is_new_customer'] = (df_featured['tenure'] <= 3).astype(int)
            df_featured['is_long_tenure'] = (df_featured['tenure'] >= 36).astype(int)
        
        # Revenue-based features
        if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
            df_featured['avg_monthly_charge'] = df_featured['TotalCharges'] / (df_featured['tenure'] + 1)
            df_featured['revenue_per_month'] = df_featured['TotalCharges'] / df_featured['tenure'].replace(0, 1)
            
            # Revenue groups
            df_featured['revenue_segment'] = pd.cut(
                df_featured['MonthlyCharges'],
                bins=4,
                labels=['Budget', 'Standard', 'Premium', 'Enterprise']
            )
        
        # Service adoption features
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        if all(col in df.columns for col in service_cols):
            df_featured['total_services'] = 0
            for col in service_cols:
                if col == 'InternetService':
                    df_featured['total_services'] += (df_featured[col] != 'No').astype(int)
                else:
                    df_featured['total_services'] += (df_featured[col] == 'Yes').astype(int)
            
            # Service adoption rate
            max_services = len(service_cols)
            df_featured['service_adoption_rate'] = df_featured['total_services'] / max_services
            
            # Service categories
            df_featured['has_internet'] = (df_featured['InternetService'] != 'No').astype(int)
            df_featured['has_phone'] = (df_featured['PhoneService'] == 'Yes').astype(int)
            df_featured['has_security'] = (df_featured['OnlineSecurity'] == 'Yes').astype(int)
            df_featured['has_streaming'] = ((df_featured['StreamingTV'] == 'Yes') | 
                                          (df_featured['StreamingMovies'] == 'Yes')).astype(int)
        
        # Contract and loyalty features
        if 'Contract' in df.columns:
            df_featured['is_monthly_contract'] = (df_featured['Contract'] == 'Month-to-month').astype(int)
            df_featured['has_long_contract'] = (df_featured['Contract'].isin(['One year', 'Two year'])).astype(int)
        
        # Customer lifecycle features
        if 'Churn' in df.columns:
            df_featured['at_risk'] = (df_featured['Churn'] == 'Yes').astype(int)
            
        return df_featured
    
    def create_preprocessing_pipeline(self, numerical_features, categorical_features, 
                                    handle_missing='median', encoding='onehot'):
        """
        Create a preprocessing pipeline for mixed data types
        
        Args:
            numerical_features: list of numerical feature names
            categorical_features: list of categorical feature names
            handle_missing: strategy for handling missing values ('median', 'mean', 'mode')
            encoding: encoding strategy for categorical variables ('onehot', 'label')
            
        Returns:
            sklearn ColumnTransformer pipeline
        """
        
        # Numerical preprocessing
        if handle_missing == 'median':
            num_imputer = SimpleImputer(strategy='median')
        elif handle_missing == 'mean':
            num_imputer = SimpleImputer(strategy='mean')
        else:
            num_imputer = SimpleImputer(strategy='constant', fill_value=0)
            
        # Categorical preprocessing
        if encoding == 'onehot':
            cat_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        else:
            cat_encoder = LabelEncoder()
            
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', [num_imputer, StandardScaler()], numerical_features),
                ('cat', cat_encoder, categorical_features)
            ],
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def fit_transform(self, df, target_col=None):
        """
        Fit the preprocessor and transform the data
        
        Args:
            df: pandas DataFrame
            target_col: target column name to exclude from preprocessing
            
        Returns:
            numpy array of preprocessed features
        """
        # Handle mixed types first
        df_clean = self.handle_mixed_types(df, target_col)
        
        # Create engineered features
        df_featured = self.create_engineered_features(df_clean)
        
        # Exclude target and ID columns
        exclude_cols = [target_col] if target_col else []
        exclude_cols.extend(['customerID', 'Churn'])  # Common ID and target columns
        
        # Identify feature types
        numerical_features, categorical_features = self.identify_feature_types(
            df_featured, exclude_cols
        )
        
        # Create and fit preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(
            numerical_features, categorical_features
        )
        
        # Transform the data
        X_processed = preprocessor.fit_transform(df_featured)
        
        # Create feature names
        if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
            cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(
                categorical_features
            )
        else:
            cat_feature_names = categorical_features
            
        feature_names = numerical_features + list(cat_feature_names)
        self.feature_names = feature_names
        
        return X_processed, feature_names
    
    def get_feature_importance_mapping(self):
        """
        Get mapping between feature names and their business meaning
        
        Returns:
            dict: mapping of feature names to business descriptions
        """
        mapping = {
            'tenure': 'Customer tenure in months',
            'MonthlyCharges': 'Monthly subscription charges',
            'TotalCharges': 'Total charges paid by customer',
            'total_services': 'Number of services subscribed',
            'service_adoption_rate': 'Percentage of available services used',
            'has_internet': 'Has internet service',
            'has_phone': 'Has phone service',
            'has_streaming': 'Has streaming services',
            'is_monthly_contract': 'Month-to-month contract',
            'avg_monthly_charge': 'Average monthly charge based on tenure',
            'revenue_per_month': 'Average revenue per month'
        }
        return mapping


def load_and_preprocess_telecom_data(file_path, target_col='Churn'):
    """
    Main function to load and preprocess telecommunications customer data
    
    Args:
        file_path: path to the CSV file
        target_col: name of the target variable column
        
    Returns:
        tuple: (X_processed, y, feature_names, preprocessor)
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Initialize preprocessor
    preprocessor = MixedDataPreprocessor()
    
    # Preprocess features
    X_processed, feature_names = preprocessor.fit_transform(df, target_col)
    
    # Extract target variable if exists
    y = None
    if target_col in df.columns:
        y = df[target_col]
        if y.dtype == 'object':
            y = (y == 'Yes').astype(int)  # Convert Yes/No to 1/0
    
    return X_processed, y, feature_names, preprocessor


if __name__ == "__main__":
    # Example usage
    file_path = "../data/telecom_customer_churn.csv"
    X, y, features, preprocessor = load_and_preprocess_telecom_data(file_path)
    
    print(f"Processed data shape: {X.shape}")
    print(f"Number of features: {len(features)}")
    print(f"Features: {features[:10]}...")  # First 10 features
    
    if y is not None:
        print(f"Target distribution: {pd.Series(y).value_counts()}")
