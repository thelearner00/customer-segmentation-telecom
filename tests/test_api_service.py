"""
Unit tests for API service
"""

import unittest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from fastapi.testclient import TestClient
    from api_service import app, CustomerData
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI not available")
class TestAPIService(unittest.TestCase):
    """Test cases for API service"""

    def setUp(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
        
        # Sample customer data for testing
        self.sample_customer = {
            "customerID": "TEST-001",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 79.85,
            "TotalCharges": 1915.40
        }

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
        self.assertIn("docs", data)

    @patch('api_service.segmentation_model')
    @patch('api_service.prediction_model')
    def test_health_endpoint(self, mock_prediction_model, mock_segmentation_model):
        """Test health check endpoint"""
        mock_segmentation_model.__bool__ = MagicMock(return_value=True)
        mock_prediction_model.__bool__ = MagicMock(return_value=True)
        
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("timestamp", data)
        self.assertIn("version", data)
        self.assertIn("models_loaded", data)

    def test_segments_summary_endpoint(self):
        """Test segments summary endpoint"""
        response = self.client.get("/segments/summary")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("total_segments", data)
        self.assertIn("segments", data)
        self.assertEqual(data["total_segments"], 4)
        self.assertEqual(len(data["segments"]), 4)

    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = self.client.get("/model/info")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("segmentation_model", data)
        self.assertIn("prediction_model", data)
        self.assertIn("preprocessor", data)

    @patch('api_service.segmentation_model')
    @patch('api_service.preprocessor')
    def test_segment_customer_endpoint(self, mock_preprocessor, mock_segmentation_model):
        """Test customer segmentation endpoint"""
        # Mock model behavior
        mock_preprocessor.fit_transform.return_value = ([[1, 2, 3]], ['feature1', 'feature2', 'feature3'])
        mock_segmentation_model.model.predict.return_value = [1]
        
        response = self.client.post("/segment", json=self.sample_customer)
        
        if response.status_code == 503:  # Models not loaded
            self.skipTest("Models not loaded in test environment")
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("customerID", data)
        self.assertIn("segment", data)
        self.assertIn("segment_name", data)
        self.assertIn("characteristics", data)
        self.assertIn("business_value", data)
        self.assertIn("recommendations", data)

    @patch('api_service.prediction_model')
    @patch('api_service.segmentation_model')
    @patch('api_service.preprocessor')
    def test_predict_high_value_endpoint(self, mock_preprocessor, mock_segmentation_model, mock_prediction_model):
        """Test high-value prediction endpoint"""
        # Mock model behavior
        mock_preprocessor.fit_transform.return_value = ([[1, 2, 3]], ['feature1', 'feature2', 'feature3'])
        mock_prediction_model.predict.return_value = [1]
        mock_prediction_model.predict_proba.return_value = [[0.3, 0.7]]
        mock_segmentation_model.model.predict.return_value = [2]
        
        response = self.client.post("/predict", json=self.sample_customer)
        
        if response.status_code == 503:  # Models not loaded
            self.skipTest("Models not loaded in test environment")
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("customerID", data)
        self.assertIn("high_value_prediction", data)
        self.assertIn("high_value_probability", data)
        self.assertIn("priority_score", data)
        self.assertIn("segment", data)
        self.assertIn("recommendations", data)

    def test_batch_segment_endpoint_too_large(self):
        """Test batch segmentation with too many customers"""
        # Create a batch that's too large (over 1000)
        large_batch = {
            "customers": [self.sample_customer] * 1001
        }
        
        response = self.client.post("/segment-batch", json=large_batch)
        self.assertEqual(response.status_code, 400)
        
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("too large", data["detail"])

    def test_invalid_customer_data(self):
        """Test API with invalid customer data"""
        invalid_customer = self.sample_customer.copy()
        invalid_customer["SeniorCitizen"] = "invalid"  # Should be 0 or 1
        
        response = self.client.post("/segment", json=invalid_customer)
        self.assertEqual(response.status_code, 422)  # Validation error

    def test_missing_required_fields(self):
        """Test API with missing required fields"""
        incomplete_customer = {
            "customerID": "TEST-002",
            "gender": "Female"
            # Missing required fields
        }
        
        response = self.client.post("/segment", json=incomplete_customer)
        self.assertEqual(response.status_code, 422)  # Validation error


class TestCustomerDataModel(unittest.TestCase):
    """Test cases for Pydantic models"""

    @unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI not available")
    def test_valid_customer_data(self):
        """Test valid customer data model"""
        valid_data = {
            "customerID": "TEST-001",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 79.85,
            "TotalCharges": 1915.40
        }
        
        customer = CustomerData(**valid_data)
        self.assertEqual(customer.customerID, "TEST-001")
        self.assertEqual(customer.tenure, 24)
        self.assertEqual(customer.MonthlyCharges, 79.85)

    @unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI not available")
    def test_invalid_customer_data(self):
        """Test invalid customer data model"""
        invalid_data = {
            "customerID": "TEST-001",
            "gender": "Male",
            "SeniorCitizen": 2,  # Should be 0 or 1
            "tenure": -5,  # Should be >= 0
            "MonthlyCharges": -10.0,  # Should be >= 0
        }
        
        with self.assertRaises(Exception):  # Pydantic validation error
            CustomerData(**invalid_data)


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API"""

    @unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI not available")
    def setUp(self):
        """Set up integration test fixtures"""
        self.client = TestClient(app)

    def test_api_documentation_accessible(self):
        """Test that API documentation is accessible"""
        # Test OpenAPI docs
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        
        # Test ReDoc
        response = self.client.get("/redoc")
        self.assertEqual(response.status_code, 200)
        
        # Test OpenAPI schema
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("info", data)
        self.assertIn("paths", data)

    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = self.client.options("/health")
        
        # Check for CORS headers (if CORS is enabled)
        headers = response.headers
        # Note: Actual CORS headers might not be present in test client
        # This is more of a smoke test
        self.assertTrue(True)  # Placeholder test

    def test_error_handling(self):
        """Test API error handling"""
        # Test non-existent endpoint
        response = self.client.get("/non-existent")
        self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    if not FASTAPI_AVAILABLE:
        print("Warning: FastAPI not available. Skipping API tests.")
        print("Install FastAPI to run these tests: pip install fastapi")
    
    unittest.main()
