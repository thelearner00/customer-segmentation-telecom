"""
Customer Segmentation API Client Examples

This script demonstrates how to interact with the deployed Customer Segmentation API
using Python requests. It shows examples for all available endpoints.
"""

import requests
import json
from typing import Dict, List

# API Configuration
API_BASE_URL = "http://localhost:8080"
API_TIMEOUT = 30

class CustomerSegmentationClient:
    """
    Python client for Customer Segmentation API
    """
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = API_TIMEOUT
    
    def health_check(self) -> Dict:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_segments_summary(self) -> Dict:
        """Get summary of all customer segments"""
        response = self.session.get(f"{self.base_url}/segments/summary")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()
    
    def segment_customer(self, customer_data: Dict) -> Dict:
        """Segment a single customer"""
        response = self.session.post(
            f"{self.base_url}/segment",
            json=customer_data
        )
        response.raise_for_status()
        return response.json()
    
    def predict_high_value(self, customer_data: Dict) -> Dict:
        """Predict if customer is high-value"""
        response = self.session.post(
            f"{self.base_url}/predict",
            json=customer_data
        )
        response.raise_for_status()
        return response.json()
    
    def segment_batch(self, customers: List[Dict]) -> List[Dict]:
        """Segment multiple customers"""
        batch_data = {"customers": customers}
        response = self.session.post(
            f"{self.base_url}/segment-batch",
            json=batch_data
        )
        response.raise_for_status()
        return response.json()


def create_sample_customer() -> Dict:
    """Create a sample customer for testing"""
    return {
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


def create_budget_customer() -> Dict:
    """Create a budget-conscious customer example"""
    return {
        "customerID": "BUDGET-001",
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 6,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 25.50,
        "TotalCharges": 153.00
    }


def main():
    """
    Main function demonstrating API usage
    """
    print("🎯 Customer Segmentation API Client Examples")
    print("=" * 60)
    
    # Initialize client
    client = CustomerSegmentationClient()
    
    try:
        # 1. Health Check
        print("\n1. 🔍 Health Check")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Models Loaded: {health['models_loaded']}")
        
        # 2. Get Segments Summary
        print("\n2. 📊 Segments Summary")
        segments = client.get_segments_summary()
        print(f"   Total Segments: {segments['total_segments']}")
        for segment in segments['segments']:
            print(f"   - {segment['name']}: {segment['percentage']}% "
                  f"(${segment['avg_monthly_revenue']:.2f} avg revenue)")
        
        # 3. Get Model Info
        print("\n3. 🤖 Model Information")
        model_info = client.get_model_info()
        for model_name, info in model_info.items():
            print(f"   {model_name}: {info['type']} (Loaded: {info['loaded']})")
        
        # 4. Segment Premium Customer
        print("\n4. 🎯 Customer Segmentation - Premium Customer")
        premium_customer = create_sample_customer()
        segment_result = client.segment_customer(premium_customer)
        print(f"   Customer ID: {segment_result['customerID']}")
        print(f"   Segment: {segment_result['segment']} - {segment_result['segment_name']}")
        print(f"   Business Value: {segment_result['business_value']}")
        print(f"   Characteristics: {', '.join(segment_result['characteristics'])}")
        print(f"   Recommendations: {', '.join(segment_result['recommendations'])}")
        
        # 5. Segment Budget Customer
        print("\n5. 💰 Customer Segmentation - Budget Customer")
        budget_customer = create_budget_customer()
        budget_result = client.segment_customer(budget_customer)
        print(f"   Customer ID: {budget_result['customerID']}")
        print(f"   Segment: {budget_result['segment']} - {budget_result['segment_name']}")
        print(f"   Business Value: {budget_result['business_value']}")
        print(f"   Characteristics: {', '.join(budget_result['characteristics'])}")
        
        # 6. High-Value Prediction
        print("\n6. 💎 High-Value Customer Prediction")
        prediction_result = client.predict_high_value(premium_customer)
        print(f"   Customer ID: {prediction_result['customerID']}")
        print(f"   High-Value Prediction: {'Yes' if prediction_result['high_value_prediction'] else 'No'}")
        print(f"   Probability: {prediction_result['high_value_probability']:.2%}")
        print(f"   Priority Score: {prediction_result['priority_score']:.3f}")
        print(f"   Recommendations: {', '.join(prediction_result['recommendations'])}")
        
        # 7. Batch Segmentation
        print("\n7. 📦 Batch Segmentation")
        customers = [premium_customer, budget_customer]
        batch_results = client.segment_batch(customers)
        print(f"   Processed {len(batch_results)} customers:")
        for result in batch_results:
            print(f"   - {result['customerID']}: Segment {result['segment']} "
                  f"({result['business_value']} value)")
        
        print(f"\n✅ All API examples completed successfully!")
        print(f"\n🌐 API Documentation: {client.base_url}/docs")
        print(f"📋 Interactive API: {client.base_url}/redoc")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API Error: {str(e)}")
        print("   Make sure the API server is running at http://localhost:8080")
        print("   You can start it with: ./deploy.sh dev")
    
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}")


if __name__ == "__main__":
    main()
