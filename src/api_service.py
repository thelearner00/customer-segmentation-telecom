"""
Customer Segmentation API Service

A FastAPI-based REST API for customer segmentation and prediction services.
Provides endpoints for clustering customers, predicting high-value customers,
and generating business insights.
"""

import os
import sys
import json
import pickle
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_preprocessing import MixedDataPreprocessor
from segmentation_models import CustomerSegmentationModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Segmentation API",
    description="Machine Learning API for customer segmentation and value prediction in telecommunications industry",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
segmentation_model = None
prediction_model = None
preprocessor = None
feature_names = None

# Pydantic models for API
class CustomerData(BaseModel):
    """Customer data input model"""
    customerID: Optional[str] = None
    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen flag (0/1)")
    Partner: str = Field(..., description="Has partner (Yes/No)")
    Dependents: str = Field(..., description="Has dependents (Yes/No)")
    tenure: int = Field(..., ge=0, description="Customer tenure in months")
    PhoneService: str = Field(..., description="Has phone service (Yes/No)")
    MultipleLines: str = Field(..., description="Multiple lines (Yes/No/No phone service)")
    InternetService: str = Field(..., description="Internet service type (DSL/Fiber optic/No)")
    OnlineSecurity: str = Field(..., description="Online security (Yes/No/No internet service)")
    OnlineBackup: str = Field(..., description="Online backup (Yes/No/No internet service)")
    DeviceProtection: str = Field(..., description="Device protection (Yes/No/No internet service)")
    TechSupport: str = Field(..., description="Tech support (Yes/No/No internet service)")
    StreamingTV: str = Field(..., description="Streaming TV (Yes/No/No internet service)")
    StreamingMovies: str = Field(..., description="Streaming movies (Yes/No/No internet service)")
    Contract: str = Field(..., description="Contract type (Month-to-month/One year/Two year)")
    PaperlessBilling: str = Field(..., description="Paperless billing (Yes/No)")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in dollars")
    TotalCharges: float = Field(..., ge=0, description="Total charges in dollars")


class BatchCustomerData(BaseModel):
    """Batch customer data input model"""
    customers: List[CustomerData]


class SegmentationResponse(BaseModel):
    """Segmentation response model"""
    customerID: Optional[str]
    segment: int
    segment_name: str
    characteristics: List[str]
    business_value: str
    recommendations: List[str]
    confidence_score: float


class PredictionResponse(BaseModel):
    """Prediction response model"""
    customerID: Optional[str]
    high_value_prediction: int
    high_value_probability: float
    priority_score: float
    segment: int
    recommendations: List[str]


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str
    models_loaded: bool


# Startup event to load models
@app.on_event("startup")
async def startup_event():
    """Load models and preprocessor on startup"""
    global segmentation_model, prediction_model, preprocessor, feature_names
    
    try:
        logger.info("Loading models and preprocessor...")
        
        # Initialize preprocessor and load sample data for training
        preprocessor = MixedDataPreprocessor()
        
        # Check if pre-trained models exist, otherwise train new ones
        model_dir = "/app/output/models"
        data_path = "/app/data/telecom_customer_churn.csv"
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Load and preprocess data
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            # Preprocess data
            X_processed, feature_names = preprocessor.fit_transform(df, 'Churn')
            
            # Train segmentation model if not exists
            seg_model_path = f"{model_dir}/segmentation_model.pkl"
            if os.path.exists(seg_model_path):
                with open(seg_model_path, 'rb') as f:
                    segmentation_model = pickle.load(f)
            else:
                segmentation_model = CustomerSegmentationModel()
                segmentation_model.fit_clustering_model(X_processed, 4, 'kmeans')
                
                # Save model
                with open(seg_model_path, 'wb') as f:
                    pickle.dump(segmentation_model, f)
            
            # Train prediction model if not exists
            pred_model_path = f"{model_dir}/prediction_model.pkl"
            if os.path.exists(pred_model_path):
                with open(pred_model_path, 'rb') as f:
                    prediction_model = pickle.load(f)
            else:
                # Create target variable (high-value customers)
                y = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)
                
                prediction_model = RandomForestClassifier(n_estimators=100, random_state=42)
                prediction_model.fit(X_processed, y)
                
                # Save model
                with open(pred_model_path, 'wb') as f:
                    pickle.dump(prediction_model, f)
            
            logger.info("Models loaded successfully")
        else:
            logger.warning("Training data not found. Models will be trained on-demand.")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Segmentation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        models_loaded=segmentation_model is not None and prediction_model is not None
    )


@app.post("/segment", response_model=SegmentationResponse, tags=["Segmentation"])
async def segment_customer(customer: CustomerData):
    """Segment a single customer"""
    try:
        if segmentation_model is None or preprocessor is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Convert to DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])
        
        # Preprocess
        X_processed, _ = preprocessor.fit_transform(df)
        
        # Predict segment
        segment = segmentation_model.model.predict(X_processed)[0]
        
        # Generate segment insights (simplified for demo)
        segment_names = {
            0: "Budget-Conscious Customers",
            1: "Standard Value Customers", 
            2: "Premium Loyal Customers",
            3: "High-Value Enterprise Customers"
        }
        
        characteristics_map = {
            0: ["Price-sensitive", "High churn risk", "Low service adoption"],
            1: ["Moderate revenue", "Balanced service usage", "Growth potential"],
            2: ["High-value", "Low churn", "Technology adopters"],
            3: ["Highest revenue", "Complete service portfolio", "Enterprise focus"]
        }
        
        recommendations_map = {
            0: ["Retention campaigns", "Value bundling", "Contract incentives"],
            1: ["Upselling opportunities", "Cross-selling", "Loyalty programs"],
            2: ["VIP treatment", "Innovation access", "Referral programs"],
            3: ["Account management", "Custom solutions", "Premium support"]
        }
        
        return SegmentationResponse(
            customerID=customer.customerID,
            segment=int(segment),
            segment_name=segment_names.get(segment, "Unknown"),
            characteristics=characteristics_map.get(segment, []),
            business_value="High" if segment >= 2 else "Medium" if segment == 1 else "Low",
            recommendations=recommendations_map.get(segment, []),
            confidence_score=0.85  # Simplified confidence score
        )
        
    except Exception as e:
        logger.error(f"Error in segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_high_value(customer: CustomerData):
    """Predict if customer is high-value"""
    try:
        if prediction_model is None or preprocessor is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Convert to DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])
        
        # Preprocess
        X_processed, _ = preprocessor.fit_transform(df)
        
        # Predict
        prediction = prediction_model.predict(X_processed)[0]
        probability = prediction_model.predict_proba(X_processed)[0, 1]
        
        # Get segment
        segment = segmentation_model.model.predict(X_processed)[0]
        
        # Calculate priority score
        priority_score = (
            probability * 0.4 + 
            (customer.MonthlyCharges / 100) * 0.3 + 
            0.3  # Simplified retention likelihood
        )
        
        # Generate recommendations
        recommendations = []
        if prediction == 1:
            recommendations = ["Maintain premium service", "Offer loyalty rewards"]
        else:
            recommendations = ["Upsell streaming services", "Promote fiber optic"]
            
        return PredictionResponse(
            customerID=customer.customerID,
            high_value_prediction=int(prediction),
            high_value_probability=float(probability),
            priority_score=float(priority_score),
            segment=int(segment),
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment-batch", response_model=List[SegmentationResponse], tags=["Batch Operations"])
async def segment_customers_batch(batch_data: BatchCustomerData):
    """Segment multiple customers at once"""
    try:
        if len(batch_data.customers) > 1000:
            raise HTTPException(status_code=400, detail="Batch size too large. Maximum 1000 customers.")
        
        results = []
        for customer in batch_data.customers:
            result = await segment_customer(customer)
            results.append(result)
            
        return results
        
    except Exception as e:
        logger.error(f"Error in batch segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segments/summary", tags=["Analytics"])
async def get_segments_summary():
    """Get summary of all customer segments"""
    try:
        segments_info = {
            "total_segments": 4,
            "segments": [
                {
                    "id": 0,
                    "name": "Budget-Conscious Customers",
                    "percentage": 26.5,
                    "avg_monthly_revenue": 32.37,
                    "churn_rate": 31.9,
                    "strategy": "Retention and value bundling"
                },
                {
                    "id": 1,
                    "name": "Standard Value Customers", 
                    "percentage": 27.0,
                    "avg_monthly_revenue": 64.76,
                    "churn_rate": 17.9,
                    "strategy": "Upselling and cross-selling"
                },
                {
                    "id": 2,
                    "name": "Premium Loyal Customers",
                    "percentage": 23.4,
                    "avg_monthly_revenue": 74.89,
                    "churn_rate": 11.2,
                    "strategy": "VIP treatment and innovation"
                },
                {
                    "id": 3,
                    "name": "High-Value Enterprise Customers",
                    "percentage": 23.1,
                    "avg_monthly_revenue": 103.79,
                    "churn_rate": 26.8,
                    "strategy": "Account management and custom solutions"
                }
            ]
        }
        return segments_info
        
    except Exception as e:
        logger.error(f"Error getting segments summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about loaded models"""
    try:
        model_info = {
            "segmentation_model": {
                "type": "K-Means Clustering",
                "n_clusters": 4,
                "loaded": segmentation_model is not None
            },
            "prediction_model": {
                "type": "Random Forest Classifier",
                "loaded": prediction_model is not None,
                "target": "High-value customer prediction"
            },
            "preprocessor": {
                "type": "Mixed Data Preprocessor",
                "loaded": preprocessor is not None,
                "features": len(feature_names) if feature_names else 0
            }
        }
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1
    )
