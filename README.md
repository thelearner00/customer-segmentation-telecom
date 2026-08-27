# Customer Segmentation for Telecommunications

## Project Overview
This project implements customer segmentation using machine learning techniques for a telecommunications company. The goal is to identify distinct customer groups based on their behavior, demographics, and usage patterns to enable targeted marketing strategies and improve customer retention.

## Key Features
- Data preprocessing and cleaning for mixed data types (numerical and categorical)
- Exploratory Data Analysis (EDA) with comprehensive visualizations
- Implementation of multiple clustering algorithms
- Customer segment profiling and business insights
- Actionable recommendations for business strategies

## Project Structure
```
customer-segmentation-telecom/
├── data/                    # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Python source code modules
├── models/                 # Trained models and artifacts
├── reports/                # Analysis reports and documentation
├── visualizations/         # Generated plots and charts
└── requirements.txt        # Project dependencies
```

## Setup Instructions

### Option 1: Docker Deployment (Recommended for Production)
```bash
# Development deployment
./deploy.sh dev

# Production deployment with Nginx
./deploy.sh prod

# Deployment with Redis caching
./deploy.sh cache
```

### Option 2: Local Development
```bash
# Automated setup
python setup.py

# Manual setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
pip install -r requirements.txt
jupyter notebook
```

## Dataset
We'll be using a publicly available telecommunications customer dataset that includes:
- Customer demographics
- Service usage patterns
- Billing information
- Churn indicators
- Product/service subscriptions

## Methodology
1. **Data Exploration**: Understand data structure, quality, and patterns
2. **Data Cleaning**: Handle missing values, outliers, and inconsistencies
3. **Feature Engineering**: Create meaningful features and encode categorical variables
4. **Segmentation**: Apply clustering algorithms to identify customer segments
5. **Validation**: Evaluate segmentation quality and business relevance
6. **Insights**: Generate actionable business recommendations

## 🚀 Production API Deployment

This project includes a production-ready REST API built with FastAPI and containerized with Docker.

### Quick Start
```bash
# Deploy API in development mode
./deploy.sh dev

# Access API documentation
open http://localhost:8080/docs
```

### API Endpoints
- `GET /health` - Health check
- `POST /segment` - Segment a single customer
- `POST /predict` - Predict high-value customer
- `POST /segment-batch` - Batch customer segmentation
- `GET /segments/summary` - Get segment statistics
- `GET /model/info` - Model information

### API Usage Example
```python
import requests

# Customer data
customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "tenure": 24,
    "MonthlyCharges": 79.85,
    "Contract": "Two year",
    # ... other fields
}

# Get customer segment
response = requests.post(
    "http://localhost:8080/segment",
    json=customer
)
result = response.json()
print(f"Customer segment: {result['segment_name']}")
```

### Deployment Options

| Command | Description | Use Case |
|---------|-------------|----------|
| `./deploy.sh dev` | Basic API service | Development and testing |
| `./deploy.sh prod` | API + Nginx reverse proxy | Production deployment |
| `./deploy.sh cache` | API + Redis caching | High-performance production |
| `./deploy.sh stop` | Stop all services | Maintenance |
| `./deploy.sh clean` | Clean containers & volumes | Reset deployment |

### Container Features
- 🐳 **Dockerized**: Consistent deployment across environments
- 🔄 **Health Checks**: Built-in container health monitoring
- 🔒 **Security**: Non-root user, minimal attack surface
- 📊 **Monitoring**: Structured logging and metrics
- ⚡ **Performance**: Optimized for production workloads
- 🔧 **Scalable**: Ready for horizontal scaling with load balancer
