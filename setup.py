#!/usr/bin/env python3
"""
Customer Segmentation Project Setup Script

This script helps set up the customer segmentation project environment
and provides utilities for running the analysis.
"""

import os
import sys
import subprocess
import platform


def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              Customer Segmentation for Telecom              ║
    ║                     Setup & Utilities                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    else:
        print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")


def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    venv_path = "venv"
    
    if not os.path.exists(venv_path):
        print("📦 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            print("✅ Virtual environment created successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            sys.exit(1)
    else:
        print("✅ Virtual environment already exists")


def get_activation_command():
    """Get activation command based on OS"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"


def install_requirements():
    """Install required packages"""
    print("📚 Installing required packages...")
    
    # Determine pip path
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    try:
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        sys.exit(1)


def setup_jupyter():
    """Set up Jupyter kernel"""
    print("🔬 Setting up Jupyter kernel...")
    
    if platform.system() == "Windows":
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    try:
        subprocess.run([python_path, "-m", "ipykernel", "install", "--user", 
                       "--name=customer-segmentation", 
                       "--display-name=Customer Segmentation"], check=True)
        print("✅ Jupyter kernel installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️ Warning: Failed to install Jupyter kernel")


def check_data_files():
    """Check if data files exist"""
    data_file = "data/telecom_customer_churn.csv"
    
    if os.path.exists(data_file):
        print("✅ Dataset found")
        # Show basic info about the dataset
        import pandas as pd
        df = pd.read_csv(data_file)
        print(f"   📊 Dataset shape: {df.shape}")
        print(f"   👥 Number of customers: {df.shape[0]:,}")
        print(f"   📈 Features: {df.shape[1]}")
    else:
        print("❌ Dataset not found")
        print("   Please ensure the dataset is downloaded to data/telecom_customer_churn.csv")


def run_quick_test():
    """Run a quick test to ensure everything is working"""
    print("🧪 Running quick test...")
    
    if platform.system() == "Windows":
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    test_code = '''
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
print("✅ All core libraries imported successfully")
print("🎯 Setup test passed!")
'''
    
    try:
        result = subprocess.run([python_path, "-c", test_code], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Setup test failed:")
        print(e.stderr)


def print_next_steps():
    """Print next steps for the user"""
    activation_cmd = get_activation_command()
    
    next_steps = f"""
    🎉 Setup completed successfully!
    
    Next steps:
    1. Activate the virtual environment:
       {activation_cmd}
       
    2. Launch Jupyter Notebook:
       jupyter notebook
       
    3. Open the main analysis notebook:
       notebooks/customer_segmentation_analysis.ipynb
       
    4. Select the 'Customer Segmentation' kernel when prompted
    
    📚 Documentation:
    - README.md - Project overview and setup instructions
    - reports/Customer_Segmentation_Report.md - Comprehensive analysis report
    
    🔧 Utility modules:
    - src/data_preprocessing.py - Data cleaning and feature engineering
    - src/segmentation_models.py - Clustering and modeling utilities
    
    Happy analyzing! 🚀
    """
    print(next_steps)


def main():
    """Main setup function"""
    print_banner()
    
    print("🔍 Checking system requirements...")
    check_python_version()
    
    print("\n🏗️ Setting up project environment...")
    create_virtual_environment()
    install_requirements()
    setup_jupyter()
    
    print("\n📊 Checking project files...")
    check_data_files()
    
    print("\n🧪 Running setup validation...")
    run_quick_test()
    
    print_next_steps()


if __name__ == "__main__":
    main()
