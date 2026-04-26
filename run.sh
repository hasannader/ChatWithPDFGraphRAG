#!/bin/bash

# Startup script for Graph RAG PDF Chatbot on Mac/Linux

echo ""
echo "========================================"
echo "Graph RAG PDF Chatbot - Startup"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: .env file not found!"
    echo "Please create a .env file with your credentials."
    echo "You can use .env.example as a template."
    echo ""
    read -p "Press Enter to continue..."
fi

# Start Streamlit app
echo ""
echo "Starting Streamlit application..."
echo "Open http://localhost:8501 in your browser"
echo ""
streamlit run app.py
