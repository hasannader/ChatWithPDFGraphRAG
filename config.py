"""
config.py
Centralized configuration management for the Graph RAG PDF Chatbot.
Loads environment variables from .env file.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Application Configuration
MAX_PDF_SIZE_MB = 50
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_DIM = 1536
