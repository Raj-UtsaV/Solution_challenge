from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Base directory (project root safe)
BASE_DIR = Path(__file__).resolve().parent

# Data directories
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# File paths
RAW_FILE_PATH = RAW_DIR / "german_credit_data.csv"
PROCESSED_FILE_PATH = PROCESSED_DIR / "processed_data.csv"

# Model paths
MODELS_DIR = BASE_DIR / "models"
ENCODER_FILE_PATH = MODELS_DIR / "encoder.pkl"
MODEL_FILE_PATH = MODELS_DIR / "model.pkl"
SCALER_FILE_PATH = MODELS_DIR / "scaler.pkl"

# Environment variables
API_KEY = os.getenv("API_KEY")

# LLM config
LLM_MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"