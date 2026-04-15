from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

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

# LLM config
LLM_MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("API_KEY")

# Template path
TEMPLATE_DIR = BASE_DIR / "templates"
PROMPT_TEMPLATE_PATH = TEMPLATE_DIR / "prompt_v2.txt"