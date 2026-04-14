import os
from dotenv import load_dotenv
load_dotenv()


RAW_DIR = "data/raw"
PROCESSESED_DIR = "data/processed"
RAW_FILE_PATH = os.path.join(RAW_DIR,"german_credit_data.csv")
PROCESSED_FILE_PATH = os.path.join(PROCESSESED_DIR,"german_credit_data.csv")

MODELS_DIR = "models"
ENCODER_FILE_PATH = os.path.join(MODELS_DIR,"encoder.pkl")
MODEL_FILE_PATH = os.path.join(MODELS_DIR,"model.pkl")
SCALER_FILE_PATH = os.path.join(MODELS_DIR,"scaler.pkl")

API_KEY = os.getenv("API_KEY")
LLM_MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"
