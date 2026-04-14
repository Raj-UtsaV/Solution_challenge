from flask import Flask, render_template, request, jsonify
import joblib
from src.preprocess import PREPROCESSING
from src.predictor import PREDICTOR
from src.llm_engine import CustomerData,CreditLLMEngine
import pandas as pd
LLM_ENGINE = CreditLLMEngine()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from the frontend
        user_input = request.json  # This is already a dict
        
        # 2. ML Preprocessing & Prediction
        processor = PREPROCESSING(user_input)
        processed_data = processor.run()
        
        predictor = PREDICTOR(processed_data)
        ml_prediction = int(predictor.run()[0]) 
        
        # 3. Create the Risk label
        risk_label = "good" if ml_prediction == 1 else "bad"
            
        # 4. Prepare LLM Input (Merge the two dicts)
        # This takes all fields from user_input and adds/updates "Risk"
        llm_payload = {**user_input, "Risk": risk_label}
        
        # 5. Get LLM Analysis using the Pydantic model
        customer = CustomerData(**llm_payload)
        llm_text = LLM_ENGINE.get_description(customer)
        
        return jsonify({
            'ml_output': "Low Risk" if ml_prediction == 1 else "High Risk",
            'llm_output': llm_text,
            'status': 'success'
        })
        
    except Exception as e:
        # Log the error to your console so you can see exactly what went wrong
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)