from flask import Flask, render_template, request, jsonify, make_response
from src.preprocess import PREPROCESSING
from src.predictor import PREDICTOR
from src.llm_engine import CustomerData, CreditLLMEngine
import os


app = Flask(__name__)
LLM_ENGINE = CreditLLMEngine()  


@app.route('/')
def home():
    response = make_response(render_template('home.html'))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route('/predict-page')
def predict_page():
    response = make_response(render_template('index.html'))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route('/predict', methods=['POST'])
def predict():
    try:

        user_input = request.get_json()
        print(user_input)
        if not user_input:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided'
            }), 400


        processor = PREPROCESSING(user_input)
        processed_data = processor.run()


        predictor = PREDICTOR(processed_data)
        ml_prediction = predictor.run()[0]  
        ml_output = "High Risk" if ml_prediction == "bad" else "Low Risk"

        llm_payload = {
            **user_input,
            "Risk": ml_output
        }

        customer = CustomerData(**llm_payload)
        llm_text = LLM_ENGINE.get_description(customer)
        print(llm_text)
        return jsonify({
            'status': 'success',
            'ml_output': ml_output,
            'risk_class': "high" if ml_prediction == "bad" else "low",   
            'llm_output': llm_text
        })

    except Exception as e:
        print(f"[ERROR] {e}")  

        return jsonify({
            'status': 'error',
            'message': 'Something went wrong on the server'
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))