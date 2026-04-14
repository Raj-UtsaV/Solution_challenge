from flask import Flask, render_template, request, jsonify
import joblib
from src.preprocess import PREPROCESSING
from src.predictor import PREDICTOR
from src.llm_engine import CustomerData,CreditLLMEngine

LLM_ENGINE = CreditLLMEngine()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from the frontend form
        user_input = request.json
        
        # 2. Run your Preprocessing logic
        processor = PREPROCESSING(user_input)
        processed_data = processor.run()
        
        # 3. Run your ML Prediction logic
        predictor = PREDICTOR(processed_data)
        ml_prediction = int(predictor.run()[0]) # Convert numpy int to Python int
        
        
        # 4. Get LLM Analysis
#         example_row = {
# #         "Age": 33, "Job": 2, "Housing": "own", 
# #         "Saving accounts": "little", "Checking account": "moderate",
# #         "Credit amount": 1169, "Duration": 6, "Purpose": "radio/TV", "Risk": "good"
# #     }

# #     # Create Pydantic model from dict
# #     customer = CustomerData(**example_row)
        llm_text = LLM_ENGINE.get_description(customer)
        
        return jsonify({
            'ml_output': "Low Risk" if ml_prediction == 1 else "High Risk",
            'llm_output': llm_text,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)