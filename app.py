from flask import Flask, render_template, request, jsonify
import joblib
# Import your custom classes
from src.preprocess import PREPROCESSING
from src.predictor import PREDICTOR

app = Flask(__name__)

# Mock function for LLM - Replace this with your actual LLM call (OpenAI, Gemini, etc.)
def get_llm_analysis(user_data, ml_result):
    """
    Generates a natural language explanation of the risk.
    """
    risk_status = "Low Risk" if ml_result == 1 else "High Risk"
    return f"Based on the applicant's profile (Age: {user_data['Age']}, Credit: {user_data['Credit amount']}), " \
           f"the system classifies this as {risk_status}. The credit amount relative to duration " \
           f"is a key factor in this decision."

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
        llm_text = get_llm_analysis(user_input, ml_prediction)
        
        return jsonify({
            'ml_output': "Low Risk" if ml_prediction == 1 else "High Risk",
            'llm_output': llm_text,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)