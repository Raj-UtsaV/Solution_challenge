# 🏦 KarzMitra — AI-Powered Loan Risk Predictor & Analyzer

> Predict loan risk instantly with Machine Learning and get a plain-English explanation powered by a Large Language Model (LLM / Generative AI).

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Visit_Now-4CAF50?style=for-the-badge)](https://solution-challenge-eg1f.onrender.com/)

---

## 📌 About

**LoanSense** is a full-stack AI web application that predicts whether a loan applicant is **High Risk** or **Low Risk** using a trained Machine Learning model. Beyond a binary prediction, it passes the result to an **LLM (Generative AI) engine** that produces a detailed, human-readable analysis of the applicant's financial profile — helping lenders understand *why* a decision was made, not just what it is.

Built as part of the **Google Solution Challenge 2026**, the project combines classical ML with modern Generative AI to make credit risk assessment smarter and more transparent.

---

## ✨ Features

- 🤖 **ML Risk Prediction** — Trained classifier predicts loan risk (`High Risk` / `Low Risk`)
- 🧠 **LLM Analysis** — Generative AI engine produces a natural-language explanation of the risk assessment
- 🌐 **Web Interface** — Clean, responsive UI built with HTML, CSS, and Flask
- 📓 **Jupyter Notebooks** — Full model training and EDA workflow included

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask |
| **ML / Data** | scikit-learn, pandas, numpy, scipy |
| **Generative AI** | OpenAI API (LLM engine via `llm_engine.py`) |
| **Frontend** | HTML5, CSS3, Jinja2 Templates |
| **Notebooks** | Jupyter, matplotlib |
| **Deployment** | Gunicorn, Render |
| **Config** | python-dotenv |

---

## 📁 Folder Structure

```
Solution_challenge/
│
├── app.py                  # Flask application entry point
├── __init__.py
├── requirements.txt        # Python dependencies
├── .gitignore
│
├── src/                    # Core application logic
│   ├── preprocess.py       # Data preprocessing pipeline (PREPROCESSING class)
│   ├── predictor.py        # ML model inference (PREDICTOR class)
│   └── llm_engine.py       # LLM/GenAI analysis engine (CreditLLMEngine)
│
├── models/                 # Trained ML model artifacts
│
├── data/                   # Training and reference datasets
│
├── notebooks/              # Jupyter notebooks for EDA and model training
│
├── config/                 # Configuration files
│
├── templates/              # Jinja2 HTML templates
│   ├── home.html           # Landing page
│   └── index.html          # Prediction form page
│
└── static/                 # CSS, JS, and static assets
```

---

## 🚀 How It Works

 1. User fills in the applicant's financial details on the web form.
 2. The form POSTs to `/predict` as JSON.
 3. **`PREPROCESSING`** cleans and encodes the input.
 4. **`PREDICTOR`** runs the trained ML model and returns `"bad"` (High Risk) or `"good"` (Low Risk).
 5. The prediction is passed to **`CreditLLMEngine`**, which calls the LLM API to generate a natural-language analysis.
 6. Both the ML result and the LLM explanation are returned to the user in real time.

---

## 💻 Run Locally

### Prerequisites

- Python 3.10+
- An OpenAI API key (or compatible LLM provider key)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Raj-UtsaV/Solution_challenge.git
cd Solution_challenge

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file in the root directory and add your API key:
echo "API_KEY=your_api_key_here" > .env

# 5. Run the Flask app
python app.py
```

Then open your browser and visit: **http://localhost:5000**

> ⚠️ The app requires a valid LLM API key to generate the AI-powered analysis. ML prediction will still work without it.

---

## 🌐 Live Demo

The app is deployed and publicly accessible:

👉 **[https://solution-challenge-eg1f.onrender.com/](https://solution-challenge-eg1f.onrender.com/)**

> Note: The app is hosted on Render's free tier and may take ~30 seconds to wake up on the first visit.

---

## 👥 Contributors

Built with ❤️ as part of the **Google Solution Challenge 2026**.

* [Utsav Raj](https://github.com/Raj-UtsaV)
* [Aditya Bansal](https://github.com/AdityaBansal0123)
* [Shivay Saurya](https://github.com/sauryashivay)

---

## 📄 License

This project is open source. Feel free to fork, explore, and build upon it.

---

[🌐 Visit Live App](https://solution-challenge-eg1f.onrender.com/) · [🐛 Report Bug](https://github.com/Raj-UtsaV/Solution_challenge/issues) · [💬 Discussions](https://github.com/Raj-UtsaV/Solution_challenge/discussions)
