# 💳 Credit Risk Prediction System

A Streamlit-powered machine learning application that predicts whether a loan will be approved based on an applicant's financial data. It features real-time predictions, SHAP explainability, visual analytics, and bulk CSV upload support.

---

## 🚀 Features

- 🔮 **Real-time Predictions**: Predict loan approvals instantly
- ⚖️ **Threshold Control**: Adjust decision threshold to tune model sensitivity
- 📁 **Bulk Upload**: Upload CSV files for batch predictions
- 📊 **Model Visualizations**: Feature importance, probability distributions, and threshold analysis
- 📈 **SHAP Explainability**: Understand individual predictions using SHAP waterfall plots
- 🌐 **Interactive Prediction Surface**: 3D probability surfaces for visual insight

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python, [scikit-learn](https://scikit-learn.org/)
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly, Matplotlib, Seaborn
- **Model Explainability**: SHAP
- **Model Serialization**: Joblib

---

## 📂 Project Structure

Credit-Risk-Prediction/
│
├── app.py # Main Streamlit application
├── models/
│ └── credit_risk_model_v2.pkl # Trained ML model
├── images/
│ ├── Harvestly.png # Logo
│ └── Divyanshi.png # Profile image
├── requirements.txt # Required Python packages
└── README.md # This file

---

## ⚙️ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Divyanshi88/Credit-Risk-prediction.git
cd Credit-Risk-prediction
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
Create a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
🧪 Sample Input Format
For bulk CSV predictions, your file should include the following columns:

yaml
Copy
Edit
Employed, Bank Balance, Annual Salary
1, 5000, 45000
0, 2000, 25000
🙋‍♀️ About the Developer
👩‍💻 Divyanshi Sharma
B.Tech CSE (Data Science) | Machine Learning Enthusiast | Data Analyst
🔗 Portfolio • GitHub • LinkedIn
📧 divyanshi12023@gmail.com

📜 License
This project is for educational and demonstration purposes only.

yaml
Copy
Edit

---

### ✅ To Use:

1. Copy the content into a new file:  
   `Credit Risk Prediction System/README.md`

2. (Optional) Add it to Git and push:
   ```bash
   git add README.md
   git commit -m "Add README.md"
   git push
