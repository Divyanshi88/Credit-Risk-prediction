import streamlit as st
st.set_page_config(
    page_title="Credit Risk Predictor", 
    layout="wide",
    page_icon="💳"
)

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load("models/credit_risk_model_v2.pkl")

model = load_model()

# --- Navigation ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["🏠 Home - Credit Risk Predictor", "📊 Model Visualization", "👩‍💻 About Us"]
)

# =============================================================================
# HOME PAGE - CREDIT RISK PREDICTOR
# =============================================================================
if page == "🏠 Home - Credit Risk Predictor":
    st.title("💳 Credit Risk Prediction System")
    st.image("./images/Harvestly.png", width=250)
    st.markdown("Predict whether a **loan will be approved** based on applicant financial information. Adjust the threshold to control sensitivity.")
    
    # --- Sidebar Inputs ---
    st.sidebar.header("🔍 Enter Applicant Details")
    
    employment_status = st.sidebar.selectbox("Are you employed?", ["Yes", "No"])
    bank_balance_raw = st.sidebar.text_input("Bank Balance ($)", "5000")
    annual_salary_raw = st.sidebar.text_input("Annual Salary ($)", "45000")
    threshold = st.sidebar.slider("⚖️ Classification Threshold", 0.0, 1.0, 0.3, 0.01)
    
    # --- Clean and Convert Input ---
    def clean_number(value):
        try:
            cleaned = str(value).replace(",", "").replace("$", "").strip()
            return float(cleaned) if cleaned else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    bank_balance = clean_number(bank_balance_raw)
    annual_salary = clean_number(annual_salary_raw)
    employed = 1 if employment_status.lower() == "yes" else 0
    
    sample_input = pd.DataFrame([{
        "Employed": employed,
        "Bank Balance": bank_balance,
        "Annual Salary": annual_salary
    }])
    
    # --- Debug Information ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Debug Info")
    st.sidebar.write(f"**Employed:** {employed}")
    st.sidebar.write(f"**Bank Balance:** {bank_balance}")
    st.sidebar.write(f"**Annual Salary:** {annual_salary}")
    
    # --- Prediction ---
    if st.sidebar.button("🔮 Predict"):
        try:
            # Ensure data types are correct
            sample_input = sample_input.astype({
                'Employed': 'int64',
                'Bank Balance': 'float64', 
                'Annual Salary': 'float64'
            })
            
            # Make prediction
            proba = model.predict_proba(sample_input)[0][1]
            prediction = int(proba > threshold)
            result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Denied"
            
            # Display results in main area
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Approval Probability", f"{proba:.4f}")
            
            with col2:
                st.metric("Threshold", f"{threshold}")
            
            with col3:
                if prediction:
                    st.success(result)
                else:
                    st.error(result)
            
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = proba,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Approval Probability"},
                delta = {'reference': threshold},
                gauge = {'axis': {'range': [None, 1]},
                        'bar': {'color': "darkgreen" if prediction else "darkred"},
                        'steps' : [
                            {'range': [0, threshold], 'color': "lightgray"},
                            {'range': [threshold, 1], 'color': "lightgreen"}],
                        'threshold' : {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': threshold}}))
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # SHAP Explainability
            try:
                st.subheader("📈 Explanation with SHAP")
                explainer = shap.Explainer(model)
                shap_values = explainer(sample_input)
                
                st.set_option('deprecation.showPyplotGlobalUse', False)
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.warning("SHAP explanation not available. Error: " + str(e))
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    # --- Bulk CSV Prediction ---
    st.markdown("---")
    st.subheader("📁 Upload CSV for Bulk Predictions")
    st.markdown("Your CSV must contain the columns: `Employed`, `Bank Balance`, and `Annual Salary`.")
    
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            required_cols = {"Employed", "Bank Balance", "Annual Salary"}
            if required_cols.issubset(df.columns):
                # Clean numeric fields
                for col in ["Bank Balance", "Annual Salary"]:
                    df[col] = df[col].astype(str).str.replace('[\$,]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                df["Employed"] = pd.to_numeric(df["Employed"], errors='coerce').fillna(0).astype(int)
                
                df = df.astype({
                    'Employed': 'int64',
                    'Bank Balance': 'float64',
                    'Annual Salary': 'float64'
                })
                
                probs = model.predict_proba(df[["Employed", "Bank Balance", "Annual Salary"]])[:, 1]
                df["Probability"] = probs
                df["Prediction"] = [
                    "✅ Loan Approved" if p > threshold else "❌ Loan Denied" for p in probs
                ]
                
                st.success("✅ Bulk predictions complete!")
                st.dataframe(df)
                
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Predictions as CSV", csv, "credit_risk_predictions.csv", "text/csv")
            else:
                st.error(f"❌ Your CSV is missing required columns. Found: {list(df.columns)}")
        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

# =============================================================================
# MODEL VISUALIZATION PAGE
# =============================================================================
elif page == "📊 Model Visualization":
    st.title("📊 Model Visualization & Analysis")
    st.markdown("Explore the credit risk model's behavior and performance metrics.")
    
    # Model Information
    st.header("🤖 Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Model Type:** {type(model).__name__}")
        if hasattr(model, 'feature_names_in_'):
            st.info(f"**Features:** {', '.join(model.feature_names_in_)}")
        if hasattr(model, 'n_features_in_'):
            st.info(f"**Number of Features:** {model.n_features_in_}")
    
    with col2:
        if hasattr(model, 'classes_'):
            st.info(f"**Classes:** {model.classes_}")
        if hasattr(model, 'n_estimators'):
            st.info(f"**Number of Estimators:** {model.n_estimators}")
        elif hasattr(model, 'max_depth'):
            st.info(f"**Max Depth:** {model.max_depth}")
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.header("🎯 Feature Importance")
        
        features = ['Employed', 'Bank Balance', 'Annual Salary']
        importances = model.feature_importances_
        
        fig_importance = px.bar(
            x=features,
            y=importances,
            title="Feature Importance",
            labels={'x': 'Features', 'y': 'Importance'},
            color=importances,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Probability Distribution Analysis
    st.header("📈 Probability Distribution Analysis")
    
    # Generate synthetic data for visualization
    np.random.seed(42)
    n_samples = 1000
    
    synthetic_data = pd.DataFrame({
        'Employed': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'Bank Balance': np.random.normal(8000, 5000, n_samples).clip(0, None),
        'Annual Salary': np.random.normal(50000, 20000, n_samples).clip(0, None)
    })
    
    synthetic_probs = model.predict_proba(synthetic_data)[:, 1]
    synthetic_data['Probability'] = synthetic_probs
    
    # Probability distribution histogram
    fig_dist = px.histogram(
        synthetic_data,
        x='Probability',
        nbins=50,
        title='Distribution of Approval Probabilities',
        labels={'Probability': 'Approval Probability', 'count': 'Frequency'}
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Feature vs Probability Analysis
    st.header("🔍 Feature Impact Analysis")
    
    feature_to_analyze = st.selectbox(
        "Select feature to analyze:",
        ['Bank Balance', 'Annual Salary', 'Employed']
    )
    
    if feature_to_analyze in ['Bank Balance', 'Annual Salary']:
        fig_scatter = px.scatter(
            synthetic_data,
            x=feature_to_analyze,
            y='Probability',
            color='Employed',
            title=f'{feature_to_analyze} vs Approval Probability',
            labels={'Probability': 'Approval Probability'},
            opacity=0.6
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        fig_box = px.box(
            synthetic_data,
            x='Employed',
            y='Probability',
            title='Employment Status vs Approval Probability'
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Threshold Analysis
    st.header("⚖️ Threshold Analysis")
    
    thresholds = np.linspace(0.1, 0.9, 50)
    metrics = []
    
    for thresh in thresholds:
        predictions = (synthetic_probs > thresh).astype(int)
        # Assuming roughly balanced classes for demonstration
        true_labels = (synthetic_probs > 0.5).astype(int)  # Simplified ground truth
        
        accuracy = np.mean(predictions == true_labels)
        precision = np.sum((predictions == 1) & (true_labels == 1)) / max(np.sum(predictions == 1), 1)
        recall = np.sum((predictions == 1) & (true_labels == 1)) / max(np.sum(true_labels == 1), 1)
        
        metrics.append({
            'Threshold': thresh,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    fig_metrics = px.line(
        metrics_df,
        x='Threshold',
        y=['Accuracy', 'Precision', 'Recall'],
        title='Model Performance vs Threshold',
        labels={'value': 'Score', 'variable': 'Metric'}
    )
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Interactive Prediction Surface
    st.header("🌐 Interactive Prediction Surface")
    
    # Create a grid for 2D visualization
    if st.checkbox("Show 2D Prediction Surface"):
        employed_val = st.radio("Employment Status:", [0, 1], index=1)
        
        balance_range = np.linspace(0, 20000, 50)
        salary_range = np.linspace(0, 100000, 50)
        
        B, S = np.meshgrid(balance_range, salary_range)
        
        grid_data = pd.DataFrame({
            'Employed': employed_val,
            'Bank Balance': B.ravel(),
            'Annual Salary': S.ravel()
        })
        
        grid_probs = model.predict_proba(grid_data)[:, 1].reshape(B.shape)
        
        fig_surface = go.Figure(data=[go.Surface(
            z=grid_probs,
            x=balance_range,
            y=salary_range,
            colorscale='viridis',
            name='Approval Probability'
        )])
        
        fig_surface.update_layout(
            title=f'Prediction Surface (Employed: {employed_val})',
            scene=dict(
                xaxis_title='Bank Balance ($)',
                yaxis_title='Annual Salary ($)',
                zaxis_title='Approval Probability'
            )
        )
        
        st.plotly_chart(fig_surface, use_container_width=True)

# =============================================================================
# ABOUT US PAGE
# =============================================================================
elif page == "👩‍💻 About Us":
    st.title("👩‍💻 About the Developer")
    
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🌟 Divyanshi Sharma
        
        **B.Tech Computer Science Student**  
        **Specialization: Data Science**
        
        Welcome to my Credit Risk Prediction System! This project showcases the power of machine learning 
        in financial technology, demonstrating how data science can be applied to real-world problems 
        in the banking and lending industry.
        """)
    
    with col2:
        # You can add a placeholder for profile image
        st.image("./images/Divyanshi.png", 
                 width=200, caption="Divyanshi Sharma")
    
    # Skills Section
    st.markdown("---")
    st.header("🚀 Technical Skills")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Programming Languages**
        - 🐍 Python
        - 💻 C/C++
        - 💾 SQL
        - 🌐 JavaScript
        """)
    
    with col2:
        st.markdown("""
        **Data Science & ML**
        - 🤖 Machine Learning
        - 📈 Statistical Analysis
        - 🔍 Data Mining
        - 🧠 Deep Learning
        """)
    
    with col3:
        st.markdown("""
        **Tools & Frameworks**
        - 🐼 Pandas, NumPy
        - 🔬 Scikit-learn
        - 📊 Matplotlib, Seaborn
        - 🌐 Streamlit
        """)
    
    # Project Details
    st.markdown("---")
    st.header("📋 About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This **Credit Risk Prediction System** is a comprehensive machine learning application that helps 
    financial institutions make informed decisions about loan approvals. The system analyzes various 
    factors to predict the likelihood of loan approval.
    
    ### 🔧 Key Features
    
    - **Real-time Predictions**: Get instant loan approval predictions
    - **Interactive Threshold Control**: Adjust sensitivity based on business requirements
    - **Bulk Processing**: Upload CSV files for batch predictions
    - **Model Explainability**: SHAP values for transparent decision-making
    - **Comprehensive Visualizations**: Understand model behavior and performance
    
    ### 🛠️ Technical Implementation
    
    - **Frontend**: Streamlit for interactive web interface
    - **Backend**: Python with scikit-learn for machine learning
    - **Visualization**: Plotly and Matplotlib for charts and graphs
    - **Explainability**: SHAP for model interpretability
    - **Data Processing**: Pandas and NumPy for data manipulation
    """)
    
    # Educational Journey
    st.markdown("---")
    st.header("📚 Educational Journey")
    
    st.markdown("""
    ### 🎓 Academic Background
    
    Currently pursuing **B.Tech in Computer Science** with a specialization in **Data Science**. 
    My academic journey has equipped me with:
    
    - Strong foundation in computer science fundamentals
    - Advanced knowledge in data science and machine learning
    - Hands-on experience with real-world projects
    - Understanding of statistical methods and algorithms
    
    ### 🌱 Learning Philosophy
    
    I believe in learning by doing. This project represents my commitment to applying theoretical 
    knowledge to solve practical problems in the financial sector.
    """)
    
    # Contact Section
    st.markdown("---")
    st.header("📧 Get in Touch")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Let's Connect!**
        
        I'm always excited to discuss data science, machine learning, and innovative projects. 
        Feel free to reach out for:
        
        - 🤝 Collaboration opportunities
        - 💡 Project discussions
        - 📖 Knowledge sharing
        - 🎯 Career advice
        """)
    
    with col2:
        st.markdown("""
        **Find me on:**
        
        - 📧 Email: [divyanshi12023@gmail.com]
        - 💼 LinkedIn: [https://www.linkedin.com/in/divyanshi-sharma-a71a4825a/]
        - 🐱 GitHub: [github.com/Divyanshi88]
        - 🌐 Portfolio: [https://divyanshi88.github.io/]
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>💡 <strong>Innovation through Data Science</strong> 💡</p>
        <p>Built with ❤️ by Divyanshi Sharma | B.Tech Computer Science (Data Science)</p>
        <p>© 2024 Credit Risk Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🚀 Built by<br><strong>Divyanshi Sharma</strong></p>
    <p>B.Tech Computer Science<br>Data Science Specialist</p>
</div>
""", unsafe_allow_html=True)