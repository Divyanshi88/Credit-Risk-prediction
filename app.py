import streamlit as st
st.set_page_config(
    page_title="Credit Risk Predictor", 
    layout="wide",
    page_icon="💳"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Configure plotly for better rendering
pio.templates.default = "plotly_white"

# Try to import optional libraries with error handling
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    st.warning("joblib not available - using mock model")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP not available - explanations disabled")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')
    plt.style.use('default')
    sns.set_style("whitegrid")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib/Seaborn not available")

# Configure pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# --- Mock Model Class for Demo ---
class MockModel:
    def __init__(self):
        self.feature_names_in_ = ['Employed', 'Bank Balance', 'Annual Salary']
        self.n_features_in_ = 3
        self.classes_ = [0, 1]
        self.feature_importances_ = [0.3, 0.4, 0.3]
        self.n_estimators = 100
    
    def predict_proba(self, X):
        """Mock prediction based on simple rules"""
        if isinstance(X, pd.DataFrame):
            employed = X['Employed'].values
            balance = X['Bank Balance'].values
            salary = X['Annual Salary'].values
        else:
            employed = X[:, 0]
            balance = X[:, 1]
            salary = X[:, 2]
        
        # Simple scoring logic
        score = (employed * 0.3 + 
                np.clip(balance / 50000, 0, 1) * 0.4 + 
                np.clip(salary / 100000, 0, 1) * 0.3)
        
        # Convert to probability
        prob_approve = 1 / (1 + np.exp(-5 * (score - 0.5)))
        prob_deny = 1 - prob_approve
        
        return np.column_stack([prob_deny, prob_approve])

# --- Load Model ---
@st.cache_resource
def load_model():
    if JOBLIB_AVAILABLE:
        try:
            model = joblib.load("models/credit_risk_model_v2.pkl")
            return model
        except FileNotFoundError:
            st.warning("⚠️ Model file not found. Using mock model for demo.")
            return MockModel()
        except Exception as e:
            st.warning(f"⚠️ Error loading model: {str(e)}. Using mock model.")
            return MockModel()
    else:
        st.info("ℹ️ Using mock model for demonstration")
        return MockModel()

# Load the model
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
    
    # Display company info instead of loading image
    st.info("🏢 **Harvestly Credit Solutions** - AI-Powered Risk Assessment")
    
    st.markdown("Predict whether a **loan will be approved** based on applicant financial information. Adjust the threshold to control sensitivity.")
    
    # --- Sidebar Inputs ---
    st.sidebar.header("🔍 Enter Applicant Details")
    
    employment_status = st.sidebar.selectbox("Are you employed?", ["Yes", "No"])
    bank_balance_raw = st.sidebar.text_input("Bank Balance ($)", "5000")
    annual_salary_raw = st.sidebar.text_input("Annual Salary ($)", "45000")
    threshold = st.sidebar.slider("⚖️ Classification Threshold", 0.0, 1.0, 0.5, 0.01)
    
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
    st.sidebar.write(f"**Bank Balance:** ${bank_balance:,.2f}")
    st.sidebar.write(f"**Annual Salary:** ${annual_salary:,.2f}")
    
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
            st.header("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Approval Probability", f"{proba:.1%}")
            
            with col2:
                st.metric("Threshold", f"{threshold:.1%}")
            
            with col3:
                if prediction:
                    st.success(result)
                else:
                    st.error(result)
            
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proba,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Approval Probability"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkgreen" if prediction else "darkred"},
                    'steps': [
                        {'range': [0, threshold], 'color': "lightgray"},
                        {'range': [threshold, 1], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75, 
                        'value': threshold
                    }
                }
            ))
            
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Simple explanation without SHAP
            st.subheader("📈 Decision Factors")
            
            factors = []
            if employed:
                factors.append("✅ Employment status: Positive impact")
            else:
                factors.append("❌ Employment status: Negative impact")
            
            if bank_balance > 10000:
                factors.append("✅ Bank balance: Strong positive impact")
            elif bank_balance > 5000:
                factors.append("🔶 Bank balance: Moderate positive impact")
            else:
                factors.append("❌ Bank balance: Negative impact")
            
            if annual_salary > 60000:
                factors.append("✅ Annual salary: Strong positive impact")
            elif annual_salary > 40000:
                factors.append("🔶 Annual salary: Moderate positive impact")
            else:
                factors.append("❌ Annual salary: Negative impact")
            
            for factor in factors:
                st.write(factor)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    # --- Bulk CSV Prediction ---
    st.markdown("---")
    st.subheader("📁 Upload CSV for Bulk Predictions")
    st.markdown("Your CSV must contain the columns: `Employed`, `Bank Balance`, and `Annual Salary`.")
    
    # Show sample CSV format
    sample_data = pd.DataFrame({
        'Employed': [1, 0, 1],
        'Bank Balance': [10000, 5000, 15000],
        'Annual Salary': [50000, 30000, 70000]
    })
    
    with st.expander("💡 View Sample CSV Format"):
        st.dataframe(sample_data)
    
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
                    "✅ Approved" if p > threshold else "❌ Denied" for p in probs
                ]
                
                st.success("✅ Bulk predictions complete!")
                st.dataframe(df)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Applications", len(df))
                with col2:
                    st.metric("Approved", len(df[df['Probability'] > threshold]))
                with col3:
                    st.metric("Approval Rate", f"{(df['Probability'] > threshold).mean():.1%}")
                
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions as CSV", 
                    csv, 
                    "credit_risk_predictions.csv", 
                    "text/csv"
                )
            else:
                st.error(f"❌ Your CSV is missing required columns. Found: {list(df.columns)}")
                st.info("Required columns: Employed, Bank Balance, Annual Salary")
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
    
    # Feature Importance
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
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Generate Synthetic Data for Analysis
    @st.cache_data
    def generate_synthetic_data():
        np.random.seed(42)
        n_samples = 1000
        
        employed = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
        bank_balance = np.random.lognormal(mean=9, sigma=1, size=n_samples)
        annual_salary = np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples)
        
        # Cap extremes
        bank_balance = np.clip(bank_balance, 0, 50000)
        annual_salary = np.clip(annual_salary, 20000, 150000)
        
        data = pd.DataFrame({
            'Employed': employed,
            'Bank Balance': bank_balance,
            'Annual Salary': annual_salary
        })
        
        probs = model.predict_proba(data)[:, 1]
        data['Probability'] = probs
        
        return data
    
    with st.spinner("Generating synthetic data for analysis..."):
        synthetic_data = generate_synthetic_data()

    # Probability Distribution
    st.header("📈 Probability Distribution Analysis")
    
    fig_dist = px.histogram(
        synthetic_data,
        x='Probability',
        nbins=50,
        title='Distribution of Approval Probabilities',
        labels={'Probability': 'Approval Probability', 'count': 'Frequency'}
    )
    fig_dist.update_layout(height=400)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Probability", f"{synthetic_data['Probability'].mean():.3f}")
    with col2:
        st.metric("Median Probability", f"{synthetic_data['Probability'].median():.3f}")
    with col3:
        st.metric("Std Deviation", f"{synthetic_data['Probability'].std():.3f}")
    with col4:
        st.metric("Approval Rate (>0.5)", f"{(synthetic_data['Probability'] > 0.5).mean():.1%}")

    # Feature Impact Analysis
    st.header("🔍 Feature Impact Analysis")
    feature_to_analyze = st.selectbox("Select feature to analyze:", ['Bank Balance', 'Annual Salary', 'Employed'])

    if feature_to_analyze in ['Bank Balance', 'Annual Salary']:
        fig_scatter = px.scatter(
            synthetic_data,
            x=feature_to_analyze,
            y='Probability',
            color='Employed',
            title=f'{feature_to_analyze} vs Approval Probability',
            labels={'Probability': 'Approval Probability'},
            opacity=0.6,
            color_discrete_map={0: '#ff7f0e', 1: '#2ca02c'}
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        fig_box = px.box(
            synthetic_data,
            x='Employed',
            y='Probability',
            title='Employment Status vs Approval Probability',
            color='Employed',
            color_discrete_map={0: '#ff7f0e', 1: '#2ca02c'}
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)

    # Interactive Prediction Surface
    st.header("🌐 Interactive Prediction Surface")

    if st.checkbox("Show 2D Prediction Surface"):
        employed_val = st.radio("Employment Status:", [0, 1], index=1)

        with st.spinner("Generating prediction surface..."):
            balance_range = np.linspace(0, 30000, 20)
            salary_range = np.linspace(20000, 100000, 20)

            B, S = np.meshgrid(balance_range, salary_range)

            grid_data = pd.DataFrame({
                'Employed': employed_val,
                'Bank Balance': B.ravel(),
                'Annual Salary': S.ravel()
            })

            try:
                grid_probs = model.predict_proba(grid_data)[:, 1].reshape(B.shape)

                fig_surface = go.Figure(data=[go.Surface(
                    z=grid_probs,
                    x=balance_range,
                    y=salary_range,
                    colorscale='RdYlGn',
                    colorbar=dict(title="Approval Probability")
                )])

                fig_surface.update_layout(
                    title=f'Prediction Surface (Employed: {"Yes" if employed_val else "No"})',
                    scene=dict(
                        xaxis_title='Bank Balance ($)',
                        yaxis_title='Annual Salary ($)',
                        zaxis_title='Approval Probability'
                    ),
                    height=600
                )

                st.plotly_chart(fig_surface, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating prediction surface: {e}")

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
        st.info("📷 Developer Profile")
        st.markdown("**Divyanshi Sharma**")
        st.markdown("Data Science Enthusiast")
    
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
        - 📊 Matplotlib, Plotly
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
    - **Model Visualization**: Understand model behavior and performance
    - **Comprehensive Analytics**: Deep insights into prediction patterns
    
    ### 🛠️ Technical Implementation
    
    - **Frontend**: Streamlit for interactive web interface
    - **Backend**: Python with scikit-learn for machine learning
    - **Visualization**: Plotly for interactive charts and graphs
    - **Data Processing**: Pandas and NumPy for data manipulation
    - **Model Deployment**: Streamlit Cloud for easy access
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
        
        - 📧 Email: divyanshi12023@gmail.com
        - 💼 LinkedIn: [Profile Link](https://www.linkedin.com/in/divyanshi-sharma-a71a4825a/)
        - 🐱 GitHub: [github.com/Divyanshi88](https://github.com/Divyanshi88)
        - 🌐 Portfolio: [Portfolio Link](https://divyanshi88.github.io/)
        """)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🚀 Built by<br><strong>Divyanshi Sharma</strong></p>
    <p>B.Tech Computer Science<br>Data Science Specialist</p>
</div>
""", unsafe_allow_html=True)