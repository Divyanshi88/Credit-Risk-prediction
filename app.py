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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Configure plotly for better rendering
pio.templates.default = "plotly_white"
# For deployment, use 'iframe' or 'streamlit' instead of 'browser'
try:
    pio.renderers.default = "streamlit"
except:
    pio.renderers.default = "iframe"

# Set matplotlib and seaborn style
plt.style.use('default')
sns.set_style("whitegrid")

# Configure pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/credit_risk_model_v2.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'credit_risk_model_v2.pkl' exists in the 'models' directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load the model with error handling
try:
    model = load_model()
    if model is None:
        st.error("❌ Failed to load the credit risk model.")
        st.stop()
except Exception as e:
    st.error(f"❌ Critical error loading model: {str(e)}")
    st.stop()

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
    
    # Load image with error handling
    try:
        st.image("./images/Harvestly.png", width=250)
    except FileNotFoundError:
        st.info("ℹ️ Company logo not found. This doesn't affect the functionality.")
    except Exception as e:
        st.info(f"ℹ️ Could not load logo: {str(e)}")
    
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
                
                # Clear any existing plots
                plt.clf()
                plt.figure(figsize=(10, 6))
                
                st.set_option('deprecation.showPyplotGlobalUse', False)
                shap.plots.waterfall(shap_values[0], show=False)
                
                # Ensure the plot is displayed
                fig = plt.gcf()
                st.pyplot(fig, bbox_inches="tight", clear_figure=True)
                plt.close(fig)
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
# MODEL VISUALIZATION PAGE - FIXED VERSION
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
    
    # 🎯 Feature Importance
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
        fig_importance.update_layout(showlegend=False, height=500, margin=dict(l=50, r=50, t=50, b=50))
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # 🔄 Enhanced Synthetic Data
    @st.cache_data
    def generate_synthetic_data():
        np.random.seed(42)
        n_samples = 1000
        
        employed = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
        bank_balance = np.random.lognormal(mean=10, sigma=0.9, size=n_samples)
        annual_salary = np.random.lognormal(mean=11.2, sigma=0.7, size=n_samples)
        
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
    
    synthetic_data = generate_synthetic_data()

    # 📈 Probability Distribution
    st.header("📈 Probability Distribution Analysis")

    if len(synthetic_data) > 0 and 'Probability' in synthetic_data.columns:
        fig_dist = px.histogram(
            synthetic_data,
            x='Probability',
            nbins=50,
            title='Distribution of Approval Probabilities',
            labels={'Probability': 'Approval Probability', 'count': 'Frequency'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_dist.update_layout(xaxis_range=[0, 1], height=500, showlegend=False, margin=dict(l=50, r=50, t=50, b=50))
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
            st.metric("Approval Rate (>0.5)", f"{(synthetic_data['Probability'] > 0.5).mean():.3f}")
    else:
        st.warning("No valid probability data generated.")

    # 🔍 Feature Impact
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
        fig_scatter.update_layout(height=500, showlegend=True, margin=dict(l=50, r=50, t=50, b=50))
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
        fig_box.update_layout(
            height=500, showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis_title='Employment Status (0=Unemployed, 1=Employed)'
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ⚖️ Threshold Analysis
    st.header("⚖️ Threshold Analysis")

    @st.cache_data
    def compute_threshold_metrics(probabilities):
        thresholds = np.linspace(0.1, 0.9, 50)
        metrics = []

        # More lenient true label generation
        true_labels = (probabilities > 0.3).astype(int)

        for thresh in thresholds:
            predictions = (probabilities > thresh).astype(int)

            tp = np.sum((predictions == 1) & (true_labels == 1))
            fp = np.sum((predictions == 1) & (true_labels == 0))
            tn = np.sum((predictions == 0) & (true_labels == 0))
            fn = np.sum((predictions == 0) & (true_labels == 1))

            accuracy = (tp + tn) / len(probabilities) if len(probabilities) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            metrics.append({
                'Threshold': thresh,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity
            })

        return pd.DataFrame(metrics)

    metrics_df = compute_threshold_metrics(synthetic_data['Probability'].values)

    if len(metrics_df) > 0:
        fig_metrics = px.line(
            metrics_df,
            x='Threshold',
            y=['Accuracy', 'Precision', 'Recall', 'Specificity'],
            title='Model Performance vs Threshold',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        fig_metrics.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1], height=500, showlegend=True, margin=dict(l=50, r=50, t=50, b=50))
        st.plotly_chart(fig_metrics, use_container_width=True)

        # F1-Score-based Optimal Threshold
        f1 = 2 * metrics_df['Precision'] * metrics_df['Recall'] / (metrics_df['Precision'] + metrics_df['Recall'] + 1e-10)
        best_idx = np.argmax(f1)
        optimal_threshold = metrics_df.iloc[best_idx]['Threshold']
        st.success(f"🎯 **Optimal Threshold (F1-Score):** {optimal_threshold:.3f}")
        
        # ROC Curve Analysis
        st.header("📈 ROC Curve Analysis")
        
        # Calculate ROC curve
        true_labels = (synthetic_data['Probability'] > 0.3).astype(int)
        probabilities = synthetic_data['Probability'].values
        
        # Calculate TPR and FPR for different thresholds
        thresholds_roc = np.linspace(0, 1, 100)
        tpr_list = []
        fpr_list = []
        
        for thresh in thresholds_roc:
            predictions = (probabilities > thresh).astype(int)
            
            tp = np.sum((predictions == 1) & (true_labels == 1))
            fp = np.sum((predictions == 1) & (true_labels == 0))
            tn = np.sum((predictions == 0) & (true_labels == 0))
            fn = np.sum((predictions == 0) & (true_labels == 1))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Calculate AUC
        auc_score = np.trapz(tpr_list, fpr_list)
        
        # Create ROC curve plot
        fig_roc = go.Figure()
        
        # Add ROC curve
        fig_roc.add_trace(go.Scatter(
            x=fpr_list,
            y=tpr_list,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Add diagonal line (random classifier)
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash', width=1)
        ))
        
        fig_roc.update_layout(
            title='ROC Curve - Model Performance',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Display AUC score
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AUC Score", f"{auc_score:.3f}")
        with col2:
            st.metric("Model Quality", "Excellent" if auc_score > 0.8 else "Good" if auc_score > 0.7 else "Fair")
        with col3:
            st.metric("Optimal Threshold", f"{optimal_threshold:.3f}")
        
        # Confusion Matrix
        st.header("🔍 Confusion Matrix")
        
        # Create confusion matrix at optimal threshold
        predictions_optimal = (probabilities > optimal_threshold).astype(int)
        
        # Calculate confusion matrix values
        tp = np.sum((predictions_optimal == 1) & (true_labels == 1))
        fp = np.sum((predictions_optimal == 1) & (true_labels == 0))
        tn = np.sum((predictions_optimal == 0) & (true_labels == 0))
        fn = np.sum((predictions_optimal == 0) & (true_labels == 1))
        
        # Create confusion matrix as a heatmap
        confusion_matrix_data = np.array([[tn, fp], [fn, tp]])
        
        fig_cm = px.imshow(
            confusion_matrix_data,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Denied', 'Approved'],
            y=['Denied', 'Approved'],
            color_continuous_scale='Blues',
            text_auto=True,
            title=f'Confusion Matrix (Threshold: {optimal_threshold:.3f})'
        )
        
        fig_cm.update_layout(height=400, margin=dict(l=50, r=50, t=50, b=50))
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Display classification metrics
        col1, col2, col3, col4 = st.columns(4)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        with col1:
            st.metric("Precision", f"{precision:.3f}")
        with col2:
            st.metric("Recall", f"{recall:.3f}")
        with col3:
            st.metric("F1-Score", f"{f1_score:.3f}")
        with col4:
            st.metric("Accuracy", f"{accuracy:.3f}")
    
    else:
        st.warning("Threshold analysis could not be completed.")

    # 🌐 Prediction Surface
    st.header("🌐 Interactive Prediction Surface")

    if st.checkbox("Show 2D Prediction Surface"):
        employed_val = st.radio("Employment Status:", [0, 1], index=1)

        balance_range = np.linspace(0, 30000, 30)
        salary_range = np.linspace(20000, 120000, 30)

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
                colorscale='viridis',
                name='Approval Probability',
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
        # Load profile image with error handling
        try:
            st.image("./images/Divyanshi.png", 
                     width=200, caption="Divyanshi Sharma")
        except FileNotFoundError:
            st.info("📷 Profile picture not found")
        except Exception as e:
            st.info(f"📷 Could not load profile picture: {str(e)}")
    
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