import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================
# 1. VISUAL PAGE ARCHITECTURE
# ==========================================
st.set_page_config(
    page_title="Heart Disease Risk Advisor",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Risk Prediction & Diagnostic Tool")
st.markdown("""
This clinical decision-support dashboard utilizes a trained **XGBoost Classifier** to estimate the probability of heart disease based on patient vitals. 
*Fill out the patient profile in the sidebar to generate a diagnostic report.*
""")

# ==========================================
# 2. CACHED MODEL & FEATURE ALIGNMENT RETRIEVAL
# ==========================================
@st.cache_resource
def load_models():
    model = joblib.load('best_xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Dynamically extract the exact list and sequence of features your scaler expects
    try:
        expected_features = scaler.feature_names_in_.tolist()
    except AttributeError:
        # Fallback if an older scikit-learn version is running
        st.error("Scaler does not contain saved feature name attributes. Please ensure it was fit on a DataFrame with columns.")
        st.stop()
        
    return model, scaler, expected_features

try:
    model, scaler, EXPECTED_FEATURES = load_models()
    st.sidebar.success("Machine Learning Models Loaded! ✅")
except Exception as e:
    st.sidebar.error("Could not find 'best_xgb_model.pkl' or 'scaler.pkl' in this directory.")
    st.stop()

# ==========================================
# 3. CLINICAL INPUT PANEL (SIDEBAR)
# ==========================================
st.sidebar.header("👤 Patient Vitals Profile")

# Continuous numerical variables
age = st.sidebar.slider("Age (years)", min_value=20, max_value=90, value=55)
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=130)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
thalch = st.sidebar.slider("Maximum Heart Rate (bpm)", min_value=60, max_value=220, value=150)
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", min_value=0.0, max_value=6.5, value=1.0, step=0.1)
ca = st.sidebar.selectbox("Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3], index=0)

# Categorical options that require matching one-hot encoding structure
sex = st.sidebar.radio("Biological Sex", options=["Female", "Male"], index=1)
cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=["typical angina", "atypical angina", "non-anginal", "asymptomatic"], index=0)
exang = st.sidebar.radio("Exercise-Induced Angina", options=["False", "True"], index=0)
fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", options=["False", "True"], index=0)

# ==========================================
# 4. ONE-HOT ENCODING & MATRIX REALIGNMENT PIPELINE
# ==========================================

# 4a. Create baseline dictionary matching notebook raw feature expectations
raw_patient_dict = {
    'age': age, 'trestbps': trestbps, 'chol': chol, 'thalch': thalch, 
    'oldpeak': oldpeak, 'ca': ca, 'sex': sex, 'cp': cp, 'exang': exang, 'fbs': fbs
}
patient_df = pd.DataFrame([raw_patient_dict])

# 4b. Perform standard pandas dummy encoding 
patient_encoded = pd.get_dummies(patient_df)

# 4c. Structural Guardrail: Reconstruct missing columns with 0 flags 
# Cleaned syntax: the word "companions" has been safely removed
for col in EXPECTED_FEATURES:
    if col not in patient_encoded.columns:
        patient_encoded[col] = 0

# 4d. Force the exact feature column ordering seen at fit time
patient_final = patient_encoded[EXPECTED_FEATURES]

# 4e. Standardize inputs with the pipeline scaler
patient_scaled = scaler.transform(patient_final)

# Execute predictions
risk_prob = model.predict_proba(patient_scaled)[0][1]
prediction = model.predict(patient_scaled)[0]

# ==========================================
# 5. RISK INTERACTIVE LAYOUT (MAIN PANEL)
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Diagnostic Assessment")
    if prediction == 1:
        st.error("### High Risk Detected 🚨")
    else:
        st.success("### Normal / Low Risk ✅")
        
    st.metric(label="Estimated Heart Disease Probability", value=f"{risk_prob * 100:.2f}%")
    
    st.markdown("#### 优质 Clinical Path Guidelines")
    if risk_prob > 0.75:
        st.write("⚠️ **Critical Action:** Immediate cardiology evaluation and advanced testing recommended.")
    elif risk_prob > 0.40:
        st.write("📈 **Moderate Monitoring:** Track metabolic indicators closely and evaluate lifestyle modifications.")
    else:
        st.write("💚 **Routine Care:** General health preservation metrics are within standard tolerances.")

with col2:
    st.subheader("📊 Feature Risk Attributes")
    
    # Generate local feature contribution matrix mapping based on the true training feature count
    importances = model.feature_importances_
    local_impact = patient_scaled[0] * importances
    
    df_plot = pd.DataFrame({
        'Feature': EXPECTED_FEATURES,
        'Impact': local_impact
    })
    
    # Filter out 0-impact features or take the top 8 most important for dashboard readability
    df_plot['Abs_Impact'] = df_plot['Impact'].abs()
    df_plot = df_plot.sort_values(by='Impact', ascending=True).tail(10)
    
    colors = ['#e63946' if w > 0 else '#2a9d8f' for w in df_plot['Impact']]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df_plot['Feature'], df_plot['Impact'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Relative Impact Direction')
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("📄 Full One-Hot Encoded Row Passed to Model")
st.dataframe(patient_final)