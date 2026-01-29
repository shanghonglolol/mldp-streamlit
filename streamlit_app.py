import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page config
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")

# Title
st.title("🍷 Wine Quality Predictor")
st.write("Predict if a wine is **Good** (≥6) or **Poor** (<6) based on its chemical properties.")

st.markdown("---")

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("wine_quality_model.pkl")
    scaler = joblib.load("wine_quality_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Error loading model: {e}")

if model_loaded:
    # Input section
    st.header("📊 Enter Wine Properties")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fixed_acidity = st.number_input("Fixed Acidity (g/L)", min_value=4.0, max_value=16.0, value=7.0, step=0.1)
        volatile_acidity = st.number_input("Volatile Acidity (g/L)", min_value=0.1, max_value=1.5, value=0.3, step=0.05)
        citric_acid = st.number_input("Citric Acid (g/L)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        residual_sugar = st.number_input("Residual Sugar (g/L)", min_value=0.5, max_value=20.0, value=2.5, step=0.5)
        chlorides = st.number_input("Chlorides (g/L)", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide (mg/L)", min_value=1.0, max_value=70.0, value=15.0, step=1.0)
    
    with col2:
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide (mg/L)", min_value=5.0, max_value=300.0, value=45.0, step=5.0)
        density = st.number_input("Density (g/cm³)", min_value=0.990, max_value=1.005, value=0.996, step=0.001, format="%.3f")
        pH = st.number_input("pH Level", min_value=2.7, max_value=4.0, value=3.3, step=0.05)
        sulphates = st.number_input("Sulphates (g/L)", min_value=0.2, max_value=1.5, value=0.5, step=0.05)
        alcohol = st.number_input("Alcohol (%)", min_value=8.0, max_value=15.0, value=10.5, step=0.1)
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Wine Quality", use_container_width=True):
        
        # Feature engineering (same as training)
        alcohol_sugar_ratio = alcohol / (residual_sugar + 0.1)
        sulfur_ratio = free_sulfur_dioxide / (total_sulfur_dioxide + 0.1)
        acidity_balance = fixed_acidity / (volatile_acidity + 0.01)
        total_acidity = fixed_acidity + volatile_acidity + citric_acid
        alcohol_sulphates = alcohol * sulphates
        high_alcohol = 1 if alcohol > 11 else 0
        low_volatile_acidity = 1 if volatile_acidity < 0.4 else 0
        bound_sulfur = total_sulfur_dioxide - free_sulfur_dioxide
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'fixed acidity': [fixed_acidity],
            'volatile acidity': [volatile_acidity],
            'citric acid': [citric_acid],
            'residual sugar': [residual_sugar],
            'chlorides': [chlorides],
            'free sulfur dioxide': [free_sulfur_dioxide],
            'total sulfur dioxide': [total_sulfur_dioxide],
            'density': [density],
            'pH': [pH],
            'sulphates': [sulphates],
            'alcohol': [alcohol],
            'alcohol_sugar_ratio': [alcohol_sugar_ratio],
            'sulfur_ratio': [sulfur_ratio],
            'acidity_balance': [acidity_balance],
            'total_acidity': [total_acidity],
            'alcohol_sulphates': [alcohol_sulphates],
            'high_alcohol': [high_alcohol],
            'low_volatile_acidity': [low_volatile_acidity],
            'bound_sulfur': [bound_sulfur]
        })
        
        # Scale and predict
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display result
            st.header("🎯 Prediction Result")
            
            if prediction == 1:
                st.success("## ✅ GOOD WINE")
                st.write("### Quality Score ≥ 6")
                st.balloons()
            else:
                st.error("## ❌ POOR WINE")
                st.write("### Quality Score < 6")
            
            # Confidence
            st.write("---")
            st.subheader("📈 Confidence")
            col_a, col_b = st.columns(2)
            col_a.metric("Poor Wine Probability", f"{probability[0]*100:.1f}%")
            col_b.metric("Good Wine Probability", f"{probability[1]*100:.1f}%")
            
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Footer
st.markdown("---")
st.caption("Wine Quality Predictor | Machine Learning Project | Random Forest Model")
