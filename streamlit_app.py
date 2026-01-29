import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #722F37;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-good {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .prediction-poor {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .feature-info {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #722F37;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #5a252c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load("wine_quality_model.pkl")
        scaler = joblib.load("wine_quality_scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'wine_quality_model.pkl' and 'wine_quality_scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_model()

# Header
st.markdown('<h1 class="main-header">🍷 Wine Quality Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict if a wine is Good (≥6) or Poor (<6) quality based on chemical properties</p>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=400", use_container_width=True)
    st.markdown("### About This App")
    st.markdown("""
    This app uses a **Random Forest** machine learning model to predict wine quality based on chemical properties.
    
    **Model Performance:**
    - F1-Score: ~0.87
    - Accuracy: ~83%
    
    **Features Used:**
    - 11 original chemical properties
    - 8 engineered features
    
    **Target Audience:**
    - Wineries
    - Wine distributors
    - Quality control teams
    """)
    
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Adjust the sliders to input wine properties
    2. Click **Predict Quality**
    3. View the prediction result
    """)

# Main content - Input features
st.markdown("### 📊 Enter Wine Chemical Properties")

# Create three columns for inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Acidity & pH")
    fixed_acidity = st.slider(
        "Fixed Acidity (g/L)",
        min_value=3.0, max_value=16.0, value=7.0, step=0.1,
        help="Primary acids in wine (tartaric acid)"
    )
    volatile_acidity = st.slider(
        "Volatile Acidity (g/L)",
        min_value=0.1, max_value=1.6, value=0.3, step=0.05,
        help="Amount of acetic acid (vinegar taste). Lower is better."
    )
    citric_acid = st.slider(
        "Citric Acid (g/L)",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="Adds freshness and flavor"
    )
    pH = st.slider(
        "pH Level",
        min_value=2.7, max_value=4.0, value=3.3, step=0.05,
        help="Acidity level (3-4 for most wines)"
    )

with col2:
    st.markdown("#### Sugar & Density")
    residual_sugar = st.slider(
        "Residual Sugar (g/L)",
        min_value=0.5, max_value=20.0, value=2.5, step=0.5,
        help="Sugar remaining after fermentation"
    )
    chlorides = st.slider(
        "Chlorides (g/L)",
        min_value=0.01, max_value=0.2, value=0.05, step=0.01,
        help="Salt content in wine"
    )
    density = st.slider(
        "Density (g/cm³)",
        min_value=0.990, max_value=1.005, value=0.996, step=0.001,
        help="Density of wine"
    )
    alcohol = st.slider(
        "Alcohol (%)",
        min_value=8.0, max_value=15.0, value=10.5, step=0.1,
        help="Alcohol content. Higher alcohol often indicates better quality."
    )

with col3:
    st.markdown("#### Sulfur & Sulphates")
    free_sulfur_dioxide = st.slider(
        "Free Sulfur Dioxide (mg/L)",
        min_value=1.0, max_value=70.0, value=15.0, step=1.0,
        help="Free SO2 prevents microbial growth"
    )
    total_sulfur_dioxide = st.slider(
        "Total Sulfur Dioxide (mg/L)",
        min_value=5.0, max_value=300.0, value=45.0, step=5.0,
        help="Total amount of SO2"
    )
    sulphates = st.slider(
        "Sulphates (g/L)",
        min_value=0.2, max_value=1.5, value=0.5, step=0.05,
        help="Wine additive that contributes to SO2 levels"
    )

st.markdown("---")

# Predict button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔮 Predict Wine Quality", use_container_width=True)

# Prediction
if predict_button:
    if model is not None and scaler is not None:
        # Create feature engineering (same as training)
        alcohol_sugar_ratio = alcohol / (residual_sugar + 0.1)
        sulfur_ratio = free_sulfur_dioxide / (total_sulfur_dioxide + 0.1)
        acidity_balance = fixed_acidity / (volatile_acidity + 0.01)
        total_acidity = fixed_acidity + volatile_acidity + citric_acid
        alcohol_sulphates = alcohol * sulphates
        high_alcohol = 1 if alcohol > 11 else 0
        low_volatile_acidity = 1 if volatile_acidity < 0.4 else 0
        bound_sulfur = total_sulfur_dioxide - free_sulfur_dioxide
        
        # Create input dataframe with all features (original + engineered)
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
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Get prediction and probability
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        
        col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
        
        with col_result2:
            if prediction == 1:
                st.markdown("""
                <div class="prediction-good">
                    <h1 style="color: #28a745; margin: 0;">✅ GOOD WINE</h1>
                    <h3 style="color: #155724; margin: 10px 0;">Quality Score ≥ 6</h3>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown("""
                <div class="prediction-poor">
                    <h1 style="color: #dc3545; margin: 0;">❌ POOR WINE</h1>
                    <h3 style="color: #721c24; margin: 10px 0;">Quality Score < 6</h3>
                </div>
                """, unsafe_allow_html=True)
        
        # Confidence scores
        st.markdown("### 📈 Confidence Scores")
        col_conf1, col_conf2 = st.columns(2)
        
        with col_conf1:
            st.metric(
                label="Probability: Poor Wine",
                value=f"{probability[0]*100:.1f}%"
            )
        
        with col_conf2:
            st.metric(
                label="Probability: Good Wine",
                value=f"{probability[1]*100:.1f}%"
            )
        
        # Progress bar for confidence
        st.progress(probability[1])
        
        # Key insights
        st.markdown("### 💡 Key Insights from Your Input")
        
        insights = []
        if alcohol > 11:
            insights.append("✅ **High alcohol content** (>11%) - This is typically associated with good quality wines.")
        else:
            insights.append("⚠️ **Low alcohol content** (<11%) - Higher alcohol often indicates better quality.")
        
        if volatile_acidity < 0.4:
            insights.append("✅ **Low volatile acidity** (<0.4) - Less vinegar taste, which is good.")
        else:
            insights.append("⚠️ **High volatile acidity** (>0.4) - May have vinegar taste, reducing quality.")
        
        if sulphates > 0.5:
            insights.append("✅ **Good sulphate levels** - Helps with wine preservation.")
        else:
            insights.append("⚠️ **Low sulphate levels** - May affect preservation.")
        
        for insight in insights:
            st.markdown(insight)
    else:
        st.error("❌ Model not loaded. Please check if model files exist.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🍷 Wine Quality Predictor | Built with Streamlit | Machine Learning Project</p>
    <p>Model: Random Forest Classifier | F1-Score: 0.87</p>
</div>
""", unsafe_allow_html=True)
