import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #722F37;
        font-size: 2.5rem;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .good-wine {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 3px solid #28a745;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .poor-wine {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 3px solid #dc3545;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .tips-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #000000;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #0066cc;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">🍷 Wine Quality Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict if a wine is Good (≥6) or Poor (&lt;6) based on chemical properties</p>', unsafe_allow_html=True)

# Load model
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
    st.error(f"⚠️ Model files not found. Please ensure wine_quality_model.pkl and wine_quality_scaler.pkl are uploaded.")

if model_loaded:
    
    # Sidebar with tips
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=400", use_container_width=True)
        
        st.markdown("### 🏆 About This App")
        st.markdown("""
        **Model:** Random Forest Classifier
        
        **Dataset:** UCI Wine Quality (6,497 samples)
        
        **Key Techniques:**
        - SMOTE (class balancing)
        - Feature Engineering (8 new features)
        - Hyperparameter Tuning
        
        **Performance:**
        - F1-Score: 0.87
        - Accuracy: 83%
        """)
        
        st.markdown("---")
        
        st.markdown("### ✅ Ideal Values (Good Wine)")
        st.markdown("""
        | Property | Ideal Range |
        |----------|-------------|
        | Alcohol | **11 – 14%** |
        | Volatile Acidity | **< 0.4 g/L** |
        | Sulphates | **0.5 – 1.0 g/L** |
        | Citric Acid | **0.3 – 0.6 g/L** |
        | Chlorides | **< 0.06 g/L** |
        | Residual Sugar | **1.5 – 4.0 g/L** |
        | Fixed Acidity | **6 – 9 g/L** |
        | pH | **3.1 – 3.4** |
        | Free SO₂ | **15 – 40 mg/L** |
        | Total SO₂ | **30 – 120 mg/L** |
        | Density | **0.992 – 0.997** |
        """)
        
        st.markdown("---")
        
        st.markdown("### ❌ Warning Signs (Poor Wine)")
        st.markdown("""
        | Property | Poor Range |
        |----------|------------|
        | Alcohol | **< 10%** |
        | Volatile Acidity | **> 0.6 g/L** |
        | Sulphates | **< 0.4 g/L** |
        | Chlorides | **> 0.1 g/L** |
        | Total SO₂ | **> 150 mg/L** |
        """)
        
        st.markdown("---")
        
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Adjust the sliders below
        2. Click **Predict Quality**
        3. See your result!
        """)
    
    # Main content
    st.markdown("---")
    st.header("📊 Enter Wine Chemical Properties")
    
    # Preset buttons
    st.markdown("#### 🔄 Quick Presets")
    preset_col1, preset_col2, preset_col3 = st.columns([1, 1, 2])
    
    with preset_col1:
        if st.button("✅ Good Wine Example", use_container_width=True):
            st.session_state['fixed_acidity'] = 7.0
            st.session_state['volatile_acidity'] = 0.25
            st.session_state['citric_acid'] = 0.35
            st.session_state['pH'] = 3.20
            st.session_state['residual_sugar'] = 2.5
            st.session_state['chlorides'] = 0.04
            st.session_state['density'] = 0.995
            st.session_state['alcohol'] = 12.5
            st.session_state['free_sulfur_dioxide'] = 30.0
            st.session_state['total_sulfur_dioxide'] = 80.0
            st.session_state['sulphates'] = 0.65
            st.rerun()
    
    with preset_col2:
        if st.button("❌ Poor Wine Example", use_container_width=True):
            st.session_state['fixed_acidity'] = 7.0
            st.session_state['volatile_acidity'] = 1.10
            st.session_state['citric_acid'] = 0.0
            st.session_state['pH'] = 3.50
            st.session_state['residual_sugar'] = 8.0
            st.session_state['chlorides'] = 0.15
            st.session_state['density'] = 1.003
            st.session_state['alcohol'] = 9.0
            st.session_state['free_sulfur_dioxide'] = 5.0
            st.session_state['total_sulfur_dioxide'] = 200.0
            st.session_state['sulphates'] = 0.30
            st.rerun()
    
    st.markdown("---")
    
    # Three columns for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🧪 Acidity & pH")
        
        fixed_acidity = st.slider(
            "Fixed Acidity (g/L)",
            min_value=4.0,
            max_value=16.0,
            value=st.session_state.get('fixed_acidity', 7.0),
            step=0.1,
            help="Primary acids in wine. Typical: 6-9 g/L",
            key='fixed_acidity'
        )
        
        volatile_acidity = st.slider(
            "Volatile Acidity (g/L)",
            min_value=0.1,
            max_value=1.5,
            value=st.session_state.get('volatile_acidity', 0.3),
            step=0.05,
            help="⚠️ Keep below 0.4 for good wine! High = vinegar taste",
            key='volatile_acidity'
        )
        
        citric_acid = st.slider(
            "Citric Acid (g/L)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('citric_acid', 0.3),
            step=0.05,
            help="Adds freshness. Typical: 0.2-0.5 g/L",
            key='citric_acid'
        )
        
        pH = st.slider(
            "pH Level",
            min_value=2.7,
            max_value=4.0,
            value=st.session_state.get('pH', 3.3),
            step=0.05,
            help="Wine acidity. Typical: 3.0-3.5",
            key='pH'
        )

    with col2:
        st.subheader("🍬 Sugar & Density")
        
        residual_sugar = st.slider(
            "Residual Sugar (g/L)",
            min_value=0.5,
            max_value=20.0,
            value=st.session_state.get('residual_sugar', 2.5),
            step=0.5,
            help="Sugar left after fermentation. Dry wine: <4 g/L",
            key='residual_sugar'
        )
        
        chlorides = st.slider(
            "Chlorides (g/L)",
            min_value=0.01,
            max_value=0.2,
            value=st.session_state.get('chlorides', 0.05),
            step=0.01,
            help="Salt content. Keep low (<0.1) for better quality",
            key='chlorides'
        )
        
        density = st.slider(
            "Density (g/cm³)",
            min_value=0.990,
            max_value=1.005,
            value=st.session_state.get('density', 0.996),
            step=0.001,
            help="Wine density. Lower = more alcohol",
            key='density'
        )
        
        alcohol = st.slider(
            "Alcohol (%)",
            min_value=8.0,
            max_value=15.0,
            value=st.session_state.get('alcohol', 10.5),
            step=0.1,
            help="⭐ Higher alcohol (>11%) often = better quality!",
            key='alcohol'
        )

    with col3:
        st.subheader("🧂 Sulfur & Sulphates")
        
        free_sulfur_dioxide = st.slider(
            "Free Sulfur Dioxide (mg/L)",
            min_value=1.0,
            max_value=70.0,
            value=st.session_state.get('free_sulfur_dioxide', 15.0),
            step=1.0,
            help="Prevents bacteria. Typical: 10-40 mg/L",
            key='free_sulfur_dioxide'
        )
        
        total_sulfur_dioxide = st.slider(
            "Total Sulfur Dioxide (mg/L)",
            min_value=5.0,
            max_value=300.0,
            value=st.session_state.get('total_sulfur_dioxide', 45.0),
            step=5.0,
            help="Total SO2. Keep under 150 mg/L",
            key='total_sulfur_dioxide'
        )
        
        sulphates = st.slider(
            "Sulphates (g/L)",
            min_value=0.2,
            max_value=1.5,
            value=st.session_state.get('sulphates', 0.5),
            step=0.05,
            help="Wine preservative. Higher = better preservation",
            key='sulphates'
        )

    st.markdown("---")
    
    # Quick feedback before prediction
    st.markdown("### 💡 Quick Analysis of Your Input")
    
    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
    
    with feedback_col1:
        if alcohol > 11:
            st.success("✅ Alcohol > 11% (Good!)")
        else:
            st.warning("⚠️ Alcohol < 11% (Could be better)")
    
    with feedback_col2:
        if volatile_acidity < 0.4:
            st.success("✅ Low volatile acidity (Good!)")
        else:
            st.error("❌ High volatile acidity (Bad - vinegar taste)")
    
    with feedback_col3:
        if sulphates >= 0.5:
            st.success("✅ Good sulphate level")
        else:
            st.warning("⚠️ Low sulphates")

    st.markdown("---")
    
    # Predict button
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_button = st.button("🔮 Predict Wine Quality", use_container_width=True, type="primary")
    
    if predict_button:
        # Feature engineering
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
        
        try:
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            
            # Result
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            
            with result_col2:
                if prediction == 1:
                    st.markdown("""
                    <div class="good-wine">
                        <h1 style="color: #155724; margin: 0;">✅ GOOD WINE</h1>
                        <h3 style="color: #155724;">Quality Score ≥ 6</h3>
                        <p style="font-size: 1.2rem; color: #000000;">Confidence: {:.1f}%</p>
                    </div>
                    """.format(probability[1]*100), unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown("""
                    <div class="poor-wine">
                        <h1 style="color: #721c24; margin: 0;">❌ POOR WINE</h1>
                        <h3 style="color: #721c24;">Quality Score &lt; 6</h3>
                        <p style="font-size: 1.2rem; color: #000000;">Confidence: {:.1f}%</p>
                    </div>
                    """.format(probability[0]*100), unsafe_allow_html=True)
            
            # Probability meters
            st.markdown("### 📊 Prediction Confidence")
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.metric("Poor Wine Probability", f"{probability[0]*100:.1f}%")
                st.progress(probability[0])
            
            with prob_col2:
                st.metric("Good Wine Probability", f"{probability[1]*100:.1f}%")
                st.progress(probability[1])
            
            # Improvement suggestions if poor wine
            if prediction == 0:
                st.markdown("### 🔧 How to Improve This Wine")
                st.markdown("""
                <div class="tips-box">
                <b>Suggestions to improve quality:</b>
                <ul>
                    <li>🍷 <b>Increase alcohol content</b> - Longer fermentation can help</li>
                    <li>🧪 <b>Reduce volatile acidity</b> - Better temperature control during fermentation</li>
                    <li>🧂 <b>Adjust sulphates</b> - Add wine preservatives carefully</li>
                    <li>🍬 <b>Balance residual sugar</b> - Control fermentation timing</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("### 🎉 Great Wine Characteristics")
                st.markdown("""
                <div class="info-box">
                <b>This wine has good characteristics:</b>
                <ul>
                    <li>✅ Well-balanced chemical properties</li>
                    <li>✅ Good alcohol-to-sugar ratio</li>
                    <li>✅ Appropriate acidity levels</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.info("Please check that all inputs are valid numbers.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 10px;">
    🍷 Wine Quality Predictor | Machine Learning Project<br>
    Model: Random Forest | F1-Score: 0.87 | Accuracy: 83%
</div>
""", unsafe_allow_html=True)