import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Pile Settlement Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with optimized modern color scheme
st.markdown("""
<style>
    .main-c    col1, col2 = st.columns([1, 1])
    with col1:
        predict_clicked = st.button("🚀 PREDICT", type="primary", use_container_width=True, key="predict_6")
    with col2:
        reset_clicked = st.button("🔄 RESET", use_container_width=True, key="reset_6")ner {
        max-width: 900px;
        margin: 0 auto;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
        padding: 1rem;
    }
    
    /* Modern gradient header sections */
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Alternative header for variety */
    .success-header {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Premium info header */
    .info-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 8px 32px rgba(255, 94, 77, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Glass morphism containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Radio button styling with modern look */
    .radio-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Input sections with premium styling */
    .input-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .input-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Enhanced input styling */
    .stNumberInput > div > div > input {
        border: 2px solid #e1e8ed;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.9);
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    /* Modern button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Success button styling */
    .success-button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3) !important;
    }
    
    /* Reset button styling */
    .reset-button {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%) !important;
        box-shadow: 0 4px 15px rgba(252, 70, 107, 0.3) !important;
    }
    
    /* Hide streamlit elements */
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {display: none;}
    .stMainBlockContainer {padding-top: 1rem;}
    
    /* Enhanced radio button styling */
    .stRadio > div {
        flex-direction: row;
        gap: 2rem;
    }
    
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        min-width: 320px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stRadio > div > label:hover {
        border-color: #667eea;
        background: rgba(255, 255, 255, 1);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    /* Enhanced input labels */
    .input-label {
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Small text styling */
    .small-text {
        color: #718096;
        font-size: 0.85rem;
        margin-top: 0.4rem;
        font-style: italic;
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    
    /* Error message styling */
    .stError {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        font-weight: 600;
    }
    
    /* Animation for sections */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .glass-container, .radio-container, .input-section {
        animation: fadeInUp 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Load models and scalers (silently)
@st.cache_resource
def load_models_and_scalers():
    """Load pile settlement prediction models and scalers silently."""
    models = {}
    
    try:
        # Load 31-feature model (try multiple paths)
        model_paths_31 = [
            'final_model_kfold.h5',
            'optimized_bayesian_model_31.h5',
            'enhanced_bayesian_model_31.h5'
        ]
        
        for model_path in model_paths_31:
            if os.path.exists(model_path):
                models['full_model'] = tf.keras.models.load_model(model_path)
                break
        
        # Load 6-feature model (try multiple paths)
        model_paths_6 = [
            'optimized_bayesian_model_6.h5',
            'enhanced_bayesian_model_6.h5'
        ]
        
        for model_path in model_paths_6:
            if os.path.exists(model_path):
                models['simplified_model'] = tf.keras.models.load_model(model_path)
                break
        
        # Load scalers for 31-feature model
        scaler_paths_31 = [
            'scaler_kfold.pkl',
            'optimized_scaler_31.pkl',
            'enhanced_scaler_31.pkl'
        ]
        
        for scaler_path in scaler_paths_31:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    models['full_scaler'] = pickle.load(f)
                break
        
        # Load scalers for 6-feature model
        scaler_paths_6 = [
            'optimized_scaler_6.pkl',
            'enhanced_scaler_6.pkl'
        ]
        
        for scaler_path in scaler_paths_6:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    models['simplified_scaler'] = pickle.load(f)
                break
                
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
    
    # Return in the expected format
    return (
        models.get('full_model'),
        models.get('simplified_model'), 
        models.get('full_scaler'),
        models.get('simplified_scaler')
    )

# Load models
model_31, model_6, scaler_31, scaler_6 = load_models_and_scalers()

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Main header with gradient styling
st.markdown("""
<div class="gradient-header">
    <h2 style="margin: 0; font-size: 1.8rem; text-align: center;">Pile Settlement Prediction Model</h2>
    <div style="margin-top: 0.8rem; font-size: 1.1rem; opacity: 0.95; text-align: center;">Predict pile settlement based on soil properties and pile characteristics</div>
</div>
""", unsafe_allow_html=True)

# Model Selection section with modern styling
st.markdown("""
<div class="success-header">
    <h3 style="margin: 0; font-size: 1.3rem; text-align: center;">⚙️ Model Selection</h3>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_type' not in st.session_state:
    st.session_state.model_type = "simplified"

# Radio button selection in glass container
with st.container():
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    model_choice = st.radio(
        "",
        options=["full", "simplified"],
        format_func=lambda x: "🔧 Full Model (31 Features)" if x == "full" else "⚙️ Simplified Model (6 Features)",
        key="model_radio",
        index=1 if st.session_state.model_type == "simplified" else 0
    )
    
    # Update session state
    st.session_state.model_type = model_choice
    
    # Add descriptions with enhanced styling
    if model_choice == "full":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea20, #764ba220); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <div class="small-text" style="color: #4a5568; font-weight: 500;">
                🔬 Uses detailed soil layer properties: C.1-C.9, N30.1-N30.9, T.1-T.9<br>
                📊 Higher accuracy with comprehensive geotechnical data
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e20, #38ef7d20); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <div class="small-text" style="color: #4a5568; font-weight: 500;">
                ⚡ Uses direct NSPT values for friction and bearing zones<br>
                🚀 Quick predictions with essential parameters only
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Input form based on selected model
if st.session_state.model_type == "simplified":
    # Simplified NSPT Input section with modern styling
    st.markdown("""
    <div class="info-header">
        <h3 style="margin: 0; font-size: 1.3rem; text-align: center;">Simplified NSPT Input</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="input-label">🔨 NS (NSPT for Friction Zone):</div>', unsafe_allow_html=True)
            nspt_friction = st.number_input("", min_value=0, max_value=100, value=15, step=1, key="nspt_friction", label_visibility="collapsed")
            st.markdown('<div class="small-text">💡 Average NSPT value for friction calculations</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="input-label">⚖️ NB (NSPT for Bearing Zone):</div>', unsafe_allow_html=True)
            nspt_bearing = st.number_input("", min_value=0, max_value=100, value=25, step=1, key="nspt_bearing", label_visibility="collapsed")
            st.markdown('<div class="small-text">💡 Average NSPT value for bearing calculations</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Properties section with gradient header
    st.markdown("""
    <div class="success-header">
        <h3 style="margin: 0; font-size: 1.3rem; text-align: center;">🔧 Additional Properties</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="input-label">Water Table Level:</div>', unsafe_allow_html=True)
            water_table = st.number_input("", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key="water_table", label_visibility="collapsed")
            
            st.markdown('<div class="input-label">Diameter:</div>', unsafe_allow_html=True)
            pile_diameter_6 = st.number_input("", min_value=0.1, max_value=5.0, value=0.6, step=0.05, key="pile_diameter_6", label_visibility="collapsed")
        
        with col2:
            st.markdown('<div class="input-label">Length of Pile:</div>', unsafe_allow_html=True)
            pile_length_6 = st.number_input("", min_value=1.0, max_value=100.0, value=20.0, step=0.5, key="pile_length_6", label_visibility="collapsed")
            
            st.markdown('<div class="input-label">Load:</div>', unsafe_allow_html=True)
            pile_load_6 = st.number_input("", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0, key="pile_load_6", label_visibility="collapsed")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Buttons section
    st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        predict_clicked = st.button("� Predict", type="primary", use_container_width=True, key="predict_6")
    with col2:
        reset_clicked = st.button("🔄 Reset", use_container_width=True, key="reset_6")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Handle prediction
    if predict_clicked:
        if model_6 is not None and scaler_6 is not None:
            try:
                # Prepare input array for 6-feature model (simplified)
                input_data = [nspt_friction, nspt_bearing, pile_length_6, pile_diameter_6, pile_load_6, water_table]
                input_array = np.array(input_data).reshape(1, -1)
                
                # Scale and predict
                input_scaled = scaler_6.transform(input_array)
                prediction = model_6.predict(input_scaled)
                settlement = float(prediction[0][0])
                
                # Enhanced result display
                st.markdown("""
                <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                           color: white; padding: 2rem; border-radius: 16px; text-align: center; 
                           margin: 2rem 0; box-shadow: 0 8px 32px rgba(17, 153, 142, 0.4);">
                    <h3 style="margin: 0; font-size: 1.5rem;">🎯 Prediction Result</h3>
                    <div style="font-size: 3rem; font-weight: bold; margin: 1rem 0;">{:.2f} mm</div>
                    <p style="font-size: 1.1rem; margin: 0; opacity: 0.9;">Predicted Pile Settlement (Simplified Model)</p>
                    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
                        🔬 Model Accuracy: R² = 0.9644 (96.44%)
                    </div>
                </div>
                """.format(settlement), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

else:  # Full model
    st.markdown("""
    <div class="blue-header">
        <h3 style="margin: 0; font-size: 1.2rem;">Full Model Input (31 Features)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Soil Properties Section
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("**Soil Properties (Layer-wise Data)**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Cohesion Values (kPa)**")
            c_values = []
            for i in range(1, 10):
                c_values.append(st.number_input(f"C.{i}", min_value=0.0, max_value=200.0, value=20.0, step=1.0, key=f"c_{i}"))
        
        with col2:
            st.markdown("**N30 Values**")
            n30_values = []
            for i in range(1, 10):
                n30_values.append(st.number_input(f"N30.{i}", min_value=0, max_value=100, value=15, step=1, key=f"n30_{i}"))
        
        with col3:
            st.markdown("**Thickness Values (m)**")
            t_values = []
            for i in range(1, 10):
                t_values.append(st.number_input(f"T.{i}", min_value=0.0, max_value=20.0, value=2.0, step=0.1, key=f"t_{i}"))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Pile Properties Section
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("**Pile Properties**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pile_length = st.number_input("Pile Length (m)", min_value=1.0, max_value=100.0, value=20.0, step=0.5)
        
        with col2:
            pile_diameter = st.number_input("Pile Diameter (m)", min_value=0.1, max_value=5.0, value=0.6, step=0.05)
        
        with col3:
            pile_load = st.number_input("Pile Load (kN)", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0)
        
        with col4:
            elastic_modulus = st.number_input("Elastic Modulus (GPa)", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        predict_full = st.button("� Predict", type="primary", use_container_width=True, key="predict_31")
    with col2:
        reset_full = st.button("🔄 Reset", use_container_width=True, key="reset_31")

    # Handle prediction
    if predict_full:
        if model_31 is not None and scaler_31 is not None:
            try:
                # Prepare input array
                input_data = c_values + n30_values + t_values + [pile_length, pile_diameter, pile_load, elastic_modulus]
                input_array = np.array(input_data).reshape(1, -1)
                
                # Scale and predict
                input_scaled = scaler_31.transform(input_array)
                prediction = model_31.predict(input_scaled)
                settlement = float(prediction[0][0])
                
                # Display result
                st.success(f"**Predicted Settlement:** {settlement:.2f} mm")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

# Close main container
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • Machine Learning Model for Pile Settlement Prediction*")
