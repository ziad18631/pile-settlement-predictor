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

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    /* Dark theme for main container */
    .main-container {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 1rem;
    }
    
    /* Section headers */
    .section-header {
        background-color: #2d2d2d;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #ff6b6b;
    }
    
    /* Input sections */
    .input-section {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Model status cards */
    .status-card {
        background-color: #2d5016;
        border: 1px solid #4a7c1a;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    /* Prediction results panel */
    .results-panel {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2d2d2d;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #404040;
        color: #ffffff;
        border-radius: 4px;
        margin-right: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ff6b6b;
        color: #ffffff;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #ff6b6b;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #ff5252;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #404040;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #404040;
        border-radius: 4px;
    }
    
    /* Hide streamlit elements */
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {display: none;}
    .stMainBlockContainer {padding-top: 1rem;}
    
    /* Text colors */
    .stMarkdown, .stText, label {
        color: #ffffff !important;
    }
    
    /* Success message styling */
    .prediction-result {
        background-color: #2d5016;
        border: 1px solid #4a7c1a;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        color: #ffffff;
        margin: 1rem 0;
    }
    
    .prediction-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4ade80;
        margin: 1rem 0;
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

# Main layout with two columns
col1, col2 = st.columns([2, 1])

with col1:
    # Model selection dropdown
    st.markdown("""
    <div style="color: #ffffff; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">
        Choose Prediction Model:
    </div>
    """, unsafe_allow_html=True)
    
    model_choice = st.selectbox(
        "",
        options=["31-Feature Detailed Model", "6-Feature Simplified Model"],
        index=0,
        label_visibility="collapsed"
    )
    
    # Soil Layer Properties or Simplified Input
    if model_choice == "31-Feature Detailed Model":
        st.markdown("""
        <div class="section-header">
            <h3 style="margin: 0;">📊 Soil Layer Properties</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for layers
        tabs = st.tabs([f"Layer {i}" for i in range(1, 10)])
        
        c_values = []
        n30_values = []
        t_values = []
        
        for i, tab in enumerate(tabs, 1):
            with tab:
                col_c, col_n, col_t = st.columns(3)
                
                with col_c:
                    st.write(f"**Cohesion C.{i}:**")
                    c_val = st.number_input(
                        f"Cohesion C.{i}",
                        min_value=0.0,
                        max_value=200.0,
                        value=20.0,
                        step=1.0,
                        key=f"c_{i}",
                        label_visibility="collapsed"
                    )
                    c_values.append(c_val)
                
                with col_n:
                    st.write(f"**N-SPT N30.{i}:**")
                    n30_val = st.number_input(
                        f"N-SPT N30.{i}",
                        min_value=0,
                        max_value=100,
                        value=15,
                        step=1,
                        key=f"n30_{i}",
                        label_visibility="collapsed"
                    )
                    n30_values.append(n30_val)
                
                with col_t:
                    st.write(f"**Thickness T.{i}:**")
                    t_val = st.number_input(
                        f"Thickness T.{i}",
                        min_value=0.0,
                        max_value=20.0,
                        value=2.0,
                        step=0.1,
                        key=f"t_{i}",
                        label_visibility="collapsed"
                    )
                    t_values.append(t_val)
        
        # Additional Properties section
        st.markdown("""
        <div class="section-header">
            <h3 style="margin: 0;">🔧 Additional Properties</h3>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.write("**Water Table Level:**")
                water_table = st.number_input("Water Table Level", min_value=0.0, max_value=50.0, value=5.0, step=0.5, label_visibility="collapsed")
            
            with col_b:
                st.write("**Length of Pile:**")
                pile_length = st.number_input("Length of Pile", min_value=1.0, max_value=100.0, value=20.0, step=0.5, label_visibility="collapsed")
            
            with col_c:
                st.write("**Diameter:**")
                pile_diameter = st.number_input("Diameter", min_value=0.1, max_value=5.0, value=0.6, step=0.05, label_visibility="collapsed")
            
            with col_d:
                st.write("**Load:**")
                pile_load = st.number_input("Load", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0, label_visibility="collapsed")
    
    else:  # 6-Feature Simplified Model
        st.markdown("""
        <div class="section-header">
            <h3 style="margin: 0;">📊 Simplified Input Parameters</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("**NSPT Friction Zone:**")
            nspt_friction = st.number_input("NSPT Friction", min_value=0, max_value=100, value=15, step=1, label_visibility="collapsed")
            
            st.write("**Water Table Level:**")
            water_table_6 = st.number_input("Water Table", min_value=0.0, max_value=50.0, value=5.0, step=0.5, label_visibility="collapsed")
            
            st.write("**Diameter:**")
            pile_diameter_6 = st.number_input("Diameter 6", min_value=0.1, max_value=5.0, value=0.6, step=0.05, label_visibility="collapsed")
        
        with col_b:
            st.write("**NSPT Bearing Zone:**")
            nspt_bearing = st.number_input("NSPT Bearing", min_value=0, max_value=100, value=25, step=1, label_visibility="collapsed")
            
            st.write("**Length of Pile:**")
            pile_length_6 = st.number_input("Length 6", min_value=1.0, max_value=100.0, value=20.0, step=0.5, label_visibility="collapsed")
            
            st.write("**Load:**")
            pile_load_6 = st.number_input("Load 6", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0, label_visibility="collapsed")

with col2:
    # Prediction Results Panel
    st.markdown("""
    <div class="section-header">
        <h3 style="margin: 0;">🎯 Prediction Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Predict button
    if st.button("🔴 Predict Settlement", use_container_width=True, type="primary"):
        if model_choice == "31-Feature Detailed Model":
            if model_31 is not None and scaler_31 is not None:
                try:
                    # Prepare input array for 31-feature model
                    input_data = c_values + n30_values + t_values + [pile_length, pile_diameter, pile_load, water_table]
                    input_array = np.array(input_data).reshape(1, -1)
                    
                    # Scale and predict
                    input_scaled = scaler_31.transform(input_array)
                    prediction = model_31.predict(input_scaled)
                    settlement = float(prediction[0][0])
                    
                    # Display result
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h4>Settlement Prediction</h4>
                        <div class="prediction-value">{settlement:.2f} mm</div>
                        <p>31-Feature Model Result</p>
                        <small>Model Accuracy: R² = 0.9909</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        else:
            if model_6 is not None and scaler_6 is not None:
                try:
                    # Prepare input array for 6-feature model
                    input_data = [nspt_friction, nspt_bearing, pile_length_6, pile_diameter_6, pile_load_6, water_table_6]
                    input_array = np.array(input_data).reshape(1, -1)
                    
                    # Scale and predict
                    input_scaled = scaler_6.transform(input_array)
                    prediction = model_6.predict(input_scaled)
                    settlement = float(prediction[0][0])
                    
                    # Display result
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h4>Settlement Prediction</h4>
                        <div class="prediction-value">{settlement:.2f} mm</div>
                        <p>6-Feature Model Result</p>
                        <small>Model Accuracy: R² = 0.9644</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    # Model Status Section
    st.markdown("""
    <div class="section-header">
        <h3 style="margin: 0;">📊 Model Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Status indicators
    if model_31 is not None:
        st.markdown("""
        <div class="status-card">
            ✅ 31-Feature Model: Ready
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #7f1d1d; border: 1px solid #dc2626; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; color: #ffffff;">
            ❌ 31-Feature Model: Not Available
        </div>
        """, unsafe_allow_html=True)
    
    if model_6 is not None:
        st.markdown("""
        <div class="status-card">
            ✅ 6-Feature Model: Ready
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #7f1d1d; border: 1px solid #dc2626; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; color: #ffffff;">
            ❌ 6-Feature Model: Not Available
        </div>
        """, unsafe_allow_html=True)
