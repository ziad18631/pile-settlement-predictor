import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from sklearn.preprocessing import StandardScaler

# Configure Streamlit page
st.set_page_config(
    page_title="Pile Settlement Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables for models
@st.cache_resource
def load_models():
    """Load pile settlement prediction models and scalers."""
    models = {}
    
    try:
        # Load 31-feature model
        model_paths_31 = [
            'final_model_kfold.h5',
            'optimized_bayesian_model_31.h5',
            'enhanced_bayesian_model_31.h5'
        ]
        
        for model_path in model_paths_31:
            if os.path.exists(model_path):
                models['full_model'] = tf.keras.models.load_model(model_path)
                st.success(f"✅ 31-feature model loaded: {model_path}")
                break
        
        # Load 6-feature model
        model_paths_6 = [
            'optimized_bayesian_model_6.h5',
            'enhanced_bayesian_model_6.h5'
        ]
        
        for model_path in model_paths_6:
            if os.path.exists(model_path):
                models['simplified_model'] = tf.keras.models.load_model(model_path)
                st.success(f"✅ 6-feature model loaded: {model_path}")
                break
        
        # Load scalers
        scaler_paths_31 = [
            'scaler_kfold.pkl',
            'optimized_scaler_31.pkl',
            'enhanced_scaler_31.pkl'
        ]
        
        for scaler_path in scaler_paths_31:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    models['full_scaler'] = pickle.load(f)
                st.success(f"✅ 31-feature scaler loaded: {scaler_path}")
                break
        
        scaler_paths_6 = [
            'optimized_scaler_6.pkl',
            'enhanced_scaler_6.pkl'
        ]
        
        for scaler_path in scaler_paths_6:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    models['simplified_scaler'] = pickle.load(f)
                st.success(f"✅ 6-feature scaler loaded: {scaler_path}")
                break
                
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
    
    return models

def main():
    # Load models
    models = load_models()
    
    # Header
    st.title("🏗️ Pile Settlement Prediction Model")
    st.markdown("**Predict pile settlement based on soil properties and pile characteristics**")
    
    # Model performance display
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("31-Feature Model", "R² = 0.9909", "99.09% accuracy")
    
    with col2:
        st.metric("6-Feature Model", "R² = 0.9644", "96.44% accuracy")
    
    st.markdown("---")
    
    # Model selection
    model_type = st.selectbox(
        "**Choose Prediction Model:**",
        ["31-Feature Detailed Model", "6-Feature Simplified Model"],
        help="Select the model type based on available data complexity"
    )
    
    # Create two columns for input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if model_type == "31-Feature Detailed Model":
            st.subheader("🗂️ Soil Layer Properties")
            
            # Create tabs for better organization
            tabs = st.tabs([f"Layer {i+1}" for i in range(9)])
            
            layer_data = {}
            for i, tab in enumerate(tabs):
                with tab:
                    layer_num = i + 1
                    col_c, col_n, col_t = st.columns(3)
                    
                    with col_c:
                        layer_data[f'C.{layer_num}'] = st.number_input(
                            f"Cohesion C.{layer_num}:",
                            min_value=0.0,
                            step=0.1,
                            key=f"c_{layer_num}"
                        )
                    
                    with col_n:
                        layer_data[f'N30.{layer_num}'] = st.number_input(
                            f"N-SPT N30.{layer_num}:",
                            min_value=0.0,
                            step=1.0,
                            key=f"n_{layer_num}"
                        )
                    
                    with col_t:
                        layer_data[f'T.{layer_num}'] = st.number_input(
                            f"Thickness T.{layer_num}:",
                            min_value=0.0,
                            step=0.1,
                            key=f"t_{layer_num}"
                        )
            
            st.subheader("🏗️ Additional Properties")
            col_w, col_l, col_d, col_load = st.columns(4)
            
            with col_w:
                wt_level = st.number_input("Water Table Level:", min_value=0.0, step=0.1)
            with col_l:
                pile_length = st.number_input("Length of Pile:", min_value=0.0, step=0.1)
            with col_d:
                diameter = st.number_input("Diameter:", min_value=0.0, step=0.01)
            with col_load:
                load = st.number_input("Load:", min_value=0.0, step=1.0)
            
            # Prepare features for 31-feature model
            features = []
            for i in range(1, 10):
                features.append(layer_data.get(f'C.{i}', 0))
                features.append(layer_data.get(f'N30.{i}', 0))
                features.append(layer_data.get(f'T.{i}', 0))
            features.extend([wt_level, pile_length, diameter, load])
            
        else:  # 6-Feature Simplified Model
            st.subheader("📊 Simplified NSPT Input")
            
            col_ns, col_nb = st.columns(2)
            with col_ns:
                NS = st.number_input(
                    "NS (NSPT for Friction Zone):",
                    min_value=0.0,
                    step=1.0,
                    help="Average NSPT value for friction calculations"
                )
            with col_nb:
                NB = st.number_input(
                    "NB (NSPT for Bearing Zone):",
                    min_value=0.0,
                    step=1.0,
                    help="Average NSPT value for bearing calculations"
                )
            
            st.subheader("🏗️ Additional Properties")
            col_w, col_l, col_d, col_load = st.columns(4)
            
            with col_w:
                wt_level = st.number_input("Water Table Level:", min_value=0.0, step=0.1, key="wt_6")
            with col_l:
                pile_length = st.number_input("Length of Pile:", min_value=0.0, step=0.1, key="length_6")
            with col_d:
                diameter = st.number_input("Diameter:", min_value=0.0, step=0.01, key="diam_6")
            with col_load:
                load = st.number_input("Load:", min_value=0.0, step=1.0, key="load_6")
            
            # Prepare features for 6-feature model
            features = [NS, NB, wt_level, pile_length, diameter, load]
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if st.button("🚀 Predict Settlement", type="primary"):
            try:
                if model_type == "31-Feature Detailed Model":
                    if 'full_model' in models and 'full_scaler' in models:
                        # Scale features
                        features_scaled = models['full_scaler'].transform([features])
                        
                        # Make prediction
                        prediction = models['full_model'].predict(features_scaled)[0][0]
                        
                        # Display results
                        st.success("✅ Prediction Complete!")
                        st.metric("Predicted Settlement", f"{prediction:.4f} mm")
                        st.info("**Model:** 31-Feature Bayesian Model\n**Training R²:** 0.9909 (99.09%)")
                        
                        # Show feature summary
                        with st.expander("📊 Feature Summary"):
                            st.write(f"**Total Features Used:** {len(features)}")
                            st.write(f"**Model Type:** ORIGINAL Bayesian Neural Network")
                            
                    else:
                        st.error("❌ 31-feature model or scaler not loaded")
                
                else:  # 6-Feature Model
                    if 'simplified_model' in models and 'simplified_scaler' in models:
                        # Scale features
                        features_scaled = models['simplified_scaler'].transform([features])
                        
                        # Make prediction
                        prediction = models['simplified_model'].predict(features_scaled)[0][0]
                        
                        # Display results
                        st.success("✅ Prediction Complete!")
                        st.metric("Predicted Settlement", f"{prediction:.4f} mm")
                        st.info("**Model:** 6-Feature Simplified Model\n**Training R²:** 0.9644 (96.44%)")
                        
                        # Show feature summary
                        with st.expander("📊 Feature Summary"):
                            st.write(f"**Features Used:** NS={NS}, NB={NB}")
                            st.write(f"**Pile Properties:** Length={pile_length}, Diameter={diameter}")
                            st.write(f"**Model Type:** ORIGINAL Simplified Bayesian Model")
                            
                    else:
                        st.error("❌ 6-feature model or scaler not loaded")
                        
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
        
        # Model status
        st.subheader("📈 Model Status")
        if 'full_model' in models:
            st.success("✅ 31-Feature Model: Ready")
        else:
            st.warning("⚠️ 31-Feature Model: Not loaded")
            
        if 'simplified_model' in models:
            st.success("✅ 6-Feature Model: Ready")
        else:
            st.warning("⚠️ 6-Feature Model: Not loaded")

if __name__ == "__main__":
    main()
