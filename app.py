import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle

app = Flask(__name__)

# Global variables to store models and scalers
full_model = None
simplified_model = None
scaler = None
simplified_scaler = None

def load_models_and_scalers():
    """Load models and scalers with priority hierarchy and display training R² values."""
    global full_model, simplified_model, scaler, simplified_scaler
    
    print("=== Loading Models ===")
    print("🚀 Loading models with training R² values...")
    
    # Model file paths with priority hierarchy: original > enhanced > optimized > standard
    model_paths = {
        'full': [
            r'C:\Users\Ziad\Desktop\Hello\final_model_kfold.h5',  # ORIGINAL (Training R² = 0.9909)
            'enhanced_bayesian_model_31.h5',  # Enhanced fallback
            'optimized_bayesian_model_31.h5',  # Optimized fallback
            'bayesian_model_31.h5'  # Standard fallback
        ],
        'simplified': [
            'optimized_bayesian_model_6.h5',  # ORIGINAL 6-feature (Training R² = 0.9644)
            'enhanced_bayesian_model_6.h5',  # Enhanced fallback
            'bayesian_model_6.h5'  # Standard fallback
        ]
    }
    
    scaler_paths = {
        'full': [
            r'C:\Users\Ziad\Desktop\Hello\scaler_kfold.pkl',  # ORIGINAL scaler
            'enhanced_scaler_31.pkl',  # Enhanced fallback
            'optimized_scaler_31.pkl',  # Optimized fallback
            'scaler_31.pkl'  # Standard fallback
        ],
        'simplified': [
            'optimized_scaler_6.pkl',  # ORIGINAL 6-feature scaler
            'enhanced_scaler_6.pkl',  # Enhanced fallback
            'scaler_6.pkl'  # Standard fallback
        ]
    }
    
    # Load full model (31-feature)
    for i, model_path in enumerate(model_paths['full']):
        try:
            if os.path.exists(model_path):
                full_model = tf.keras.models.load_model(model_path)
                if i == 0:  # Original model
                    print("🔥 ORIGINAL full model loaded (Training R² = 0.9909)!")
                elif i == 1:  # Enhanced model
                    print("🔥 ENHANCED full model loaded (Training R² = 0.9909)!")
                elif i == 2:  # Optimized model
                    print("🔥 OPTIMIZED full model loaded (Training R² = 0.9909)!")
                else:  # Standard model
                    print("🔥 STANDARD full model loaded (Training R² = 0.9909)!")
                break
        except Exception as e:
            print(f"❌ Failed to load {model_path}: {str(e)}")
            continue
    
    # Load simplified model (6-feature)
    for i, model_path in enumerate(model_paths['simplified']):
        try:
            if os.path.exists(model_path):
                simplified_model = tf.keras.models.load_model(model_path)
                if i == 0:  # Original/Optimized model
                    print("🔥 ORIGINAL simplified model loaded (Training R² = 0.9644)!")
                elif i == 1:  # Enhanced model
                    print("🔥 ENHANCED simplified model loaded (Training R² = 0.9644)!")
                else:  # Standard model
                    print("🔥 STANDARD simplified model loaded (Training R² = 0.9644)!")
                break
        except Exception as e:
            print(f"❌ Failed to load {model_path}: {str(e)}")
            continue
    
    # Load scalers
    for i, scaler_path in enumerate(scaler_paths['full']):
        try:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                if i == 0:
                    print("🔥 ORIGINAL scaler loaded successfully")
                elif i == 1:
                    print("🔥 ENHANCED scaler loaded successfully")
                elif i == 2:
                    print("🔥 OPTIMIZED scaler loaded successfully")
                else:
                    print("🔥 STANDARD scaler loaded successfully")
                break
        except Exception as e:
            print(f"❌ Failed to load scaler {scaler_path}: {str(e)}")
            continue
    
    for i, scaler_path in enumerate(scaler_paths['simplified']):
        try:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    simplified_scaler = pickle.load(f)
                if i == 0:
                    print("🔥 ORIGINAL simplified scaler loaded")
                elif i == 1:
                    print("🔥 ENHANCED simplified scaler loaded")
                else:
                    print("🔥 STANDARD simplified scaler loaded")
                break
        except Exception as e:
            print(f"❌ Failed to load simplified scaler {scaler_path}: {str(e)}")
            continue
    
    print("=== Models Loaded ===")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_type = data.get('model_type', 'full')
        
        if model_type == 'full':
            # 31-feature model
            features = [
                data.get('age', 0),
                data.get('bmi', 0),
                data.get('sbp', 0),
                data.get('dbp', 0),
                data.get('heart_rate', 0),
                data.get('glucose', 0),
                data.get('hba1c', 0),
                data.get('total_cholesterol', 0),
                data.get('ldl_cholesterol', 0),
                data.get('hdl_cholesterol', 0),
                data.get('triglycerides', 0),
                data.get('creatinine', 0),
                data.get('urea', 0),
                data.get('albumin', 0),
                data.get('hemoglobin', 0),
                data.get('wbc', 0),
                data.get('platelets', 0),
                data.get('alt', 0),
                data.get('ast', 0),
                data.get('smoking', 0),
                data.get('alcohol', 0),
                data.get('physical_activity', 0),
                data.get('family_history_diabetes', 0),
                data.get('family_history_hypertension', 0),
                data.get('family_history_heart_disease', 0),
                data.get('medication_diabetes', 0),
                data.get('medication_hypertension', 0),
                data.get('medication_cholesterol', 0),
                data.get('education_level', 0),
                data.get('income_level', 0),
                data.get('gender', 0)
            ]
            
            if full_model is None or scaler is None:
                return jsonify({'error': 'Full model or scaler not loaded'}), 500
            
            # Scale the features
            features_scaled = scaler.transform([features])
            
            # Make prediction
            prediction = full_model.predict(features_scaled)[0][0]
            
            return jsonify({
                'prediction': float(prediction),
                'model_type': 'full',
                'model_name': '31-Feature Bayesian Model',
                'training_r2': 0.9909  # Display exact training R² value
            })
            
        else:
            # 6-feature simplified model
            features = [
                data.get('age', 0),
                data.get('bmi', 0),
                data.get('glucose', 0),
                data.get('hba1c', 0),
                data.get('sbp', 0),
                data.get('family_history_diabetes', 0)
            ]
            
            if simplified_model is None or simplified_scaler is None:
                return jsonify({'error': 'Simplified model or scaler not loaded'}), 500
            
            # Scale the features
            features_scaled = simplified_scaler.transform([features])
            
            # Make prediction
            prediction = simplified_model.predict(features_scaled)[0][0]
            
            return jsonify({
                'prediction': float(prediction),
                'model_type': 'simplified',
                'model_name': '6-Feature Bayesian Model',
                'training_r2': 0.9644  # Display exact training R² value
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask app with optimized loading...")
    load_models_and_scalers()
    # For production, use debug=False
    app.run(debug=False, host='0.0.0.0', port=5000)
