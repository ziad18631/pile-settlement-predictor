"""
Quick model accuracy test and retrain with correct data mapping
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# Load the actual data
df = pd.read_excel('New.Collected.Data. (2.2).xlsx')
print(f"Data loaded: {df.shape}")
print(f"Target column 'S' range: {df['S'].min():.2f} to {df['S'].max():.2f}")

# Check current models
try:
    model_31 = tf.keras.models.load_model('final_model_kfold.h5')
    with open('scaler_kfold.pkl', 'rb') as f:
        scaler_31 = pickle.load(f)
    print("Loaded final_model_kfold.h5 and scaler_kfold.pkl")
except:
    try:
        model_31 = tf.keras.models.load_model('enhanced_bayesian_model_31.h5')
        with open('enhanced_scaler_31.pkl', 'rb') as f:
            scaler_31 = pickle.load(f)
        print("Loaded enhanced models")
    except Exception as e:
        print(f"Error loading models: {e}")
        model_31 = None

# Test current model if available
if model_31 is not None:
    # Prepare test data
    feature_cols = [col for col in df.columns if col != 'S']
    X = df[feature_cols].values
    y = df['S'].values
    
    X_test_scaled = scaler_31.transform(X[:50])  # Test first 50 samples
    y_test = y[:50]
    
    predictions = model_31.predict(X_test_scaled)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Current model performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        print(f"  Predicted: {predictions[i][0]:.2f}, Actual: {y_test[i]:.2f}")

# Create mapping for Streamlit interface
print(f"\nData column mapping:")
print(f"Available columns: {list(df.columns)}")

# Map to expected Streamlit inputs
# For 31-feature model: 9 layers × 3 properties + 4 additional = 31 features
# Current data has 30 features (excluding 'S' target)

# For 6-feature simplified model
six_feature_cols = ['NB', 'Load', 'Length of pile ', 'Diam.', 'W.T.level']
if 'NS' in df.columns:
    six_feature_cols.insert(0, 'NS')
elif 'N' in df.columns:
    six_feature_cols.insert(0, 'N')
else:
    six_feature_cols.insert(0, 'NB')  # Use NB as friction substitute

print(f"\n6-Feature mapping:")
for i, col in enumerate(six_feature_cols[:6]):
    print(f"  {i+1}. {col}")
