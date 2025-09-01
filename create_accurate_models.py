"""
Create properly matched models for the actual data structure
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# Load data
df = pd.read_excel('New.Collected.Data. (2.2).xlsx')
print(f"Data shape: {df.shape}")

# Prepare features and target
feature_columns = [col for col in df.columns if col != 'S']
X = df[feature_columns].values
y = df['S'].values

print(f"Features: {len(feature_columns)} - {feature_columns}")
print(f"Target range: {y.min():.2f} to {y.max():.2f}")

# === Train 30-feature model (matching actual data) ===
print("\n=== Training 30-Feature Model ===")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler_30 = StandardScaler()
X_train_scaled = scaler_30.fit_transform(X_train)
X_test_scaled = scaler_30.transform(X_test)

# Create model
model_30 = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(30,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model_30.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Train model
history = model_30.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=150,
    batch_size=32,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
    ]
)

# Evaluate
y_pred = model_30.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"30-Feature Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save model and scaler
model_30.save('accurate_model_30.h5')
with open('accurate_scaler_30.pkl', 'wb') as f:
    pickle.dump(scaler_30, f)

print("Saved: accurate_model_30.h5 and accurate_scaler_30.pkl")

# === Train 6-feature simplified model ===
print("\n=== Training 6-Feature Simplified Model ===")

# Select key features for 6-feature model
key_features = ['NB', 'Load', 'Length of pile ', 'Diam.', 'W.T.level', 'N']
key_indices = [feature_columns.index(feat) for feat in key_features if feat in feature_columns]

# If we don't have exactly 6, use the most important ones
if len(key_indices) < 6:
    # Add other important features
    additional = ['c', 'φ', 'E', 'γd', 'qc', 'LL']
    for feat in additional:
        if feat in feature_columns and len(key_indices) < 6:
            key_indices.append(feature_columns.index(feat))

key_indices = key_indices[:6]  # Take exactly 6
X_6 = X[:, key_indices]

print(f"6-Feature columns: {[feature_columns[i] for i in key_indices]}")

# Split and scale
X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(X_6, y, test_size=0.2, random_state=42)

scaler_6 = StandardScaler()
X_train_6_scaled = scaler_6.fit_transform(X_train_6)
X_test_6_scaled = scaler_6.transform(X_test_6)

# Create 6-feature model
model_6 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model_6.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Train
history_6 = model_6.fit(
    X_train_6_scaled, y_train_6,
    validation_data=(X_test_6_scaled, y_test_6),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    ]
)

# Evaluate
y_pred_6 = model_6.predict(X_test_6_scaled)
r2_6 = r2_score(y_test_6, y_pred_6)
rmse_6 = np.sqrt(mean_squared_error(y_test_6, y_pred_6))

print(f"6-Feature Model Performance:")
print(f"R² Score: {r2_6:.4f}")
print(f"RMSE: {rmse_6:.4f}")

# Save
model_6.save('accurate_model_6.h5')
with open('accurate_scaler_6.pkl', 'wb') as f:
    pickle.dump(scaler_6, f)

print("Saved: accurate_model_6.h5 and accurate_scaler_6.pkl")

# Show sample predictions
print(f"\nSample predictions (6-feature model):")
for i in range(5):
    print(f"Predicted: {y_pred_6[i][0]:.2f}, Actual: {y_test_6[i]:.2f}, Error: {abs(y_pred_6[i][0] - y_test_6[i]):.2f}")
