<<<<<<< HEAD
# 🏗️ Pile Settlement Prediction ML Model

A machine learning web application for predicting pile settlement based on soil properties and pile characteristics.

## 🎯 Features

- **Dual Model Support**: Choose between 31-feature detailed model or 6-feature simplified model
- **High Accuracy**: Original models achieve R² of 0.9909 (31-feature) and 0.9644 (6-feature)
- **Web Interface**: User-friendly Bootstrap interface
- **Real-time Predictions**: Instant settlement predictions

## 🚀 Live Demo

Deploy to Render.com to get your live URL!

## 🛠️ Technology Stack

- **Backend**: Flask, Python 3.9
- **ML**: TensorFlow, Scikit-learn
- **Frontend**: Bootstrap 5, JavaScript
- **Deployment**: Render.com, Gunicorn

## 📊 Models

### Full Model (31 Features)
- **Training R²**: 0.9909 (99.09%)
- **Features**: Detailed soil layer properties (C.1-C.9, N30.1-N30.9, T.1-T.9), pile characteristics

### Simplified Model (6 Features) 
- **Training R²**: 0.9644 (96.44%)
- **Features**: NSPT values, basic pile properties

## Setup and Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up Windows Firewall rules (requires administrator privileges):
   - Run `setup_firewall.bat` as administrator
   - This will add the necessary firewall rules for the application

3. Run the application:
   ```
   python app.py
   ```

4. Access the application:
   - Local access: `http://localhost:5000`
   - Global access: `http://[your-ip-address]:5000`
   
   To find your IP address, open Command Prompt and type:
   ```
   ipconfig
   ```
   Look for the "IPv4 Address" under your active network adapter.

## How it Works

- The application uses a machine learning model to make predictions
- Enter values in the input fields:
  - Soil Properties (9 layers with Cohesion, N-SPT, and Thickness)
  - Additional Properties (Water Table Level, Length of Pile, Diameter, and Load)
- The application will process the inputs and return predictions

## Security Considerations

1. The application is accessible to anyone on your network
2. Consider implementing authentication if needed
3. Make sure your firewall is properly configured
4. Keep your system and dependencies up to date

## Customization

You can replace the current model with your own machine learning model by modifying the `initialize_model()` function in `app.py`.

## Using Your Pre-trained Model

The application expects two files in the root directory:
1. `model.h5` - Your pre-trained Keras/TensorFlow model
2. `scaler.pkl` - The scaler used to normalize your training data (optional)

To use your model:
1. Save your Keras/TensorFlow model using:
   ```python
   model.save('model.h5')
   ```

2. If you have a scaler, save it using:
   ```python
   import pickle
   with open('scaler.pkl', 'wb') as f:
       pickle.dump(scaler, f)
   ```

3. Place both files in the root directory of the application

4. Make sure your model expects the following features in order:
   - For each of the 9 layers:
     - Cohesion (C.1 to C.9)
     - N-SPT (N30.1 to N30.9)
     - Thickness (T.1 to T.9)
   - Additional properties:
     - Water Table Level (W.T.level)
     - Length of Pile (Length.of.pile)
     - Diameter (Diam.)
     - Load

5. Your model should:
   - Accept input shape of (None, 31) - representing the 31 features
   - Output shape of (None, 1) - representing the prediction

The application will automatically load your model when it starts. If there's any error loading the model, check the application logs for details.

## Deployment on Render (Free Hosting)

1. Create a [Render](https://render.com) account if you don't have one

2. Connect your GitHub repository:
   - Create a new GitHub repository
   - Push your code to GitHub:
     ```bash
     git init
     git add .
     git commit -m "Initial commit"
     git remote add origin <your-github-repo-url>
     git push -u origin main
     ```
   - In Render dashboard, click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Select the repository with this code

3. Configure the Web Service:
   - Name: Choose a name for your service
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Select the Free plan

4. Environment Variables:
   - Click on "Environment" tab
   - Add any necessary environment variables

5. Deploy:
   - Click "Create Web Service"
   - Wait for the deployment to complete
   - Your app will be available at: `https://<your-app-name>.onrender.com`

Note: The free tier of Render has some limitations:
- Spins down after 15 minutes of inactivity
- Limited bandwidth and compute resources
- May take a few seconds to spin up on first request
=======
# pile-settlement-predictor
>>>>>>> 2453e3c3262650b26a7687330ccdbd0e7a8b471e
