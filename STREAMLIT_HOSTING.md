# Streamlit Hosting Guide

## Quick Deploy Options

### 1. Streamlit Cloud (Easiest - Free)
1. Go to https://share.streamlit.io/
2. Connect your GitHub account
3. Select repository: `ziad18631/pile-settlement-predictor`
4. Set main file path: `streamlit_app.py`
5. Deploy automatically

### 2. Railway (Free tier available)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway deploy
```

### 3. Render (Free tier)
1. Go to https://render.com/
2. Connect GitHub repo
3. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

### 4. Heroku
```bash
# Install Heroku CLI, then:
heroku create your-app-name
git push heroku main
```

## Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

## Files Updated for Hosting
- `requirements.txt` - Added streamlit dependency
- `Procfile` - Updated for Streamlit instead of Flask
- `render.yaml` - Updated start command
- `streamlit_config.toml` - Dark theme config
- `start.sh` - Railway start script

## Environment Variables (if needed)
Most platforms will automatically set PORT. No additional variables required.
