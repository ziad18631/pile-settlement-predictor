# Deploy to Render.com

## Steps:
1. Create account at render.com
2. Connect your GitHub repository
3. Create new Web Service
4. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Environment: Python
   - Instance Type: Free tier (512MB RAM)

## Cost: FREE tier available
## Pros: Easy setup, automatic deployments
## Cons: Cold start delays on free tier

Your app is already configured with render.yaml!
