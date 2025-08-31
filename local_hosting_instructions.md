# Local Network Hosting

## Option 1: Flask Development Server (Current)
Your app is already running on: http://localhost:5000

To make it accessible on your local network:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

Then access via: http://YOUR_IP_ADDRESS:5000

## Option 2: Production Server (Recommended)
```bash
# Install gunicorn (already in requirements.txt)
pip install gunicorn

# Run with gunicorn
gunicorn --bind 0.0.0.0:5000 app:app

# Or with more workers
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

## Option 3: Using ngrok (Public tunnel)
```bash
# Install ngrok
# Download from ngrok.com

# Run your app locally first
python app.py

# In another terminal, create tunnel
ngrok http 5000
```

This gives you a public URL like: https://abc123.ngrok.io
