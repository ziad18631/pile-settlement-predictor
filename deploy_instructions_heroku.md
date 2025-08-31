# Deploy to Heroku

## Prerequisites:
1. Install Heroku CLI
2. Create Heroku account

## Steps:
```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-ml-app-name

# Set Python version
echo "python-3.9.0" > runtime.txt

# Deploy
git add .
git commit -m "Deploy ML app"
git push heroku main
```

## Cost: $7/month for basic dyno
## Pros: Easy deployment, good documentation
## Cons: Not free anymore, limited resources
