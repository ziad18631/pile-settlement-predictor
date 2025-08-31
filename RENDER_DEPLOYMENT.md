# 🚀 Render.com Deployment Instructions

## Step 1: Create GitHub Repository

1. **Go to GitHub.com and create new repository**
   - Repository name: `pile-settlement-predictor` (or your choice)
   - Make it public
   - Don't initialize with README (you already have one)

2. **Initialize Git in your project folder**
   Open PowerShell in your WEB_3 directory and run:
   ```bash
   cd "C:\Users\Ziad\Desktop\Master Thesis\WEB_3"
   git init
   git add .
   git commit -m "Initial commit - ML pile settlement predictor"
   ```

3. **Connect to GitHub and push**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/pile-settlement-predictor.git
   git branch -M main
   git push -u origin main
   ```

## Step 2: Deploy to Render.com

1. **Sign up at Render.com**
   - Go to https://render.com
   - Click "Get Started for Free"
   - Sign up with your GitHub account

2. **Create New Web Service**
   - Click "New +" button
   - Select "Web Service"
   - Click "Connect account" to connect GitHub
   - Find and select your repository: `pile-settlement-predictor`

3. **Configure Service Settings**
   - **Name**: `pile-settlement-predictor` (or your choice)
   - **Environment**: `Python`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

4. **Advanced Settings**
   - **Instance Type**: `Free` (0.5 CPU, 512MB RAM)
   - **Python Version**: Will auto-detect from requirements

5. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for build and deployment
   - Your app will be live at: `https://pile-settlement-predictor.onrender.com`

## Step 3: Verify Deployment

1. **Check Build Logs**
   - Render will show real-time build logs
   - Look for "Models Loaded" messages
   - Ensure no errors in TensorFlow loading

2. **Test Your App**
   - Visit your live URL
   - Test both model types
   - Verify predictions work

## ⚠️ Important Notes

### File Size Considerations
Your model files are large (~100MB+). Render free tier has some limitations:
- 512MB RAM
- Longer cold start times
- May timeout on first load

### Optimization Tips
1. **Model files are included** (render.yaml configured correctly)
2. **Dependencies optimized** (requirements.txt ready)
3. **Production server** (gunicorn configured)

### If Deployment Fails
Common issues and solutions:

**Issue**: Build timeout
**Solution**: Models too large for free tier
```bash
# Consider model compression or paid tier
```

**Issue**: Memory error
**Solution**: Reduce model complexity or upgrade to paid tier

**Issue**: TensorFlow import error
**Solution**: Check requirements.txt has correct versions

## Step 4: Custom Domain (Optional)

1. **Free subdomain**: `your-app-name.onrender.com`
2. **Custom domain**: Available on paid plans ($7/month)

## 🎉 Success!

Once deployed, you'll have:
- ✅ Live ML model accessible worldwide
- ✅ HTTPS enabled automatically
- ✅ Automatic deployments from GitHub
- ✅ Professional URL to share

## Next Steps

1. **Share your URL** with stakeholders
2. **Monitor usage** in Render dashboard
3. **Update models** by pushing to GitHub (auto-deploys)
4. **Consider paid tier** if you need more resources

Your ML model is now live and accessible globally! 🌍
