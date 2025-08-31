# Quick Git Setup Script for Render Deployment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Render.com Deployment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project directory
Set-Location "C:\Users\Ziad\Desktop\Master Thesis\WEB_3"
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Green
Write-Host ""

# Check if git is installed
try {
    git --version
    Write-Host "Git is installed ✓" -ForegroundColor Green
} catch {
    Write-Host "Git is not installed. Please install Git first." -ForegroundColor Red
    Write-Host "Download from: https://git-scm.com/download/windows"
    exit
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Create GitHub repository at: https://github.com/new" -ForegroundColor White
Write-Host "2. Repository name: pile-settlement-predictor" -ForegroundColor White
Write-Host "3. Make it PUBLIC" -ForegroundColor White
Write-Host "4. Don't initialize with README" -ForegroundColor White
Write-Host ""

$continue = Read-Host "Have you created the GitHub repository? (y/n)"

if ($continue -eq "y" -or $continue -eq "Y") {
    $username = Read-Host "Enter your GitHub username"
    $repoName = Read-Host "Enter repository name (default: pile-settlement-predictor)"
    
    if (-not $repoName) {
        $repoName = "pile-settlement-predictor"
    }
    
    Write-Host ""
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    
    # Initialize git
    git init
    
    # Add all files
    Write-Host "Adding files..." -ForegroundColor Yellow
    git add .
    
    # Commit
    Write-Host "Creating initial commit..." -ForegroundColor Yellow
    git commit -m "Initial commit - ML pile settlement predictor"
    
    # Add remote origin
    $repoUrl = "https://github.com/$username/$repoName.git"
    Write-Host "Adding remote origin: $repoUrl" -ForegroundColor Yellow
    git remote add origin $repoUrl
    
    # Push to GitHub
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    git branch -M main
    git push -u origin main
    
    Write-Host ""
    Write-Host "✅ Success! Code pushed to GitHub" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Go to https://render.com" -ForegroundColor White
    Write-Host "2. Sign up with GitHub account" -ForegroundColor White
    Write-Host "3. Create New Web Service" -ForegroundColor White
    Write-Host "4. Connect your repository: $repoName" -ForegroundColor White
    Write-Host "5. Use these settings:" -ForegroundColor White
    Write-Host "   - Build Command: pip install -r requirements.txt" -ForegroundColor Gray
    Write-Host "   - Start Command: gunicorn app:app" -ForegroundColor Gray
    Write-Host "   - Instance Type: Free" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Your app will be live at: https://$repoName.onrender.com" -ForegroundColor Green
    
} else {
    Write-Host ""
    Write-Host "Please create a GitHub repository first, then run this script again." -ForegroundColor Yellow
    Write-Host "Go to: https://github.com/new" -ForegroundColor White
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
