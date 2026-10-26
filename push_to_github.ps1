param(
    [Parameter(Mandatory=$false)]
    [string]$RepoUrl
)

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host " Healthcare IoT Big Data Analytics - GitHub Push & Streamlit Deploy" -ForegroundColor Cyan
Write-Host " Supervisor: Ms. G. Hema Sri | Course: Big Data Analytics" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan

if (-not $RepoUrl) {
    $RepoUrl = Read-Host "Please enter your GitHub repository URL (e.g., https://github.com/username/repo.git)"
}

if (-not $RepoUrl) {
    Write-Host "[ERROR] Repository URL cannot be empty." -ForegroundColor Red
    exit 1
}

Write-Host "`n[1/5] Initializing Git repository..." -ForegroundColor Yellow
git init

Write-Host "`n[2/5] Staging files..." -ForegroundColor Yellow
git add .

Write-Host "`n[3/5] Creating commit..." -ForegroundColor Yellow
git commit -m "Initial commit: Scalable Clustering of Heterogeneous Healthcare IoT Big Data via Distributed XGBoost"

Write-Host "`n[4/5] Setting main branch & remote..." -ForegroundColor Yellow
git branch -M main
git remote remove origin 2>$null
git remote add origin $RepoUrl

Write-Host "`n[5/5] Pushing to GitHub ($RepoUrl)..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=======================================================================" -ForegroundColor Green
    Write-Host " SUCCESS! Project pushed to GitHub." -ForegroundColor Green
    Write-Host "=======================================================================" -ForegroundColor Green
    Write-Host "To deploy on Streamlit Community Cloud for FREE:" -ForegroundColor Cyan
    Write-Host " 1. Visit: https://share.streamlit.io/" -ForegroundColor White
    Write-Host " 2. Sign in with GitHub" -ForegroundColor White
    Write-Host " 3. Click 'New app' and pick your repo" -ForegroundColor White
    Write-Host " 4. Main file path: app.py" -ForegroundColor White
    Write-Host " 5. Click 'Deploy!'" -ForegroundColor Green
} else {
    Write-Host "`n[!] Push failed. Verify your repository URL and GitHub credentials." -ForegroundColor Red
}
