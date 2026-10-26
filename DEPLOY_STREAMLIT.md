# GitHub Push & Streamlit Cloud Deployment Guide
# ప్రాజెక్ట్‌ను GitHub లోకి Push చేయడం & Streamlit లోకి Deploy చేసే విధానం

**Project Title:** Scalable Clustering of Heterogeneous Big Data via Distributed Ensemble Gradient-boosting Models (XGBoost on Spark/Dask), Applied to Healthcare IoT Patient Monitoring at Scale  
**Supervisor / Guide:** Ms. G. Hema Sri  
**Course:** Big Data Analytics  

---

## Part 1: GitHub లో కొత్త Repository తయారుచేయడం (Create GitHub Repository)

1. **GitHub లోకి లాగిన్ అవ్వండి:** [https://github.com/](https://github.com/) ఓపెన్ చేసి మీ అకౌంట్ లోకి లాగిన్ అవ్వండి.
2. పైన కుడివైపు ఉన్న **`+`** ఐకాన్ క్లిక్ చేసి **`New repository`** ఎంచుకోండి.
3. Repository వివరాలు ఇవ్వండి:
   - **Repository name:** `healthcare-iot-bigdata-clustering`
   - **Visibility:** `Public` (Streamlit Cloud ఉచిత డిప్లాయ్‌మెంట్ కోసం Public రికమండ్ చేయబడింది).
   - *గమనిక:* "Add a README file" లేదా "Add .gitignore" టిక్ చేయవద్దు (ఎందుకంటే ప్రాజెక్ట్‌లో ఇప్పటికే అన్ని ఫైల్స్ సిద్ధంగా ఉన్నాయి).
4. **`Create repository`** బటన్ క్లిక్ చేయండి.
5. మీ రిపోజిటరీ లింక్ కాపీ చేసుకోండి (ఉదాహరణకి: `https://github.com/YOUR_USERNAME/healthcare-iot-bigdata-clustering.git`).

---

## Part 2: కోడ్‌ను GitHub లోకి Push చేయడం (Push Code to GitHub)

### విధానం A: ఒకే క్లిక్ స్క్రిప్ట్ ద్వారా (Easiest Method)
ప్రాజెక్ట్ ఫోల్డర్ (`C:\Users\vivek\.gemini\antigravity\scratch\healthcare_iot_bigdata_clustering`) లో `push_to_github.bat` లేదా `push_to_github.ps1` సిద్ధంగా ఉంది:

PowerShell లో:
```powershell
cd C:\Users\vivek\.gemini\antigravity\scratch\healthcare_iot_bigdata_clustering
.\push_to_github.ps1 -RepoUrl "https://github.com/YOUR_USERNAME/healthcare-iot-bigdata-clustering.git"
```

లేదా Command Prompt లో:
```cmd
cd C:\Users\vivek\.gemini\antigravity\scratch\healthcare_iot_bigdata_clustering
push_to_github.bat https://github.com/YOUR_USERNAME/healthcare-iot-bigdata-clustering.git
```

---

### విధానం B: మ్యాన్యువల్ Git కమాండ్స్ ద్వారా (Step-by-Step Manual Commands)
మీకు నచ్చితే స్వయంగా ఈ కమాండ్స్ రన్ చేయవచ్చు:

```bash
cd C:\Users\vivek\.gemini\antigravity\scratch\healthcare_iot_bigdata_clustering

# 1. Initialize local Git
git init

# 2. Stage all project files
git add .

# 3. Commit files
git commit -m "Initial commit: Healthcare IoT Big Data Clustering via Distributed XGBoost on Spark"

# 4. Set branch to main
git branch -M main

# 5. Connect to your GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/healthcare-iot-bigdata-clustering.git

# 6. Push code to GitHub
git push -u origin main
```

---

## Part 3: Streamlit Community Cloud లోకి Deploy చేయడం (Deploy to Streamlit Cloud in 3 Clicks)

కోడ్ GitHub లోకి వెళ్లిన వెంటనే, ఉచితంగా లైవ్ వెబ్ లింక్ పొందడానికి:

1. **Streamlit Cloud వెబ్‌సైట్‌కు వెళ్లండి:** [https://share.streamlit.io/](https://share.streamlit.io/)
2. **Continue with GitHub** క్లిక్ చేసి సైన్ ఇన్ అవ్వండి.
3. పైన కుడివైపు **`Create app`** లేదా **`New app`** క్లిక్ చేయండి.
4. కింది వివరాలను ఎంపిక చేయండి:
   - **Repository:** `YOUR_USERNAME/healthcare-iot-bigdata-clustering`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** మీ ఇష్టమైన పేరు (ఉదా: `healthcare-iot-analytics.streamlit.app`)
5. **`Deploy!`** బటన్ నొక్కండి.

2 నుండి 3 నిమిషాల్లో మీ యాప్ ప్రపంచవ్యాప్తంగా లైవ్ అవుతుంది! 
మీరు పొందే లింక్ ని మీ గైడ్ **Ms. G. Hema Sri** గారికి, మీ రెజ్యూమ్‌లో లేదా ప్రాజెక్ట్ డెమోలో నేరుగా షేర్ చేయవచ్చు! 🎉

---

## Part 4: లోకల్ గా Streamlit App రన్ చేయడం (Run Locally)

మీ సిస్టమ్‌లో వెంటనే చూసుకోవడానికి:
```bash
pip install -r requirements.txt
streamlit run app.py
```
బటన్ నొక్కగానే బ్రౌజర్‌లో `http://localhost:8501` లో ఇంటరాక్టివ్ డాష్‌బోర్డ్ ఓపెన్ అవుతుంది.
