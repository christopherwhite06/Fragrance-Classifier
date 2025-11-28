# FragranceMatch AI  
A machine learning insight tool that predicts **consumer personas**, **demographic preferences**, and **P&G product-fit** based on natural-language fragrance descriptions.

---

## 🔍 Overview
FragranceMatch AI transforms any free-text scent description into a set of predictive insights using:

- Sentence-BERT (all-mpnet-base-v2) embeddings  
- Multinomial Logistic Regression classifiers  
- A Streamlit-powered insights dashboard  

This project is designed to demonstrate how AI can support **product targeting**, **consumer understanding**, and **fragrance design** across major P&G categories.

---

## 🎯 Key Features

### **1. Consumer Persona Prediction**
Predicts probability distributions for:
- Gender  
- Age Group (with average age)  
- Mood preference  
- Country / region  

### **2. Product-Fit Classification**
Identifies which P&G category the scent best matches:
- Ariel / Tide (Laundry Fresh)  
- Pampers (Baby Care)  
- Gillette (Men’s Grooming)  
- Head & Shoulders (Menthol Fresh Shampoo)  
- Olay (Warm Floral Skin Care)  
- Febreze (Home Freshening)  
- Fairy (Lemon Dishwashing)  

### **3. Insights Dashboard**
Interactive visualisations including:
- Probability bar charts  
- Age gauge visualisation  
- Persona summary with icons  
- Clean, modern UI  

### **4. Fully Modular Pipeline**
- `generate_dataset.py` → creates synthetic fragrance dataset  
- `embedder.py` → generates BERT embeddings  
- `train.py` → trains all probabilistic classifiers  
- `predict.py` → runs inference + returns structured insights  
- `web_app.py` → Streamlit dashboard  

---

## How to run

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/web_app.py