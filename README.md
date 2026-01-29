# 🚀 Customer Categorizer API

A **production‑ready Machine Learning API** that categorizes customers into **Low Value, Medium Value, or High Value** segments based on demographic and purchasing behavior.

Built end‑to‑end with **Python, Scikit‑learn, FastAPI, SHAP, Docker**, and designed for **real‑world deployment (Render‑ready)**.

---

## 1️⃣ Why this project matters (Recruiter view 👀)

This project demonstrates **much more than model training**:

* ✅ End‑to‑end ML pipeline (data → features → model → API)
* ✅ Clean project structure (industry‑style `src/` layout)
* ✅ Model explainability using **SHAP**
* ✅ REST API with **FastAPI + Swagger UI**
* ✅ Dockerized for cloud deployment
* ✅ Ready for platforms like **Render / AWS / GCP**

👉 This mirrors how ML systems are actually built in companies.

---

## 2️⃣ Key Features

1. **Customer Segmentation Model**

   * Predicts: `Low Value`, `Medium Value`, `High Value`
   * Trained on marketing campaign data

2. **FastAPI Backend**

   * `/predict` → Customer category
   * `/explain` → SHAP‑based feature importance
   * `/health` → Service health check

3. **Explainable AI (XAI)**

   * Uses **SHAP TreeExplainer**
   * Returns top contributing features per prediction

4. **Production‑Grade Design**

   * Modular codebase
   * Consistent preprocessing at train & inference time
   * Docker support

---

## 3️⃣ Project Architecture

```text
customer-categorizer-project/
│
├── data/                     # Raw & processed datasets
├── models/                   # Trained ML model (.pkl)
├── screenshots/              # API & Swagger screenshots
├── src/
│   ├── app.py                # FastAPI app entrypoint
│   ├── model_training.py     # Model training pipeline
│   ├── model_loader.py       # Model loading logic
│   ├── feature_engineering.py# Feature creation
│   ├── features.py           # Shared feature builder
│   ├── predict.py            # Prediction logic
│   ├── shap_explainer.py     # SHAP explanations
│   ├── data_preprocessing.py # Data cleaning
│   └── customer_labeling.py  # Target generation
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4️⃣ Tech Stack

| Layer          | Tools                       |
| -------------- | --------------------------- |
| Language       | Python                      |
| ML             | Scikit‑learn, NumPy, Pandas |
| API            | FastAPI, Uvicorn            |
| Explainability | SHAP                        |
| DevOps         | Docker                      |
| Deployment     | Render (Docker‑based)       |

---

## 5️⃣ API Endpoints

### 🔹 Health Check

```
GET /health
```

Returns service status.

---

### 🔹 Predict Customer Category

```
POST /predict
```

**Sample Request**

```json
{
  "Year_Birth": 1988,
  "Education": "Graduation",
  "Marital_Status": "Married",
  "Income": 52000,
  "Kidhome": 1,
  "Teenhome": 0,
  "Recency": 30,
  "MntWines": 300,
  "MntFruits": 50,
  "MntMeatProducts": 200,
  "MntFishProducts": 40,
  "MntSweetProducts": 30,
  "MntGoldProds": 20,
  "NumDealsPurchases": 2,
  "NumWebPurchases": 6,
  "NumCatalogPurchases": 1,
  "NumStorePurchases": 5,
  "NumWebVisitsMonth": 4,
  "AcceptedCmp1": 0,
  "AcceptedCmp2": 0,
  "AcceptedCmp3": 0,
  "AcceptedCmp4": 0,
  "AcceptedCmp5": 0,
  "Complain": 0
}
```

**Response**

```json
{
  "Customer_Category": "Low Value"
}
```

---

### 🔹 Explain Prediction (SHAP)

```
POST /explain
```

**Response**

```json
{
  "Customer_Category": "Low Value",
  "Top_Features": {
    "num__Total_Spending": 0.30,
    "num__NumCatalogPurchases": -0.27,
    "num__MntWines": 0.07
  }
}
```

This helps business users understand **why** a customer falls into a category.

---

## 6️⃣ Swagger UI (Live API Testing)

FastAPI auto‑generated docs:

📍 `http://localhost:8000/docs`

### 📸 Screenshots

link: https://github.com/aadity-singh/customer-categorizer-api/tree/main/screenshots

## 7️⃣ Run Locally (Without Docker)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start API
uvicorn src.app:app --reload
```

## 8️⃣ Run with Docker 🐳

```bash
# Build image
docker build -t customer-categorizer .

# Run container
docker run -p 8000:8000 customer-categorizer
```

---

## 9️⃣ Deployment (Render)

1. Push repo to GitHub
2. Create **New Web Service** on Render
3. Select **Docker** runtime
4. Set port: `8000`
5. Deploy 🚀

---

## 🔟 Model Performance

* High overall accuracy on validation set
* Handles class imbalance
* Uses consistent preprocessing pipeline

---

## 1️⃣1️⃣ What I learned from this project

* Building ML systems ≠ just training models
* Importance of **feature parity** between training & inference
* Handling **multiclass SHAP explanations**
* Dockerizing ML APIs for real deployments

---

## 1️⃣2️⃣ Future Improvements

* Authentication (JWT)
* Model monitoring & drift detection
* CI/CD pipeline
* Cloud storage for models

---

## 👨‍💻 Author

**Aadity Singh**
Aspiring Data Scientist / ML Engineer

📌 If you’re a recruiter: this project reflects **real‑world ML engineering practices**, not just notebooks.

⭐ If you like this project, don’t forget to star the repo!
