from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from src.predict import predict_customer
from src.shap_explainer import explain_customer

app = FastAPI(title="Customer Categorizer API")


# -----------------------------
# Request schema
# -----------------------------
class CustomerInput(BaseModel):
    Year_Birth: int
    Education: str
    Marital_Status: str
    Income: float

    Kidhome: int
    Teenhome: int
    Recency: int

    MntWines: float
    MntFruits: float
    MntMeatProducts: float
    MntFishProducts: float
    MntSweetProducts: float
    MntGoldProds: float

    NumDealsPurchases: int
    NumWebPurchases: int
    NumCatalogPurchases: int
    NumStorePurchases: int
    NumWebVisitsMonth: int

    AcceptedCmp1: Optional[int] = 0
    AcceptedCmp2: Optional[int] = 0
    AcceptedCmp3: Optional[int] = 0
    AcceptedCmp4: Optional[int] = 0
    AcceptedCmp5: Optional[int] = 0
    Response: Optional[int] = 0
    Complain: Optional[int] = 0


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "API running"}


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: CustomerInput):
    prediction = predict_customer(data.dict())
    return {"Customer_Category": prediction}


# -----------------------------
# Explainability endpoint
# -----------------------------
@app.post("/explain")
def explain(data: CustomerInput):
    input_dict = data.dict()
    prediction = predict_customer(input_dict)
    explanation = explain_customer(input_dict)

    return {
        "Customer_Category": prediction,
        "Top_Features": explanation
    }
