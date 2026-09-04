from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ---------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------

app = FastAPI(
    title="Loan Approval Prediction API",
    description="ML Loan Approval Prediction using Random Forest",
    version="1.0"
)


# ---------------------------------------------------
# Load trained ML model
# ---------------------------------------------------

model = joblib.load("loan_model.pkl")


# ---------------------------------------------------
# Request body
# ---------------------------------------------------

class LoanApplication(BaseModel):
    cibil_score: int
    monthly_income: float
    existing_emi: float
    loan_amount: float
    employment_years: int


# ---------------------------------------------------
# Home endpoint
# ---------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Loan Approval ML API is running"
    }


# ---------------------------------------------------
# Health endpoint
# ---------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "RandomForestClassifier"
    }


# ---------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------

@app.post("/predict")
def predict(application: LoanApplication):

    applicant = pd.DataFrame(
        [
            {
                "cibil_score": application.cibil_score,
                "monthly_income": application.monthly_income,
                "existing_emi": application.existing_emi,
                "loan_amount": application.loan_amount,
                "employment_years": application.employment_years
            }
        ]
    )

    prediction = model.predict(applicant)[0]

    probabilities = model.predict_proba(applicant)[0]

    approval_probability = probabilities[1]

    if prediction == 1:
        result = "Approved"
    else:
        result = "Rejected"

    return {
        "loan_status": result,
        "approval_probability": round(
            float(approval_probability) * 100,
            2
        )
    }
