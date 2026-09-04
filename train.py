import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------------------------------------------
# 1. Create sample historical loan applicant dataset
# ---------------------------------------------------

data = {
    "cibil_score": [
        780, 750, 720, 690, 650,
        620, 580, 540, 810, 790,
        760, 710, 680, 640, 600,
        560, 830, 770, 700, 610,
        740, 660, 590, 520, 800,
        730, 670, 550, 820, 630
    ],

    "monthly_income": [
        90000, 80000, 70000, 60000, 50000,
        45000, 35000, 30000, 120000, 100000,
        85000, 65000, 55000, 48000, 40000,
        32000, 150000, 95000, 62000, 42000,
        78000, 52000, 38000, 28000, 110000,
        72000, 51000, 33000, 140000, 46000
    ],

    "existing_emi": [
        10000, 12000, 15000, 18000, 20000,
        22000, 18000, 17000, 15000, 10000,
        12000, 16000, 18000, 20000, 21000,
        17000, 20000, 12000, 15000, 19000,
        14000, 16000, 18000, 16000, 13000,
        15000, 19000, 15000, 18000, 20000
    ],

    "loan_amount": [
        500000, 600000, 700000, 800000, 900000,
        700000, 800000, 600000, 1000000, 700000,
        650000, 750000, 850000, 700000, 600000,
        500000, 1200000, 800000, 700000, 650000,
        600000, 750000, 700000, 550000, 900000,
        650000, 800000, 600000, 1100000, 700000
    ],

    "employment_years": [
        6, 5, 4, 4, 3,
        3, 2, 1, 8, 7,
        6, 5, 4, 3, 2,
        1, 10, 7, 5, 2,
        5, 4, 2, 1, 8,
        5, 3, 1, 9, 3
    ],

    # 1 = Approved
    # 0 = Rejected
    "loan_approved": [
        1, 1, 1, 1, 0,
        0, 0, 0, 1, 1,
        1, 1, 1, 0, 0,
        0, 1, 1, 1, 0,
        1, 1, 0, 0, 1,
        1, 1, 0, 1, 0
    ]
}


# ---------------------------------------------------
# 2. Convert dataset into DataFrame
# ---------------------------------------------------

df = pd.DataFrame(data)


# ---------------------------------------------------
# 3. Define features and target
# ---------------------------------------------------

X = df[
    [
        "cibil_score",
        "monthly_income",
        "existing_emi",
        "loan_amount",
        "employment_years"
    ]
]

y = df["loan_approved"]


# ---------------------------------------------------
# 4. Split dataset
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------
# 5. Create Random Forest model
# ---------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ---------------------------------------------------
# 6. Train model
# ---------------------------------------------------

model.fit(X_train, y_train)


# ---------------------------------------------------
# 7. Evaluate model
# ---------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Model Accuracy: {accuracy:.2f}")


# ---------------------------------------------------
# 8. Save trained model
# ---------------------------------------------------

joblib.dump(
    model,
    "loan_model.pkl"
)

print("loan_model.pkl created successfully")
