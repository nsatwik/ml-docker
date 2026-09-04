### Steps 
**Step 1**

**Install packages**
```bash
apt update
apt install docker.io -y
apt install -y python3-venv
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install pandas scikit-learn joblib
pip list
python train.py
```

**Step2**

**Run the FastAPI application**
```bash
pip install fastapi uvicorn
```
**Now start the API**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
**Open another SSH terminal and run**
```bash
curl http://localhost:8000/health
```
```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "cibil_score": 780,
  "monthly_income": 90000,
  "existing_emi": 10000,
  "loan_amount": 500000,
  "employment_years": 6
}'
```

**Step3**

**Dockerize & push to ECR**

```bash
docker build --no-cache -t loan-approval-api .
```

```bash
docker run -d   --name loan-api   -p 8000:8000   loan-approval-api
```

```bash
docker logs loan-api
```

**In new terminal**
```bash
curl http://localhost:8000/health
```
