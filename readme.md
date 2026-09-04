### Steps 
**Step 1**
```bash
apt update
apt install -y python3-venv
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install pandas scikit-learn joblib
pip list
python train.py
```
