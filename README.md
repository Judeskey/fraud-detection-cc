# Credit Card Fraud Detection — Model + Demo

Detects fraudulent transactions on a highly imbalanced dataset (≈0.17% fraud).

**Stack:** scikit-learn, imbalanced-learn, Streamlit (demo), FastAPI (API)

## Quickstart (local)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py         # exports pipeline.pkl & threshold.json
streamlit run streamlit_app.py
# optional API:
# uvicorn app:app --reload
