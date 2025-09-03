# train_model.py
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from imblearn.ensemble import BalancedRandomForestClassifier

# 1) Load data (CSV must be in the same folder)
df = pd.read_csv("creditcard.csv")

# 2) Split features/labels
y = df["Class"].astype(int)
X = df.drop(columns=["Class"])

# 3) Train/validation split (keep imbalance in val)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Model that handles imbalance internally (no SMOTE leakage)
clf = BalancedRandomForestClassifier(
    n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)

# 5) Choose threshold on validation (F1-optimal)
probs_val = clf.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, probs_val)
f1 = (2 * prec * rec) / (prec + rec + 1e-9)
thr_opt = float(thr[max(range(len(thr)), key=lambda i: f1[i])])

# 6) Save artifacts for demo/API
joblib.dump(clf, "pipeline.pkl")
with open("threshold.json", "w") as f:
    json.dump({"threshold": thr_opt}, f)

print("Saved pipeline.pkl and threshold.json")
print(f"Chosen threshold (F1-opt): {thr_opt:.4f}")
