
### 9b) `model-card.md` (starter)
```markdown
# Model Card — Credit Card Fraud Detection

**Intended use:** Educational/demo for fraud-risk scoring on PCA-transformed credit card data.  
**Data:** Kaggle Credit Card Fraud Detection (Time, V1..V28, Amount, Class).  
**Imbalance:** ~0.17% fraud.

**Model:** BalancedRandomForestClassifier (scikit-learn).  
**Thresholding:** F1-optimal on validation PR curve.  
**Primary metrics:** PR-AUC, Precision@K, Recall@K.

**Limitations:** PCA features; not production-ready for specific banks; potential domain shift.  
**Fairness & governance:** No PII; continue monitoring drift; retrain periodically.  
**Retraining plan:** Monthly with new labels; compare PR-AUC and P&L, re-pick threshold.
