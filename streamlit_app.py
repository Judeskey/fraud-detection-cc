import json
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("pipeline.pkl")
    with open("threshold.json") as f:
        thr = float(json.load(f).get("threshold", 0.5))
    return model, thr

pipeline, THRESHOLD = load_artifacts()

st.title("💳 Credit Card Fraud Detection — Demo")
st.write("Upload the Kaggle creditcard.csv (no PII; PCA features) or paste a single row as JSON.")

FEATURES = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14",
    "V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

def score_one(row_dict):
    X = pd.DataFrame([row_dict], columns=FEATURES)
    prob = float(pipeline.predict_proba(X)[0, 1])
    flag = int(prob >= THRESHOLD)
    return prob, flag

tab1, tab2 = st.tabs(["Upload CSV", "Paste JSON"])

with tab1:
    file = st.file_uploader("Upload creditcard.csv", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df_feat = df.drop(columns=["Class"]) if "Class" in df.columns else df
        st.write("Preview:", df_feat.head())
        row_idx = st.number_input("Row index", min_value=0, max_value=len(df_feat)-1, value=0, step=1)
        if st.button("Score selected row"):
            payload = df_feat.iloc[int(row_idx)].to_dict()
            payload = {k: float(v) for k, v in payload.items()}
            prob, flag = score_one(payload)
            st.metric("Fraud Probability", f"{prob:.4f}")
            st.metric("Decision (flag)", "FRAUD" if flag else "OK")
            st.caption(f"Threshold = {THRESHOLD:.4f}")

with tab2:
    import json as pyjson
    sample = {
        "Time": 0.0, "V1": -1.3598071336738, "V2": -0.0727811733098497, "V3": 2.53634673796914,
        "V4": 1.37815522427443, "V5": -0.338320769942518, "V6": 0.462387777762292, "V7": 0.239598554061257,
        "V8": 0.0986979012610507, "V9": 0.363786969611213, "V10": 0.0907941719789316, "V11": -0.551599533260813,
        "V12": -0.617800855762348, "V13": -0.991389847235408, "V14": -0.311169353699879, "V15": 1.46817697209427,
        "V16": -0.470400525259478, "V17": 0.207971241929242, "V18": 0.0257905801985591, "V19": 0.403992960255733,
        "V20": 0.251412098239705, "V21": -0.018306777944153, "V22": 0.277837575558899, "V23": -0.110473910188767,
        "V24": 0.0669280749146731, "V25": 0.128539358273528, "V26": -0.189114843888824, "V27": 0.133558376740387,
        "V28": -0.0210530534538215, "Amount": 149.62
    }
    json_in = st.text_area("Paste JSON payload", value=pyjson.dumps(sample, indent=2), height=300)
    if st.button("Score JSON"):
        try:
            payload = pyjson.loads(json_in)
            payload = {k: float(payload[k]) for k in FEATURES}
            prob, flag = score_one(payload)
            st.metric("Fraud Probability", f"{prob:.4f}")
            st.metric("Decision (flag)", "FRAUD" if flag else "OK")
            st.caption(f"Threshold = {THRESHOLD:.4f}")
        except Exception as e:
            st.error(f"Invalid JSON or missing fields: {e}")
