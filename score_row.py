import argparse, json
import pandas as pd
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="creditcard.csv")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--api", default="http://127.0.0.1:8000/score")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    payload = df.iloc[args.row].to_dict()
    payload = {k: float(v) for k, v in payload.items()}

    r = requests.post(args.api, json=payload, timeout=10)
    print(r.status_code, json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    main()
