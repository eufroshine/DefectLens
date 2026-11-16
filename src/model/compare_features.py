import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, average_precision_score)
from xgboost import XGBClassifier

VERSIONS = ["1.3", "1.4", "1.5", "1.6", "1.7"]
PROMISE_COLS = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm",
                "lcom3", "loc", "dam", "moa", "mfa", "cam", "ic", "cbm",
                "amc", "max_cc", "avg_cc"]
GIT_COLS = ["change_freq", "lines_added", "lines_deleted", "churn", "n_authors"]

dfs = []
for v in VERSIONS:
    d = pd.read_csv(f"data/processed/merged_ant-{v}.csv")
    d["version"] = v
    dfs.append(d)
data = pd.concat(dfs, ignore_index=True)
data["bug"] = (data["bug"] > 0).astype(int)
data[GIT_COLS] = data[GIT_COLS].fillna(0)

train = data[data["version"] != "1.7"]   # rilis lama = belajar
test = data[data["version"] == "1.7"]    # rilis terbaru = ujian
print("Data latih:", len(train), "| data uji:", len(test),
      "| proporsi bug di uji:", round(test["bug"].mean(), 3))


def models(y):
    pos, neg = y.sum(), (y == 0).sum()
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            scale_pos_weight=neg / max(pos, 1),
            eval_metric="logloss", random_state=42),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000)),
    }


rows = []
for fs_name, cols in {"A: PROMISE saja": PROMISE_COLS,
                      "B: PROMISE + Git": PROMISE_COLS + GIT_COLS}.items():
    for m_name, model in models(train["bug"]).items():
        model.fit(train[cols], train["bug"])
        pred = model.predict(test[cols])
        proba = model.predict_proba(test[cols])[:, 1]
        rows.append({
            "fitur": fs_name, "model": m_name,
            "precision": precision_score(test["bug"], pred, zero_division=0),
            "recall": recall_score(test["bug"], pred, zero_division=0),
            "f1": f1_score(test["bug"], pred, zero_division=0),
            "pr_auc": average_precision_score(test["bug"], proba),
        })

res = pd.DataFrame(rows)
res.to_csv("data/processed/compare_features_ant.csv", index=False)
print(res.round(3).to_string(index=False))