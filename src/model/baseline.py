import glob, os, re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, average_precision_score)
from xgboost import XGBClassifier

BASE = "data/raw/PROMISE-backup-master/bug-data"
LABEL = "bug"
DROP = ["name", "version", "name.1"]


def version_key(path, project):
    name = os.path.basename(path).replace(project, "")
    return tuple(int(x) for x in re.findall(r"\d+", name))


def load_versions(project):
    files = glob.glob(f"{BASE}/{project}/{project}-*.csv")
    files.sort(key=lambda f: version_key(f, project))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df[LABEL] = (df[LABEL] > 0).astype(int)  # jumlah bug -> biner
        df = df.drop(columns=[c for c in DROP if c in df.columns])
        dfs.append((os.path.basename(f), df))
    return dfs


def get_models(y_train):
    pos, neg = y_train.sum(), (y_train == 0).sum()
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
projects = sorted(d for d in os.listdir(BASE)
                  if os.path.isdir(os.path.join(BASE, d)))

for project in projects:
    versions = load_versions(project)
    if len(versions) < 3:
        continue

    # split kronologis: rilis lama = train, rilis terbaru = test
    train_df = pd.concat([d for _, d in versions[:-1]], ignore_index=True)
    test_name, test_df = versions[-1]

    common = [c for c in train_df.columns
              if c in test_df.columns and c != LABEL
              and pd.api.types.is_numeric_dtype(train_df[c])]
    X_tr, y_tr = train_df[common], train_df[LABEL]
    X_te, y_te = test_df[common], test_df[LABEL]

    if y_te.sum() == 0 or y_tr.sum() == 0:
        continue

    for name, model in get_models(y_tr).items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        rows.append({
            "project": project, "test": test_name, "model": name,
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
            "pr_auc": average_precision_score(y_te, proba),
        })

res = pd.DataFrame(rows)
os.makedirs("data/processed", exist_ok=True)
res.to_csv("data/processed/baseline_results.csv", index=False)

pd.set_option("display.width", 160)
print(res.round(3).to_string(index=False))
print("\nRata-rata per model:")
print(res.groupby("model")[["precision", "recall", "f1", "pr_auc"]]
      .mean().round(3))