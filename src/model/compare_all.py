import glob
import os
import re

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

GIT_COLS = ["change_freq", "lines_added", "lines_deleted", "churn", "n_authors"]
NON_FEATURE = {"name", "file", "bug", "version", "project"} | set(GIT_COLS)


def load_project(project):
    dfs = []
    for f in glob.glob(f"data/processed/merged_{project}-*.csv"):
        v = re.search(rf"{project}-(.+)\.csv", os.path.basename(f)).group(1)
        d = pd.read_csv(f)
        d["version"] = v
        d["project"] = project
        dfs.append(d)
    return dfs


def vkey(v):
    return tuple(int(x) for x in re.findall(r"\d+", v))


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


projects = sorted({os.path.basename(f).split("_")[1].split("-")[0]
                   for f in glob.glob("data/processed/merged_*.csv")})
rows = []
for p in projects:
    dfs = load_project(p)
    dfs.sort(key=lambda d: vkey(d["version"].iloc[0]))
    if len(dfs) < 2:
        print(f"{p}: versi kurang dari 2, dilewati")
        continue

    train = pd.concat(dfs[:-1], ignore_index=True)
    test = dfs[-1].copy()
    for d in (train, test):
        d["bug"] = (d["bug"] > 0).astype(int)
        d[GIT_COLS] = d[GIT_COLS].fillna(0)

    promise_cols = [c for c in train.columns
                    if c not in NON_FEATURE
                    and pd.api.types.is_numeric_dtype(train[c])
                    and c in test.columns]
    if train["bug"].sum() == 0 or test["bug"].sum() == 0:
        print(f"{p}: tidak ada kelas bug, dilewati")
        continue

    base = test["bug"].mean()
    print(f"{p}: latih {len(train)}, uji {len(test)} "
          f"({dfs[-1]['version'].iloc[0]}), proporsi bug uji {base:.3f}")

    for fs, cols in {"A": promise_cols, "B": promise_cols + GIT_COLS}.items():
        for m, model in models(train["bug"]).items():
            model.fit(train[cols], train["bug"])
            pred = model.predict(test[cols])
            proba = model.predict_proba(test[cols])[:, 1]
            rows.append({
                "project": p, "fitur": fs, "model": m, "bug_uji": round(base, 3),
                "precision": precision_score(test["bug"], pred, zero_division=0),
                "recall": recall_score(test["bug"], pred, zero_division=0),
                "f1": f1_score(test["bug"], pred, zero_division=0),
                "pr_auc": average_precision_score(test["bug"], proba),
            })

res = pd.DataFrame(rows)
res.to_csv("data/processed/compare_all.csv", index=False)
pd.set_option("display.width", 160)
print(res.round(3).to_string(index=False))

print("\n=== Rata-rata semua proyek (A = PROMISE saja, B = + Git) ===")
print(res.groupby(["model", "fitur"])[["precision", "recall", "f1", "pr_auc"]]
      .mean().round(3))

piv = res.pivot_table(index=["project", "model"], columns="fitur",
                      values="pr_auc")
print(f"\nB lebih tinggi dari A (PR-AUC) pada {(piv['B'] > piv['A']).sum()} "
      f"dari {len(piv)} kombinasi proyek-model")