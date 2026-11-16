import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

GIT_COLS = ["change_freq", "lines_added", "lines_deleted", "churn", "n_authors"]
NON_FEATURE = {"name", "file", "bug", "version", "project"} | set(GIT_COLS)
SEEDS = [0, 1, 2, 3, 4]


def vkey(v):
    return tuple(int(x) for x in re.findall(r"\d+", v))


def load_project(p):
    dfs = []
    for f in glob.glob(f"data/processed/merged_{p}-*.csv"):
        v = re.search(rf"{p}-(.+)\.csv", os.path.basename(f)).group(1)
        d = pd.read_csv(f)
        d["version"] = v
        d["bug"] = (d["bug"] > 0).astype(int)
        d[GIT_COLS] = d[GIT_COLS].fillna(0)
        dfs.append(d)
    dfs.sort(key=lambda d: vkey(d["version"].iloc[0]))
    return dfs


def make_model(name, y, seed):
    pos, neg = y.sum(), (y == 0).sum()
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=seed)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                             scale_pos_weight=neg / max(pos, 1),
                             eval_metric="logloss", random_state=seed)
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced",
                                            max_iter=2000))


projects = sorted({os.path.basename(f).split("_")[1].split("-")[0]
                   for f in glob.glob("data/processed/merged_*.csv")})
rows = []
for p in projects:
    dfs = load_project(p)
    for k in range(1, len(dfs)):
        train = pd.concat(dfs[:k], ignore_index=True)
        test = dfs[k]
        if train["bug"].sum() == 0 or test["bug"].sum() == 0:
            continue
        pcols = [c for c in train.columns if c not in NON_FEATURE
                 and pd.api.types.is_numeric_dtype(train[c])
                 and c in test.columns]
        for m in ["RandomForest", "XGBoost", "LogisticRegression"]:
            for fs, cols in {"A": pcols, "B": pcols + GIT_COLS}.items():
                pr, f1 = [], []
                for s in SEEDS:
                    mdl = make_model(m, train["bug"], s)
                    mdl.fit(train[cols], train["bug"])
                    proba = mdl.predict_proba(test[cols])[:, 1]
                    pr.append(average_precision_score(test["bug"], proba))
                    f1.append(f1_score(test["bug"], proba >= 0.5,
                                       zero_division=0))
                    if m == "LogisticRegression":
                        break                      # deterministik
                rows.append({"project": p, "test_version": test["version"].iloc[0],
                             "model": m, "fitur": fs, "base": test["bug"].mean(),
                             "pr_auc": np.mean(pr), "f1": np.mean(f1)})

res = pd.DataFrame(rows)
res.to_csv("data/processed/rolling_eval.csv", index=False)

piv = res.pivot_table(index=["project", "test_version", "model"],
                      columns="fitur", values=["pr_auc", "f1"])
d = pd.DataFrame({
    "d_pr_auc": piv[("pr_auc", "B")] - piv[("pr_auc", "A")],
    "d_f1": piv[("f1", "B")] - piv[("f1", "A")],
}).reset_index()

print(f"Jumlah pasangan latih-uji: {d.groupby(['project','test_version']).ngroups}")
print("\n=== Selisih B - A (positif = fitur Git membantu) ===")
for m, g in d.groupby("model"):
    out = f"{m:20s} n={len(g)}  rata-rata dPR-AUC={g['d_pr_auc'].mean():+.3f}  " \
          f"menang={int((g['d_pr_auc'] > 0).sum())}/{len(g)}"
    try:
        out += f"  Wilcoxon p={wilcoxon(g['d_pr_auc']).pvalue:.3f}"
    except ValueError:
        pass
    print(out)

try:
    p_all = wilcoxon(d["d_pr_auc"]).pvalue
    print(f"\nSemua model digabung: rata-rata dPR-AUC={d['d_pr_auc'].mean():+.3f}, "
          f"menang={int((d['d_pr_auc'] > 0).sum())}/{len(d)}, Wilcoxon p={p_all:.3f}")
except ValueError:
    pass
print("\nPer proyek:")
print(d.groupby("project")[["d_pr_auc", "d_f1"]].mean().round(3))