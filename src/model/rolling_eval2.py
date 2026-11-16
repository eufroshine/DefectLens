import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CUM = ["change_freq", "lines_added", "lines_deleted", "churn", "n_authors"]
PER = ["p_change_freq", "p_lines_added", "p_lines_deleted", "p_churn",
       "p_n_authors", "p_recency_days", "p_fix_commits"]
NON_FEATURE = {"name", "file", "bug", "version", "project"} | set(CUM) | set(PER)
SEEDS = [0, 1, 2, 3, 4]


def vkey(v):
    return tuple(int(x) for x in re.findall(r"\d+", v))


def load_project(p):
    dfs = []
    for f in glob.glob(f"data/processed/merged2_{p}-*.csv"):
        v = re.search(rf"{p}-(.+)\.csv", os.path.basename(f)).group(1)
        d = pd.read_csv(f)
        d["version"] = v
        d["bug"] = (d["bug"] > 0).astype(int)
        d[CUM + PER] = np.log1p(d[CUM + PER].fillna(0))   # bantu Logistic Reg.
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
                   for f in glob.glob("data/processed/merged2_*.csv")})
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
        sets = {"A": pcols, "B": pcols + CUM, "C": pcols + PER}
        for m in ["RandomForest", "XGBoost", "LogisticRegression"]:
            for fs, cols in sets.items():
                pr = []
                for s in SEEDS:
                    mdl = make_model(m, train["bug"], s)
                    mdl.fit(train[cols], train["bug"])
                    proba = mdl.predict_proba(test[cols])[:, 1]
                    pr.append(average_precision_score(test["bug"], proba))
                    if m == "LogisticRegression":
                        break
                rows.append({"project": p, "test_version": test["version"].iloc[0],
                             "model": m, "fitur": fs, "pr_auc": np.mean(pr)})

res = pd.DataFrame(rows)
res.to_csv("data/processed/rolling_eval2.csv", index=False)

piv = res.pivot_table(index=["project", "test_version", "model"],
                      columns="fitur", values="pr_auc").reset_index()
piv["B-A"] = piv["B"] - piv["A"]
piv["C-A"] = piv["C"] - piv["A"]
piv["C-B"] = piv["C"] - piv["B"]

print(f"Pasangan latih-uji: {piv.groupby(['project', 'test_version']).ngroups}")
print("\nRata-rata PR-AUC per model dan skenario:")
print(res.groupby(["model", "fitur"])["pr_auc"].mean().unstack().round(3))


def report(col, title):
    print(f"\n=== {title} ===")
    for m, g in piv.groupby("model"):
        x = g[col]
        line = (f"{m:20s} rata-rata={x.mean():+.3f}  "
                f"menang={int((x > 0).sum())}/{len(x)}")
        try:
            line += f"  p={wilcoxon(x).pvalue:.3f}"
        except ValueError:
            pass
        print(line)
    x = piv[col]
    line = f"{'SEMUA':20s} rata-rata={x.mean():+.3f}  " \
           f"menang={int((x > 0).sum())}/{len(x)}"
    try:
        line += f"  p={wilcoxon(x).pvalue:.3f}"
    except ValueError:
        pass
    print(line)


report("B-A", "B - A : Git kumulatif vs PROMISE saja")
report("C-A", "C - A : Git periode vs PROMISE saja")
report("C-B", "C - B : Git periode vs Git kumulatif")

print("\nSelisih rata-rata PR-AUC per proyek:")
print(piv.groupby("project")[["B-A", "C-A", "C-B"]].mean().round(3))