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
PMD = ["pmd_total", "pmd_high"]
LZ = ["lz_nloc", "lz_token", "lz_nfunc", "lz_ccn_sum", "lz_ccn_max",
      "lz_ccn_avg", "lz_len_max", "lz_param_max", "lz_param_avg"]
NON_FEATURE = {"name", "file", "bug", "version", "project"} \
    | set(CUM) | set(PER) | set(PMD) | set(LZ)
SEEDS = [0, 1, 2, 3, 4]


def vkey(v):
    return tuple(int(x) for x in re.findall(r"\d+", v))


def load(p):
    dfs = []
    for f in glob.glob(f"data/processed/merged4_{p}-*.csv"):
        v = re.search(rf"{p}-(.+)\.csv", os.path.basename(f)).group(1)
        d = pd.read_csv(f)
        d["version"] = v
        d["bug"] = (d["bug"] > 0).astype(int)
        cols = CUM + PER + PMD + LZ
        d[cols] = np.log1p(d[cols].fillna(0))
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
                   for f in glob.glob("data/processed/merged4_*.csv")})
rows = []
for p in projects:
    dfs = load(p)
    for k in range(1, len(dfs)):
        train = pd.concat(dfs[:k], ignore_index=True)
        test = dfs[k]
        if train["bug"].sum() == 0 or test["bug"].sum() == 0:
            continue
        pcols = [c for c in train.columns if c not in NON_FEATURE
                 and pd.api.types.is_numeric_dtype(train[c])
                 and c in test.columns]
        sets = {"C": pcols + PER,            # skenario lama (metrik PROMISE)
                "E": LZ + PER,               # mandiri: lizard + Git periode
                "L": LZ}                     # mandiri: lizard saja
        for m in ["RandomForest", "XGBoost", "LogisticRegression"]:
            for fs, cols in sets.items():
                pr = []
                for s in SEEDS:
                    mdl = make_model(m, train["bug"], s)
                    mdl.fit(train[cols], train["bug"])
                    pr.append(average_precision_score(
                        test["bug"], mdl.predict_proba(test[cols])[:, 1]))
                    if m == "LogisticRegression":
                        break
                rows.append({"project": p, "test_version": test["version"].iloc[0],
                             "model": m, "fitur": fs, "pr_auc": np.mean(pr)})

res = pd.DataFrame(rows)
res.to_csv("data/processed/rolling_eval4.csv", index=False)
piv = res.pivot_table(index=["project", "test_version", "model"],
                      columns="fitur", values="pr_auc").reset_index()
piv["E-C"], piv["E-L"] = piv["E"] - piv["C"], piv["E"] - piv["L"]

print(f"Pasangan latih-uji: {piv.groupby(['project', 'test_version']).ngroups}")
print("\nRata-rata PR-AUC:")
print(res.groupby(["model", "fitur"])["pr_auc"].mean().unstack().round(3))
for col, title in [("E-C", "E - C : fitur mandiri vs metrik PROMISE"),
                   ("E-L", "E - L : tambahan Git periode di atas lizard")]:
    print(f"\n=== {title} ===")
    for m, g in list(piv.groupby("model")) + [("SEMUA", piv)]:
        x = g[col]
        line = (f"{m:20s} rata-rata={x.mean():+.3f}  "
                f"menang={int((x > 0).sum())}/{len(x)}")
        try:
            line += f"  p={wilcoxon(x).pvalue:.3f}"
        except ValueError:
            pass
        print(line)