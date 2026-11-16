import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CUM = ["change_freq", "lines_added", "lines_deleted", "churn", "n_authors"]
PER = ["p_change_freq", "p_lines_added", "p_lines_deleted", "p_churn",
       "p_n_authors", "p_recency_days", "p_fix_commits"]
PMD = ["pmd_total", "pmd_high"]
NON_FEATURE = {"name", "file", "bug", "version", "project"} \
    | set(CUM) | set(PER) | set(PMD)
SEEDS = [0, 1, 2, 3, 4]


def vkey(v):
    return tuple(int(x) for x in re.findall(r"\d+", v))


def load_raw(p):
    dfs = []
    for f in glob.glob(f"data/processed/merged3_{p}-*.csv"):
        v = re.search(rf"{p}-(.+)\.csv", os.path.basename(f)).group(1)
        d = pd.read_csv(f)
        d["version"], d["project"] = v, p
        d["bug"] = (d["bug"] > 0).astype(int)
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
                   for f in glob.glob("data/processed/merged3_*.csv")})
raw = {p: load_raw(p) for p in projects}

# ---------- Bagian 1: hubungan PMD dengan bug ----------
allr = pd.concat([d for p in projects for d in raw[p]], ignore_index=True)
allr["pmd_per_kloc"] = allr["pmd_total"] / (allr["loc"].clip(lower=1) / 1000)
print("=== Hubungan temuan PMD dengan bug (semua data digabung) ===")
rho_b = spearmanr(allr["pmd_total"], allr["bug"])[0]
rho_l = spearmanr(allr["pmd_total"], allr["loc"])[0]
rho_lb = spearmanr(allr["loc"], allr["bug"])[0]
print(f"Spearman pmd_total ~ bug : {rho_b:.3f}")
print(f"Spearman loc       ~ bug : {rho_lb:.3f}")
print(f"Spearman pmd_total ~ loc : {rho_l:.3f}   (tinggi = PMD banyak mengukur ukuran)")
print("\nROC-AUC satu fitur untuk memisahkan bug (0,5 = acak):")
for c in ["pmd_total", "pmd_high", "pmd_per_kloc", "loc"]:
    print(f"  {c:14s}: {roc_auc_score(allr['bug'], allr[c]):.3f}")
print("\nSpearman pmd_total ~ bug per proyek:")
for p in projects:
    d = pd.concat(raw[p])
    print(f"  {p:8s}: {spearmanr(d['pmd_total'], d['bug'])[0]:.3f}")

# ---------- Bagian 2: skenario A, C, D ----------
rows = []
for p in projects:
    dfs = []
    for d in raw[p]:
        d = d.copy()
        cols = CUM + PER + PMD
        d[cols] = np.log1p(d[cols].fillna(0))
        dfs.append(d)
    for k in range(1, len(dfs)):
        train = pd.concat(dfs[:k], ignore_index=True)
        test = dfs[k]
        if train["bug"].sum() == 0 or test["bug"].sum() == 0:
            continue
        pcols = [c for c in train.columns if c not in NON_FEATURE
                 and pd.api.types.is_numeric_dtype(train[c])
                 and c in test.columns]
        sets = {"A": pcols, "C": pcols + PER, "D": pcols + PER + PMD}
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
res.to_csv("data/processed/rolling_eval3.csv", index=False)
piv = res.pivot_table(index=["project", "test_version", "model"],
                      columns="fitur", values="pr_auc").reset_index()
piv["C-A"], piv["D-A"], piv["D-C"] = piv["C"] - piv["A"], piv["D"] - piv["A"], \
    piv["D"] - piv["C"]

print(f"\nPasangan latih-uji: {piv.groupby(['project', 'test_version']).ngroups}")
print("\nRata-rata PR-AUC:")
print(res.groupby(["model", "fitur"])["pr_auc"].mean().unstack().round(3))


def report(col, title):
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


report("C-A", "C - A : Git periode vs PROMISE saja")
report("D-A", "D - A : Git periode + PMD vs PROMISE saja")
report("D-C", "D - C : tambahan PMD di atas Git periode")
print("\nSelisih rata-rata per proyek:")
print(piv.groupby("project")[["C-A", "D-A", "D-C"]].mean().round(3))