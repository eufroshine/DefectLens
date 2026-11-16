import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier

CUM = ["change_freq", "lines_added", "lines_deleted", "churn", "n_authors"]
PER = ["p_change_freq", "p_lines_added", "p_lines_deleted", "p_churn",
       "p_n_authors", "p_recency_days", "p_fix_commits"]
PMD = ["pmd_total", "pmd_high"]
NON_FEATURE = {"name", "file", "bug", "version", "project"} \
    | set(CUM) | set(PER) | set(PMD)
SEEDS = [0, 1, 2, 3, 4]
FRACS = [0.1, 0.2]
rng = np.random.RandomState(0)


def vkey(v):
    return tuple(int(x) for x in re.findall(r"\d+", v))


def load(p):
    dfs = []
    for f in glob.glob(f"data/processed/merged3_{p}-*.csv"):
        v = re.search(rf"{p}-(.+)\.csv", os.path.basename(f)).group(1)
        d = pd.read_csv(f)
        d["version"] = v
        d["bug"] = (d["bug"] > 0).astype(int)
        d[CUM + PER + PMD] = np.log1p(d[CUM + PER + PMD].fillna(0))
        dfs.append(d)
    dfs.sort(key=lambda d: vkey(d["version"].iloc[0]))
    return dfs


def recall_at(score, y, frac):
    score = np.asarray(score, dtype=float) + rng.rand(len(y)) * 1e-9  # pecah seri
    n = max(1, int(round(len(y) * frac)))
    top = np.argsort(-score)[:n]
    return y[top].sum() / y.sum()


def pct_rank(x):
    return pd.Series(x).rank(pct=True).values


projects = sorted({os.path.basename(f).split("_")[1].split("-")[0]
                   for f in glob.glob("data/processed/merged3_*.csv")})
rows = []
for p in projects:
    dfs = load(p)
    for k in range(1, len(dfs)):
        train = pd.concat(dfs[:k], ignore_index=True)
        test = dfs[k]
        y = test["bug"].values
        if train["bug"].sum() == 0 or y.sum() == 0:
            continue
        pcols = [c for c in train.columns if c not in NON_FEATURE
                 and pd.api.types.is_numeric_dtype(train[c])
                 and c in test.columns]
        feats = pcols + PER
        proba = np.zeros(len(test))
        for s in SEEDS:
            m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=s)
            m.fit(train[feats], train["bug"])
            proba += m.predict_proba(test[feats])[:, 1] / len(SEEDS)

        scores = {
            "ML (skenario C)": proba,
            "PMD saja": test["pmd_total"].values,
            "Ukuran saja (loc)": test["loc"].values,
            "Gabungan ML+PMD": 0.5 * pct_rank(proba)
                               + 0.5 * pct_rank(test["pmd_total"].values),
        }
        for name, sc in scores.items():
            for fr in FRACS:
                rows.append({"project": p, "test_version": test["version"].iloc[0],
                             "metode": name, "top": fr,
                             "recall": recall_at(sc, y, fr),
                             "base_bug": y.mean()})

res = pd.DataFrame(rows)
res.to_csv("data/processed/ablation_priority.csv", index=False)
print(f"Pasangan latih-uji: {res.groupby(['project', 'test_version']).ngroups}")
print("(Acak murni = recall sama dengan persentase file yang diperiksa)\n")

for fr in FRACS:
    print(f"=== Recall jika hanya memeriksa {int(fr * 100)}% file teratas ===")
    r = res[res["top"] == fr]
    print(r.groupby("metode")["recall"].mean().round(3).sort_values(ascending=False))
    piv = r.pivot_table(index=["project", "test_version"], columns="metode",
                        values="recall")
    for a, b in [("Gabungan ML+PMD", "ML (skenario C)"),
                 ("ML (skenario C)", "PMD saja"),
                 ("ML (skenario C)", "Ukuran saja (loc)")]:
        d = piv[a] - piv[b]
        line = (f"  {a} - {b}: rata-rata={d.mean():+.3f} "
                f"menang={int((d > 0).sum())}/{len(d)}")
        try:
            line += f" p={wilcoxon(d).pvalue:.3f}"
        except ValueError:
            pass
        print(line)
    print()

print("Per proyek (top 20%):")
print(res[res["top"] == 0.2].pivot_table(index="project", columns="metode",
                                         values="recall").round(3))