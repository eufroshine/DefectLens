import glob
import json
import os
import re

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CUM = ["change_freq", "lines_added", "lines_deleted", "churn", "n_authors"]
PER = ["p_change_freq", "p_lines_added", "p_lines_deleted", "p_churn",
       "p_n_authors", "p_recency_days", "p_fix_commits"]
NON_FEATURE = {"name", "file", "bug", "version", "project"} | set(CUM) | set(PER)
TOP_N = 3


def vkey(v):
    return tuple(int(x) for x in re.findall(r"\d+", v))


def load_all():
    dfs = []
    for f in glob.glob("data/processed/merged2_*.csv"):
        m = re.search(r"merged2_(.+?)-(.+)\.csv", os.path.basename(f))
        d = pd.read_csv(f)
        d["project"], d["version"] = m.group(1), m.group(2)
        d["bug"] = (d["bug"] > 0).astype(int)
        dfs.append(d)
    return dfs


def positive_class(sv):
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


dfs = load_all()
first = dfs[0]
metric_cols = [c for c in first.columns if c not in NON_FEATURE
               and pd.api.types.is_numeric_dtype(first[c])]
for d in dfs:
    metric_cols = [c for c in metric_cols if c in d.columns]
FEATURES = metric_cols + PER
print(f"Jumlah fitur: {len(FEATURES)} ({len(metric_cols)} metrik kode + "
      f"{len(PER)} fitur Git periode)")

all_df = pd.concat(dfs, ignore_index=True)
all_df[PER] = all_df[PER].fillna(0)
raw = all_df[FEATURES].copy()                       # nilai asli untuk teks
X = all_df[FEATURES].copy()
X[PER] = np.log1p(X[PER])                           # sama seperti rolling_eval2

# rilis terakhir tiap proyek = data penjelasan
last = all_df.groupby("project")["version"].transform(
    lambda s: max(s.unique(), key=vkey))
is_test = (all_df["version"] == last).values
Xtr, ytr = X[~is_test], all_df.loc[~is_test, "bug"]
Xte, yte = X[is_test], all_df.loc[is_test, "bug"]
raw_te = raw[is_test]
meta_te = all_df.loc[is_test, ["project", "version", "name", "file"]]

print(f"Latih: {len(Xtr)} | Penjelasan: {len(Xte)} | "
      f"proporsi bug di data penjelasan: {yte.mean():.3f}")

model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=42, n_jobs=-1)
model.fit(Xtr, ytr)
proba = model.predict_proba(Xte)[:, 1]
print(f"PR-AUC (sanity check, bukan hasil utama): "
      f"{average_precision_score(yte, proba):.3f}")

sv = positive_class(shap.TreeExplainer(model).shap_values(Xte))

# ---- 1. Pentingnya fitur ----
imp = pd.DataFrame({
    "fitur": FEATURES,
    "mean_abs_shap": np.abs(sv).mean(axis=0),
    "arah_korelasi": [
        np.corrcoef(Xte[c], sv[:, i])[0, 1] if Xte[c].std() > 0 else np.nan
        for i, c in enumerate(FEATURES)],
}).sort_values("mean_abs_shap", ascending=False)
imp["grup"] = imp["fitur"].apply(
    lambda f: "Git periode" if f in PER else "Metrik kode")
imp.to_csv("data/processed/shap_importance.csv", index=False)
print("\n=== 15 fitur paling berpengaruh ===")
print(imp.head(15).round(3).to_string(index=False))
print("\nKontribusi per grup (jumlah mean |SHAP|):")
print(imp.groupby("grup")["mean_abs_shap"].sum().round(3))

top = imp.head(15).iloc[::-1]
plt.figure(figsize=(8, 6))
colors = ["tab:orange" if g == "Git periode" else "tab:blue" for g in top["grup"]]
plt.barh(top["fitur"], top["mean_abs_shap"], color=colors)
plt.xlabel("Rata-rata |SHAP| (pengaruh pada prediksi)")
plt.title("Fitur paling berpengaruh (oranye = riwayat Git)")
plt.tight_layout()
plt.savefig("data/processed/shap_importance.png", dpi=150)
plt.close()

# ---- 2. Penjelasan per file ----
rows = []
for i in range(len(Xte)):
    idx = np.argsort(-np.abs(sv[i]))[:TOP_N]
    reasons = "; ".join(
        f"{FEATURES[j]}={raw_te.iloc[i, j]:g} "
        f"({'menaikkan' if sv[i, j] > 0 else 'menurunkan'} risiko)"
        for j in idx)
    rows.append({**meta_te.iloc[i].to_dict(), "risk": round(proba[i], 3),
                 "actual_bug": int(yte.iloc[i]), "alasan": reasons})
expl = pd.DataFrame(rows).sort_values("risk", ascending=False)
expl.to_csv("data/processed/shap_explanations.csv", index=False)
print("\n=== Contoh: 5 file berisiko tertinggi ===")
for _, r in expl.head(5).iterrows():
    print(f"{r['project']}-{r['version']} {r['name']}  risiko={r['risk']}  "
          f"(bug asli={r['actual_bug']})\n   {r['alasan']}")

# ---- 3. Model final (semua rilis) ----
final = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=42, n_jobs=-1)
final.fit(X, all_df["bug"])
os.makedirs("models", exist_ok=True)
joblib.dump(final, "models/rf_scenarioC.joblib")
with open("models/features.json", "w") as f:
    json.dump({"features": FEATURES, "log1p_features": PER}, f, indent=2)
print("\nModel disimpan: models/rf_scenarioC.joblib dan models/features.json")