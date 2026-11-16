"""Logika inti DefectLens (tanpa Streamlit, supaya mudah diuji)."""
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA = "data/processed"
CUM = ["change_freq", "lines_added", "lines_deleted", "churn", "n_authors"]
PER = ["p_change_freq", "p_lines_added", "p_lines_deleted", "p_churn",
       "p_n_authors", "p_recency_days", "p_fix_commits"]
PMD = ["pmd_total", "pmd_high"]
NON_FEATURE = {"name", "file", "bug", "version", "project"} \
    | set(CUM) | set(PER) | set(PMD)
TOP_REASONS = 3

# Nama fitur yang mudah dibaca
LABELS = {
    "p_lines_added": "baris ditambah (1 tahun terakhir)",
    "p_lines_deleted": "baris dihapus (1 tahun terakhir)",
    "p_churn": "total perubahan baris (1 tahun terakhir)",
    "p_change_freq": "jumlah perubahan (1 tahun terakhir)",
    "p_n_authors": "jumlah penulis (1 tahun terakhir)",
    "p_recency_days": "hari sejak perubahan terakhir",
    "p_fix_commits": "jumlah commit perbaikan (1 tahun terakhir)",
    "wmc": "kompleksitas kelas (wmc)",
    "loc": "jumlah baris kode (loc)",
    "npm": "jumlah method publik (npm)",
    "rfc": "response for class (rfc)",
    "cbo": "keterkaitan antar kelas (cbo)",
}


def label(feature):
    return LABELS.get(feature, feature)


def vkey(v):
    return tuple(int(x) for x in re.findall(r"\d+", v))


def list_versions():
    """{proyek: [versi terurut]} dari file merged3_*.csv."""
    out = {}
    for f in glob.glob(f"{DATA}/merged3_*.csv"):
        m = re.search(r"merged3_(.+?)-(.+)\.csv", os.path.basename(f))
        if m:
            out.setdefault(m.group(1), []).append(m.group(2))
    return {p: sorted(v, key=vkey) for p, v in sorted(out.items())}


def load_version(project, version):
    d = pd.read_csv(f"{DATA}/merged3_{project}-{version}.csv")
    d["bug"] = (d["bug"] > 0).astype(int)
    return d


def prep(d):
    """log1p pada fitur riwayat Git dan PMD (sama seperti saat evaluasi)."""
    x = d.copy()
    cols = [c for c in CUM + PER + PMD if c in x.columns]
    x[cols] = np.log1p(x[cols].fillna(0))
    return x


def positive_class(sv):
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def analyze(project, version):
    """Latih model HANYA dari rilis sebelumnya, lalu nilai rilis ini.

    Mengembalikan None bila tidak ada data latih yang layak.
    """
    versions = list_versions()[project]
    idx = versions.index(version)
    if idx == 0:
        return None
    train_raw = pd.concat([load_version(project, v) for v in versions[:idx]],
                          ignore_index=True)
    test_raw = load_version(project, version)
    if train_raw["bug"].nunique() < 2:
        return None

    train, test = prep(train_raw), prep(test_raw)
    pcols = [c for c in train.columns
             if c not in NON_FEATURE
             and pd.api.types.is_numeric_dtype(train[c]) and c in test.columns]
    feats = pcols + PER

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(train[feats], train["bug"])
    proba = model.predict_proba(test[feats])[:, 1]

    import shap  # diimpor di sini agar pesan error jelas bila belum terpasang
    sv = positive_class(shap.TreeExplainer(model).shap_values(test[feats]))
    return {"feats": feats, "proba": proba, "sv": sv, "raw": test_raw,
            "n_train": len(train)}


def pct_rank(x):
    return pd.Series(np.asarray(x, dtype=float)).rank(pct=True).values


def build_table(res):
    raw, sv, feats, proba = res["raw"], res["sv"], res["feats"], res["proba"]
    reasons = []
    for i in range(len(raw)):
        idx = np.argsort(-np.abs(sv[i]))[:TOP_REASONS]
        reasons.append("; ".join(
            f"{label(feats[j])} = {raw[feats[j]].iloc[i]:g} "
            f"({'naik' if sv[i, j] > 0 else 'turun'})" for j in idx))
    pmd_total = raw["pmd_total"].fillna(0).astype(int).values
    table = pd.DataFrame({
        "kelas": raw["name"].values,
        "file": raw["file"].values,
        "risiko_ml": proba,
        "pmd_total": pmd_total,
        "pmd_high": raw["pmd_high"].fillna(0).astype(int).values,
        "skor_gabungan": 0.5 * pct_rank(proba) + 0.5 * pct_rank(pmd_total),
        "alasan": reasons,
        "bug_asli": raw["bug"].values,
    })
    return table


def contributions(res, i, top=10):
    """Kontribusi SHAP satu file, untuk grafik batang."""
    feats, sv, raw = res["feats"], res["sv"], res["raw"]
    idx = np.argsort(-np.abs(sv[i]))[:top]
    return pd.DataFrame({
        "fitur": [label(feats[j]) for j in idx],
        "nilai_asli": [raw[feats[j]].iloc[i] for j in idx],
        "kontribusi": [float(sv[i, j]) for j in idx],
    })


def pmd_findings(project, version, file_path):
    """Temuan PMD untuk satu file (dari pmd_raw_*.csv hasil run_pmd.py)."""
    p = f"{DATA}/pmd_raw_{project}-{version}.csv"
    if not os.path.exists(p):
        return None
    try:
        v = pd.read_csv(p)
    except Exception:
        return None
    if v.empty or "File" not in v.columns:
        return v.iloc[0:0]
    base = os.path.abspath(f"data/raw/pmd_src/{project}-{version}")
    base = base.replace("\\", "/").lower().rstrip("/") + "/"
    key = v["File"].map(lambda x: str(x).replace("\\", "/").lower())
    key = key.str.replace(base, "", regex=False)
    target = str(file_path).replace("\\", "/").lower()
    sel = v[key == target]
    cols = [c for c in ["Priority", "Line", "Rule", "Description"]
            if c in sel.columns]
    sel = sel[cols]
    sort_cols = [c for c in ["Priority", "Line"] if c in sel.columns]
    return sel.sort_values(sort_cols) if sort_cols else sel


def recall_at(score, y, frac):
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    n = max(1, int(round(len(y) * frac)))
    top = np.argsort(-np.asarray(score, dtype=float), kind="stable")[:n]
    return float(y[top].sum() / y.sum())