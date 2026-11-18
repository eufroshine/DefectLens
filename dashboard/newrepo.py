"""Analisis repository Java baru: lizard + riwayat Git (365 hari) + PMD.

Model: Random Forest dilatih dari semua rilis ant, poi, lucene, camel
(skenario E = fitur lizard + fitur riwayat Git periode). PMD hanya ditampilkan,
tidak dipakai sebagai fitur.
"""
import glob
import os
import re
import shutil
import subprocess
import tarfile
import io
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import lizard
import numpy as np
import pandas as pd
from pydriller import Repository
from sklearn.ensemble import RandomForestClassifier

from dashboard import core

DATA = "data/processed"
NEW_REPOS = "data/raw/new_repos"
WORK = "data/raw/new_pmd_src"
PMD_BIN = os.environ.get("PMD_PATH", r"D:\Tools\pmd-bin-7.29.0\bin\pmd.bat")
RULESET = "rulesets/java/quickstart.xml"
WINDOW_DAYS = 365
FIX_RE = re.compile(r"\b(fix(e[sd])?|bugs?|defects?|faults?)\b", re.I)
SKIP_DIRS = {"test", "tests"}

LZ = ["lz_nloc", "lz_token", "lz_nfunc", "lz_ccn_sum", "lz_ccn_max",
      "lz_ccn_avg", "lz_len_max", "lz_param_max", "lz_param_avg"]
PER = core.PER
FEATURES = LZ + PER                       # skenario E

core.LABELS.update({
    "lz_nloc": "baris kode tanpa komentar",
    "lz_token": "jumlah token",
    "lz_nfunc": "jumlah fungsi",
    "lz_ccn_sum": "total kompleksitas siklomatik",
    "lz_ccn_max": "kompleksitas fungsi terkompleks",
    "lz_ccn_avg": "rata-rata kompleksitas fungsi",
    "lz_len_max": "panjang fungsi terpanjang",
    "lz_param_max": "parameter terbanyak pada satu fungsi",
    "lz_param_avg": "rata-rata parameter per fungsi",
})


def norm(p):
    return str(p).replace("\\", "/").lower()


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def prepare_repo(source, log):
    source = source.strip().strip('"')
    if os.path.isdir(source):
        if not os.path.isdir(os.path.join(source, ".git")):
            raise ValueError("Folder itu bukan repository Git "
                             "(tidak ada folder .git).")
        path = os.path.abspath(source)
        return path, re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(path))
    if not source.startswith("https://"):
        raise ValueError("Isi dengan folder repository lokal, atau URL yang "
                         "diawali https://")
    name = re.sub(r"\.git$", "", source.rstrip("/").split("/")[-1]) or "repo"
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    dest = os.path.abspath(os.path.join(NEW_REPOS, name))
    if os.path.isdir(os.path.join(dest, ".git")):
        log(f"Memakai clone yang sudah ada: {dest}")
        return dest, name
    os.makedirs(NEW_REPOS, exist_ok=True)
    log("Clone repository (bisa lama untuk repo besar)...")
    r = run(["git", "clone", source, dest])
    if r.returncode != 0:
        raise RuntimeError("git clone gagal:\n" + (r.stderr or "")[-500:])
    return dest, name


def export_java(repo, dest):
    """Ekspor file .java di HEAD (tanpa folder test) ke folder kerja."""
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    data = subprocess.run(["git", "-C", repo, "archive", "--format=tar", "HEAD"],
                          capture_output=True, check=True).stdout
    n = 0
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            if not (m.isfile() and m.name.endswith(".java")):
                continue
            parts = m.name.lower().split("/")
            if SKIP_DIRS & set(parts[:-1]) or m.name.startswith("/") \
                    or ".." in parts:
                continue
            target = os.path.join(dest, *m.name.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with tf.extractfile(m) as src, open(target, "wb") as out:
                out.write(src.read())
            n += 1
    return n


def lizard_features(src):
    rows, failed = [], 0
    base = os.path.abspath(src)
    for root, _, files in os.walk(src):
        for fn in files:
            if not fn.endswith(".java"):
                continue
            full = os.path.join(root, fn)
            rel = norm(os.path.relpath(full, base))
            try:
                with open(full, "rb") as f:
                    code = f.read().decode("utf-8", errors="ignore")
                info = lizard.analyze_file.analyze_source_code(rel, code)
            except Exception:
                failed += 1
                continue
            fs = info.function_list
            ccn = [f.cyclomatic_complexity for f in fs]
            par = [f.parameter_count for f in fs]
            rows.append({
                "key": rel, "lz_nloc": info.nloc, "lz_token": info.token_count,
                "lz_nfunc": len(fs), "lz_ccn_sum": sum(ccn),
                "lz_ccn_max": max(ccn) if ccn else 0,
                "lz_ccn_avg": sum(ccn) / len(ccn) if ccn else 0,
                "lz_len_max": max((f.nloc for f in fs), default=0),
                "lz_param_max": max(par) if par else 0,
                "lz_param_avg": sum(par) / len(par) if par else 0,
            })
    return pd.DataFrame(rows, columns=["key"] + LZ), failed


def git_period_features(repo, end):
    start = end - timedelta(days=WINDOW_DAYS)
    recs = defaultdict(lambda: {"n": 0, "add": 0, "del": 0, "auth": set(),
                                "fix": 0, "last": None})
    commits = 0
    for c in Repository(repo, since=start, to=end).traverse_commits():
        commits += 1
        is_fix = bool(FIX_RE.search(c.msg or ""))
        for m in c.modified_files:
            path = m.new_path or m.old_path
            if path and path.endswith(".java"):
                r = recs[norm(path)]
                r["n"] += 1
                r["add"] += m.added_lines
                r["del"] += m.deleted_lines
                r["auth"].add(c.author.email)
                r["fix"] += int(is_fix)
                d = c.committer_date
                if r["last"] is None or d > r["last"]:
                    r["last"] = d
    rows = [{"key": k, "p_change_freq": r["n"], "p_lines_added": r["add"],
             "p_lines_deleted": r["del"], "p_churn": r["add"] + r["del"],
             "p_n_authors": len(r["auth"]),
             "p_recency_days": (end - r["last"]).days,
             "p_fix_commits": r["fix"]} for k, r in recs.items()]
    cols = ["key"] + PER
    return pd.DataFrame(rows, columns=cols), commits


def run_pmd(src, out_csv):
    cmd = [PMD_BIN, "check", "-d", os.path.abspath(src), "-R", RULESET,
           "-f", "csv", "-r", os.path.abspath(out_csv),
           "--no-cache", "--no-fail-on-violation"]
    r = run(cmd)
    if r.returncode not in (0, 4, 5):
        raise RuntimeError("PMD gagal (kode %s):\n%s"
                           % (r.returncode, (r.stderr or r.stdout)[-500:]))
    try:
        v = pd.read_csv(out_csv)
    except Exception:
        return pd.DataFrame()
    if v.empty or "File" not in v.columns:
        return pd.DataFrame()
    base = norm(os.path.abspath(src)).rstrip("/") + "/"
    v["key"] = v["File"].map(norm).str.replace(base, "", regex=False)
    return v


def train_model():
    """Latih dari semua merged4_*.csv (4 proyek, semua rilis)."""
    dfs = [pd.read_csv(f) for f in glob.glob(f"{DATA}/merged4_*.csv")]
    if not dfs:
        raise FileNotFoundError("Tidak ada data/processed/merged4_*.csv. "
                                "Jalankan src\\extract\\build_standalone.py.")
    d = pd.concat(dfs, ignore_index=True)
    y = (d["bug"] > 0).astype(int)
    X = np.log1p(d[FEATURES].fillna(0))
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, len(d)


def analyze_repo(source, model, log=lambda m: None):
    repo, name = prepare_repo(source, log)

    head = run(["git", "-C", repo, "log", "-1", "--format=%ct", "HEAD"])
    if head.returncode != 0 or not head.stdout.strip().isdigit():
        raise RuntimeError("Tidak bisa membaca commit terakhir repository.")
    end = datetime.fromtimestamp(int(head.stdout.strip()), tz=timezone.utc)

    src = os.path.join(WORK, name)
    log("Mengekspor file .java (tanpa folder test)...")
    n_java = export_java(repo, src)
    if n_java == 0:
        raise ValueError("Tidak ada file .java di repository ini.")

    log(f"Menghitung ukuran dan kompleksitas ({n_java} file)...")
    lz, failed = lizard_features(src)

    log(f"Membaca riwayat Git {WINDOW_DAYS} hari terakhir...")
    gp, commits = git_period_features(repo, end)

    log("Menjalankan PMD...")
    pmd_csv = f"{DATA}/pmd_new_{name}.csv"
    os.makedirs(DATA, exist_ok=True)
    pmd = run_pmd(src, pmd_csv)

    df = lz.merge(gp, on="key", how="left")
    changed = df["p_change_freq"].notna()
    for c in PER:
        df[c] = df[c].fillna(WINDOW_DAYS if c == "p_recency_days" else 0)

    X_raw = df[FEATURES].copy()
    X = np.log1p(X_raw)
    proba = model.predict_proba(X)[:, 1]

    import shap
    sv = core.positive_class(shap.TreeExplainer(model).shap_values(X))

    if len(pmd):
        agg = pmd.groupby("key").agg(
            pmd_total=("Priority", "size"),
            pmd_high=("Priority", lambda s: int((s <= 2).sum()))).reset_index()
        df = df.merge(agg, on="key", how="left")
    else:
        df["pmd_total"], df["pmd_high"] = 0, 0
    df[["pmd_total", "pmd_high"]] = df[["pmd_total", "pmd_high"]].fillna(0).astype(int)

    reasons = []
    for i in range(len(df)):
        idx = np.argsort(-np.abs(sv[i]))[:core.TOP_REASONS]
        reasons.append("; ".join(
            f"{core.label(FEATURES[j])} = {X_raw[FEATURES[j]].iloc[i]:g} "
            f"({'naik' if sv[i, j] > 0 else 'turun'})" for j in idx))

    table = pd.DataFrame({
        "idx": np.arange(len(df)), "file": df["key"].values,
        "risiko_ml": proba,
        "persentil": pd.Series(proba).rank(pct=True).values,
        "pmd_total": df["pmd_total"].values,
        "pmd_high": df["pmd_high"].values,
        "alasan": reasons,
    }).sort_values("risiko_ml", ascending=False).reset_index(drop=True)

    warnings = []
    if commits < 30:
        warnings.append(f"Hanya {commits} commit dalam {WINDOW_DAYS} hari "
                        "terakhir. Fitur riwayat Git hampir kosong, jadi skor "
                        "kurang bisa dipercaya.")
    if changed.mean() < 0.2:
        warnings.append(f"Hanya {changed.mean():.0%} file yang berubah dalam "
                        f"{WINDOW_DAYS} hari terakhir.")
    if len(df) < 30:
        warnings.append("Jumlah file sedikit, peringkat kurang bermakna.")
    if failed:
        warnings.append(f"{failed} file gagal dianalisis lizard dan dilewati.")

    return {"name": name, "anchor": end, "n_files": len(df), "commits": commits,
            "table": table, "sv": sv, "X_raw": X_raw, "pmd": pmd,
            "warnings": warnings}


def contributions(res, idx, top=10):
    sv, X = res["sv"], res["X_raw"]
    order = np.argsort(-np.abs(sv[idx]))[:top]
    return pd.DataFrame({
        "fitur": [core.label(FEATURES[j]) for j in order],
        "nilai_asli": [X[FEATURES[j]].iloc[idx] for j in order],
        "kontribusi": [float(sv[idx, j]) for j in order],
    })


def pmd_for_file(res, key):
    v = res["pmd"]
    if v is None or len(v) == 0:
        return pd.DataFrame()
    sel = v[v["key"] == key]
    cols = [c for c in ["Priority", "Line", "Rule", "Description"]
            if c in sel.columns]
    sel = sel[cols]
    sort_cols = [c for c in ["Priority", "Line"] if c in sel.columns]
    return sel.sort_values(sort_cols) if sort_cols else sel