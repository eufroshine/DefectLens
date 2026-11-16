import io
import os
import shutil
import subprocess
import sys
import tarfile

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from run_all import PROJECTS, resolve_tag

PMD = r"D:\Tools\pmd-bin-7.29.0\bin\pmd.bat"      # <-- GANTI sesuai folder PMD-mu
RULESET = "rulesets/java/quickstart.xml"
REPOS = "data/raw/repos"
WORK = "data/raw/pmd_src"
PMD_COLS = ["pmd_total", "pmd_high"]


def norm(p):
    return p.replace("\\", "/")


def export_java(repo, tag, dest):
    """Ekspor file .java dari tag (tanpa mengubah repo), kecuali folder test."""
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    data = subprocess.run(["git", "-C", repo, "archive", "--format=tar", tag],
                          capture_output=True, check=True).stdout
    n = 0
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            name = m.name
            if not (m.isfile() and name.endswith(".java")):
                continue
            if "src/test" in name or name.startswith("/") or ".." in name:
                continue
            target = os.path.join(dest, *name.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with tf.extractfile(m) as src, open(target, "wb") as out:
                out.write(src.read())
            n += 1
    return n


def run_pmd(src, out_csv):
    cmd = [PMD, "check", "-d", os.path.abspath(src), "-R", RULESET,
           "-f", "csv", "-r", os.path.abspath(out_csv),
           "--no-cache", "--no-fail-on-violation"]
    return subprocess.run(cmd, capture_output=True, text=True)


def per_file(csv_path, src):
    empty = pd.DataFrame(columns=["key"] + PMD_COLS)
    try:
        v = pd.read_csv(csv_path)
    except Exception:
        return empty
    if v.empty or "File" not in v.columns:
        return empty
    base = norm(os.path.abspath(src)).lower().rstrip("/") + "/"
    v["key"] = v["File"].map(lambda p: norm(str(p)).lower()).str.replace(
        base, "", regex=False)
    g = v.groupby("key").agg(
        pmd_total=("Priority", "size"),
        pmd_high=("Priority", lambda s: int((s <= 2).sum())),
    ).reset_index()
    return g


if __name__ == "__main__":
    project = sys.argv[1]
    only = set(sys.argv[2:])            # opsional: versi tertentu, mis. 1.3
    repo = f"{REPOS}/{project}"
    for v in PROJECTS[project]["versions"]:
        if only and v not in only:
            continue
        src_csv = f"data/processed/merged2_{project}-{v}.csv"
        out = f"data/processed/merged3_{project}-{v}.csv"
        if not os.path.exists(src_csv):
            print(f"[{project}-{v}] merged2 tidak ada, dilewati")
            continue
        if os.path.exists(out):
            print(f"[{project}-{v}] sudah ada, dilewati")
            continue
        tag = resolve_tag(repo, project, v)
        if not tag:
            print(f"[{project}-{v}] tag tidak ditemukan, dilewati")
            continue

        dest = f"{WORK}/{project}-{v}"
        n = export_java(repo, tag, dest)
        print(f"[{project}-{v}] tag {tag}: {n} file .java diekspor, "
              f"menjalankan PMD...")
        raw_csv = f"data/processed/pmd_raw_{project}-{v}.csv"
        r = run_pmd(dest, raw_csv)
        if r.returncode not in (0, 4, 5):
            print("   PMD GAGAL, kode", r.returncode)
            print((r.stderr or r.stdout)[-600:])
            continue

        feats = per_file(raw_csv, dest)
        df = pd.read_csv(src_csv)
        df["key"] = df["file"].map(norm).str.lower()
        merged = df.merge(feats, on="key", how="left").drop(columns="key")
        merged[PMD_COLS] = merged[PMD_COLS].fillna(0)
        merged.to_csv(out, index=False)
        share = (merged["pmd_total"] > 0).mean()
        print(f"   -> {out} {merged.shape}, file dengan temuan PMD: {share:.0%}, "
              f"rata-rata temuan/file: {merged['pmd_total'].mean():.1f}")
        shutil.rmtree(dest, ignore_errors=True)