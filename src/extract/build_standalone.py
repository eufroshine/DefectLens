import io
import os
import sys
import tarfile
import subprocess

import lizard
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from run_all import PROJECTS, resolve_tag

REPOS = "data/raw/repos"
LZ = ["lz_nloc", "lz_token", "lz_nfunc", "lz_ccn_sum", "lz_ccn_max",
      "lz_ccn_avg", "lz_len_max", "lz_param_max", "lz_param_avg"]


def norm(p):
    return p.replace("\\", "/").lower()


def lizard_features(repo, tag):
    data = subprocess.run(["git", "-C", repo, "archive", "--format=tar", tag],
                          capture_output=True, check=True).stdout
    rows, failed = [], 0
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            if not (m.isfile() and m.name.endswith(".java")):
                continue
            if "src/test" in m.name:
                continue
            code = tf.extractfile(m).read().decode("utf-8", errors="ignore")
            try:
                info = lizard.analyze_file.analyze_source_code(m.name, code)
            except Exception:
                failed += 1
                continue
            fs = info.function_list
            ccn = [f.cyclomatic_complexity for f in fs]
            par = [f.parameter_count for f in fs]
            rows.append({
                "key": norm(m.name),
                "lz_nloc": info.nloc, "lz_token": info.token_count,
                "lz_nfunc": len(fs), "lz_ccn_sum": sum(ccn),
                "lz_ccn_max": max(ccn) if ccn else 0,
                "lz_ccn_avg": sum(ccn) / len(ccn) if ccn else 0,
                "lz_len_max": max((f.nloc for f in fs), default=0),
                "lz_param_max": max(par) if par else 0,
                "lz_param_avg": sum(par) / len(par) if par else 0,
            })
    return pd.DataFrame(rows, columns=["key"] + LZ), failed

def find_path(name, by_base):
    outer = str(name).split("$")[0]
    rel = outer.replace(".", "/").lower() + ".java"
    base = rel.split("/")[-1]
    for p in by_base.get(base, []):
        if p == rel or p.endswith("/" + rel):
            return p
    return None

if __name__ == "__main__":
    for p in (sys.argv[1:] or list(PROJECTS)):
        repo = f"{REPOS}/{p}"
        for v in PROJECTS[p]["versions"]:
            src = f"data/processed/merged3_{p}-{v}.csv"
            out = f"data/processed/merged4_{p}-{v}.csv"
            if not os.path.exists(src) or os.path.exists(out):
                continue
            tag = resolve_tag(repo, p, v)
            if not tag:
                print(f"[{p}-{v}] tag tidak ditemukan")
                continue
            feats, failed = lizard_features(repo, tag)
            feats = feats.drop_duplicates("key")
            by_base = {}
            for k in feats["key"]:
                by_base.setdefault(k.split("/")[-1], []).append(k)
            df = pd.read_csv(src)
            df["key"] = df["name"].map(lambda n: find_path(n, by_base))
            merged = df.merge(feats, on="key", how="left").drop(columns="key")
            matched = merged["lz_nloc"].notna().mean()
            if matched < 0.95:
                print(f"   PERINGATAN: hanya {matched:.0%} cocok")
            merged[LZ] = merged[LZ].fillna(0)
            merged.to_csv(out, index=False)
            print(f"[{p}-{v}] {tag}: {len(feats)} file dianalisis, "
                  f"gagal {failed}, cocok dengan PROMISE {matched:.0%} -> {out}")