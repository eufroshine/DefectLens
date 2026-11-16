import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd
from pydriller import Repository

sys.path.insert(0, os.path.dirname(__file__))
from run_all import PROJECTS, resolve_tag, git   # pakai ulang logika tag

WINDOW_DAYS = 365          # bisa diganti, mis. 180
REPOS = "data/raw/repos"
FIX_RE = re.compile(r"\b(fix(e[sd])?|bugs?|defects?|faults?)\b", re.I)
NEW_COLS = ["p_change_freq", "p_lines_added", "p_lines_deleted", "p_churn",
            "p_n_authors", "p_recency_days", "p_fix_commits"]


def norm(p):
    return p.replace("\\", "/")


def period_features(repo, tag):
    ts = int(git(repo, "log", "-1", "--format=%ct", tag))
    end = datetime.fromtimestamp(ts, tz=timezone.utc)
    start = end - timedelta(days=WINDOW_DAYS)

    recs = defaultdict(lambda: {"n": 0, "add": 0, "del": 0, "auth": set(),
                                "fix": 0, "last": None})
    for c in Repository(repo, since=start, to=end).traverse_commits():
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

    rows = []
    for key, r in recs.items():
        rows.append({
            "key": key,
            "p_change_freq": r["n"],
            "p_lines_added": r["add"],
            "p_lines_deleted": r["del"],
            "p_churn": r["add"] + r["del"],
            "p_n_authors": len(r["auth"]),
            "p_recency_days": (end - r["last"]).days,
            "p_fix_commits": r["fix"],
        })
    return pd.DataFrame(rows, columns=["key"] + NEW_COLS)


if __name__ == "__main__":
    targets = sys.argv[1:] or list(PROJECTS)
    log = []
    for p in targets:
        repo = f"{REPOS}/{p}"
        for v in PROJECTS[p]["versions"]:
            src = f"data/processed/merged_{p}-{v}.csv"
            out = f"data/processed/merged2_{p}-{v}.csv"
            if not os.path.exists(src):
                print(f"[{p}-{v}] file merged_ tidak ada, dilewati")
                continue
            if os.path.exists(out):
                print(f"[{p}-{v}] sudah ada, dilewati")
                continue
            tag = resolve_tag(repo, p, v)
            if not tag:
                print(f"[{p}-{v}] tag tidak ditemukan, dilewati")
                continue
            print(f"[{p}-{v}] tag: {tag}, menghitung fitur periode...")
            feats = period_features(repo, tag)

            df = pd.read_csv(src)
            df["key"] = df["file"].map(norm)
            merged = df.merge(feats, on="key", how="left").drop(columns="key")
            for c in NEW_COLS:
                merged[c] = merged[c].fillna(
                    WINDOW_DAYS if c == "p_recency_days" else 0)
            merged.to_csv(out, index=False)
            share = (merged["p_change_freq"] > 0).mean()
            print(f"   -> {out} {merged.shape}, "
                  f"file yang berubah di jendela: {share:.0%}")
            log.append({"project": p, "version": v, "tag": tag})

    if log:
        pd.DataFrame(log).to_csv("data/processed/tags_used.csv", index=False)