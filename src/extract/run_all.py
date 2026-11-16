import os
import re
import subprocess
import sys

import pandas as pd

PROJECTS = {
    "ant":    {"url": "https://github.com/apache/ant.git",
               "versions": ["1.3", "1.4", "1.5", "1.6", "1.7"]},
    "poi":    {"url": "https://github.com/apache/poi.git",
               "versions": ["1.5", "2.0", "2.5", "3.0"]},
    "lucene": {"url": "https://github.com/apache/lucene.git",
               "versions": ["2.0", "2.2", "2.4"]},
    "camel":  {"url": "https://github.com/apache/camel.git",
               "versions": ["1.0", "1.2", "1.4", "1.6"]},
}

# Isi manual kalau tebakan otomatis salah, contoh: ("poi", "2.5"): "REL_2_5_FINAL"
TAG_OVERRIDE = {
    ("poi", "2.0"): "REL_2_0_RC1",
    ("poi", "2.5"): "REL_2_5_1",
}

BAD = re.compile(r"alpha|beta|milestone|snapshot|[-_.]rc\d*|[-_.]b\d|[-_.]m\d",
                 re.I)
REPOS = "data/raw/repos"


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args],
                       capture_output=True, text=True)
    return r.stdout.strip()


def resolve_tag(repo, project, version):
    if (project, version) in TAG_OVERRIDE:
        return TAG_OVERRIDE[(project, version)]
    want = version.split(".")
    cands = []
    for t in git(repo, "tag").splitlines():
        if BAD.search(t):
            continue
        nums = re.findall(r"\d+", t)
        if nums == want or nums == want + ["0"]:
            ts = git(repo, "log", "-1", "--format=%ct", t)
            cands.append((int(ts) if ts.isdigit() else 0, t))
    return min(cands)[1] if cands else None


def run_project(project):
    cfg = PROJECTS[project]
    repo = f"{REPOS}/{project}"
    os.makedirs(REPOS, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.isdir(repo):
        print(f"[{project}] clone (bisa lama)...")
        subprocess.run(["git", "clone", cfg["url"], repo], check=True)

    summary = []
    for v in cfg["versions"]:
        out = f"data/processed/merged_{project}-{v}.csv"
        if os.path.exists(out):
            print(f"[{project}-{v}] sudah ada, dilewati")
            summary.append({"project": project, "version": v,
                            "tag": "(sudah ada)", "cocok": ""})
            continue

        tag = resolve_tag(repo, project, v)
        if not tag:
            print(f"[{project}-{v}] TAG TIDAK DITEMUKAN, isi TAG_OVERRIDE")
            summary.append({"project": project, "version": v,
                            "tag": "-", "cocok": "tag tidak ditemukan"})
            continue

        print(f"[{project}-{v}] pakai tag: {tag}")
        r1 = subprocess.run([sys.executable, "src/extract/git_features.py",
                             project, tag])
        if r1.returncode != 0:
            summary.append({"project": project, "version": v,
                            "tag": tag, "cocok": "git_features gagal"})
            continue

        r2 = subprocess.run([sys.executable, "src/extract/merge_promise.py",
                             project, v, tag], capture_output=True, text=True)
        print(r2.stdout)
        if r2.returncode != 0:
            print(r2.stderr[-500:])
        m = re.search(r"cocok: (\d+) dari (\d+) \((\d+)%\)", r2.stdout)
        summary.append({"project": project, "version": v, "tag": tag,
                        "cocok": f"{m.group(3)}%" if m else "gagal"})

    return summary


if __name__ == "__main__":
    targets = sys.argv[1:] or list(PROJECTS)
    rows = []
    for p in targets:
        rows += run_project(p)
    df = pd.DataFrame(rows)
    df.to_csv("data/processed/run_summary.csv", index=False)
    print("\n=== RINGKASAN ===")
    print(df.to_string(index=False))