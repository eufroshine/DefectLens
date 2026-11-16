import sys
from collections import defaultdict
import pandas as pd
from pydriller import Repository

project, tag = sys.argv[1], sys.argv[2]
REPO = f"data/raw/repos/{project}"

stats = defaultdict(lambda: {"commits": 0, "added": 0,
                             "deleted": 0, "authors": set()})

for c in Repository(REPO, to_tag=tag).traverse_commits():
    for m in c.modified_files:
        path = m.new_path or m.old_path
        if path and path.endswith(".java"):
            s = stats[path]
            s["commits"] += 1
            s["added"] += m.added_lines
            s["deleted"] += m.deleted_lines
            s["authors"].add(c.author.email)

rows = [{"file": p, "change_freq": s["commits"],
         "lines_added": s["added"], "lines_deleted": s["deleted"],
         "churn": s["added"] + s["deleted"],
         "n_authors": len(s["authors"])} for p, s in stats.items()]

df = pd.DataFrame(rows)
out = f"data/processed/git_{project}_{tag.replace('/', '_')}.csv"
df.to_csv(out, index=False)
print(df.shape, "->", out)