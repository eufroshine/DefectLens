import sys
import pandas as pd

# contoh: python src\extract\merge_promise.py ant 1.3 rel/1.3
project, version, tag = sys.argv[1], sys.argv[2], sys.argv[3]

promise = pd.read_csv(
    f"data/raw/PROMISE-backup-master/bug-data/{project}/{project}-{version}.csv")
git = pd.read_csv(f"data/processed/git_{project}_{tag.replace('/', '_')}.csv")

git["norm"] = git["file"].str.replace("\\", "/", regex=False)
git["base"] = git["norm"].str.split("/").str[-1]
by_base = {b: g for b, g in git.groupby("base")}


def find(name):
    outer = name.split("$")[0]              # kelas inner -> file induknya
    rel = outer.replace(".", "/") + ".java"
    base = rel.split("/")[-1]
    if base in by_base:
        g = by_base[base]
        hit = g[g["norm"].str.endswith("/" + rel) | (g["norm"] == rel)]
        if len(hit):
            return hit.iloc[0]["file"]
    return None


promise["file"] = promise["name"].apply(find)
matched = promise.dropna(subset=["file"])
print(f"cocok: {len(matched)} dari {len(promise)} "
      f"({len(matched) / len(promise):.0%})")

merged = matched.merge(git.drop(columns=["norm", "base"]), on="file", how="left")
out = f"data/processed/merged_{project}-{version}.csv"
merged.to_csv(out, index=False)
print(merged.shape, "->", out)
print("Contoh yang tidak cocok:",
      promise[promise["file"].isna()]["name"].head(10).tolist())