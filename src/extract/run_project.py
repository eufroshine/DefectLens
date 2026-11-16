import subprocess
import sys

project = sys.argv[1]
for pair in sys.argv[2:]:                  # format: versi=tag
    version, tag = pair.split("=", 1)
    subprocess.run([sys.executable, "src/extract/git_features.py",
                    project, tag], check=True)
    subprocess.run([sys.executable, "src/extract/merge_promise.py",
                    project, version, tag], check=True)