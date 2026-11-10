"""Mine symbol-level history for every SWE-bench Verified repository.

Only commits reachable from some task's base_commit are read, so the mined
history never extends past the newest task snapshot of each repository.
"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mine_history import mine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def job(args):
    warnings.simplefilter("ignore")
    name, revs = args
    out = ROOT / "data" / "history_swe" / f"{name}.jsonl"
    n = mine(str(ROOT / "data" / "repos" / name), str(out), revs)
    return f"{name}: {n} commits"


if __name__ == "__main__":
    d = pd.read_parquet(ROOT / "data" / "swebench" / "verified.parquet")
    (ROOT / "data" / "history_swe").mkdir(exist_ok=True)
    jobs = [(repo.split("/")[1], sorted(set(g.base_commit))) for repo, g in d.groupby("repo")]
    with Pool(len(jobs)) as p:
        for msg in p.imap_unordered(job, jobs):
            print(msg, flush=True)
