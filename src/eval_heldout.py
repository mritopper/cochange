"""Experiment 1: edit completion on held-out history.

Setting: an agent has found ONE symbol that a fix must edit (the seed). Which
other symbols does the same fix edit? For every commit in the final 25% of a
repository's history that edits 2-15 non-test symbols, each edited symbol
takes a turn as the seed and every ranker ranks the other symbols that exist
in the parent snapshot. Co-change evidence comes only from ancestors of the
test commit, so nothing leaks from the future.

Rankers
  samefile   symbols in the seed's file, nearest first
  static     1-hop then 2-hop neighbours in the static graph (static_graph.py)
  cochange   symbols most often edited together with the seed in the past
  a+b        reciprocal rank fusion of the named lists

Usage: python src/eval_heldout.py [repo ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cochange import (CoChangeIndex, ancestors, code_symbols, fmt_ci, load_history,  # noqa: E402
                      paired_bootstrap, recall_at, reciprocal_rank, rrf)
from static_graph import Snapshots  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAX_SYMS = 15
MAX_TEST = 300   # held-out commits evaluated per repository
KS = (1, 5, 10, 20)
RANKERS = ("samefile", "static", "cochange", "cochange-decay", "samefile+static",
           "samefile+cc", "static+cc", "samefile+static+cc")


def rank_samefile(seed, syms, file_of):
    f, line = file_of[seed], syms[seed]["line"]
    peers = [s for s in syms if file_of[s] == f and s != seed]
    return sorted(peers, key=lambda s: (abs(syms[s]["line"] - line), s))


def rank_static(seed, adj):
    one = adj.get(seed, set())
    two = set().union(*(adj[n] for n in one)) - one - {seed} if one else set()
    key = lambda s: (len(adj[s]), s)          # specific (low-degree) neighbours first
    return sorted(one, key=key) + sorted(two, key=key)


def evaluate(name: str) -> dict:
    repo = str(ROOT / "data" / "repos" / name)
    hist = ROOT / "data" / "history_swe" / f"{name}.jsonl"
    if not hist.exists():
        hist = ROOT / "data" / "history" / f"{name}.jsonl"
    commits = load_history(hist)
    index = CoChangeIndex(commits)
    snaps = Snapshots(repo)
    split = int(len(commits) * 0.75)
    rows = []                      # one per seed: {"commit":..., ranker: {metric: value}}
    pair = Counter()
    eligible = [c for c in commits[split:] if 2 <= len(code_symbols(c)) <= MAX_SYMS]
    if len(eligible) > MAX_TEST:   # evenly spaced, deterministic subsample
        step = len(eligible) / MAX_TEST
        eligible = [eligible[int(i * step)] for i in range(MAX_TEST)]
    for c in eligible:
        code = code_symbols(c)
        syms, file_of, adj = snaps.at(c["sha"] + "^")
        code = [s for s in code if s in syms]          # symbols that exist before the fix
        if len(code) < 2:
            continue
        visible = ancestors(repo, c["sha"] + "^")
        universe = set(syms)
        for seed in code:
            gold = set(code) - {seed}
            sf = rank_samefile(seed, syms, file_of)
            st = rank_static(seed, adj)
            cc = index.rank(seed, visible, universe)
            ccd = index.rank(seed, visible, universe, now=c["time"], half_life_days=365)
            lists = {"samefile": sf, "static": st, "cochange": cc, "cochange-decay": ccd,
                     "samefile+static": rrf([sf, st]), "samefile+cc": rrf([sf, cc]),
                     "static+cc": rrf([st, cc]), "samefile+static+cc": rrf([sf, st, cc])}
            row = {"commit": c["sha"]}
            for r, lst in lists.items():
                row[r] = {"mrr": reciprocal_rank(lst, gold),
                          **{f"r@{k}": recall_at(lst, gold, k) for k in KS},
                          **{f"c@{k}": float(gold <= set(lst[:k])) for k in KS}}
            rows.append(row)
            one = adj.get(seed, set())
            two = set().union(*(adj[n] for n in one)) | one if one else set()
            hist = index.neighbours(seed, visible)
            for g in gold:
                same, s2, h = file_of[g] == file_of[seed], g in two, g in hist
                pair["pairs"] += 1
                pair["same_file"] += same
                pair["static_1hop"] += g in one
                pair["static_2hop"] += s2
                pair["history"] += h
                pair["history_only"] += h and not s2 and not same
                pair["cross_file"] += not same
                pair["cross_file_static2"] += (not same) and s2
                pair["cross_file_history"] += (not same) and h
                pair["none"] += not (h or s2 or same)
    res = {"repo": name, "test_commits": len({r["commit"] for r in rows}), "seeds": len(rows),
           "pairs": dict(pair)}
    for r in RANKERS:
        res[r] = {m: sum(row[r][m] for row in rows) / max(len(rows), 1) for m in rows[0][r]} if rows else {}
    groups = [r["commit"] for r in rows]
    res["delta_r@10"] = {
        "static+cc - static": paired_bootstrap([r["static+cc"]["r@10"] for r in rows],
                                               [r["static"]["r@10"] for r in rows], groups),
        "samefile+static+cc - samefile+static": paired_bootstrap(
            [r["samefile+static+cc"]["r@10"] for r in rows],
            [r["samefile+static"]["r@10"] for r in rows], groups),
    }
    res["_rows"] = rows
    return res


def show(res: dict) -> None:
    print(f"\n== {res['repo']}: {res['test_commits']} held-out commits, {res['seeds']} seeds")
    print(f"{'ranker':20s} {'MRR':>6s} " + " ".join(f"{'R@'+str(k):>6s}" for k in KS) + "  " +
          " ".join(f"{'C@'+str(k):>6s}" for k in KS))
    for r in RANKERS:
        m = res[r]
        print(f"{r:20s} {m['mrr']:6.3f} " + " ".join(f"{m[f'r@{k}']:6.3f}" for k in KS) + "  " +
              " ".join(f"{m[f'c@{k}']:6.3f}" for k in KS))
    for k, v in res["delta_r@10"].items():
        print(f"  dR@10 {k}: {fmt_ci(v)}")
    p = res["pairs"]
    n, cf = max(p["pairs"], 1), max(p["cross_file"], 1)
    print(f"  pairs {n}: same-file {p['same_file']/n:.1%} | static<=2 {p['static_2hop']/n:.1%} | "
          f"history {p['history']/n:.1%} | history-only {p['history_only']/n:.1%} | none {p['none']/n:.1%}")
    print(f"  cross-file pairs {p['cross_file']}: static<=2 {p['cross_file_static2']/cf:.1%} | "
          f"history {p['cross_file_history']/cf:.1%}")


def _job(name):
    import warnings
    warnings.simplefilter("ignore")
    return evaluate(name)


ALL_REPOS = ["httpx", "rich", "fastapi", "requests", "flask", "seaborn", "pylint", "pytest", "xarray",
             "sphinx", "scikit-learn", "astropy", "django", "matplotlib", "sympy"]

if __name__ == "__main__":
    from multiprocessing import Pool
    names = sys.argv[1:] or ALL_REPOS
    allres, allrows = [], []
    with Pool(min(len(names), 8)) as pool:
        for r in pool.imap(_job, names):
            show(r)
            allrows += [dict(row, repo=r["repo"]) for row in r.pop("_rows")]
            allres.append(r)
    if len(names) > 1:
        g = [f"{r['repo']}:{r['commit']}" for r in allrows]
        print(f"\n== pooled over {len(names)} repos: {len(set(g))} commits, {len(allrows)} seeds")
        for rk in RANKERS:
            print(f"{rk:20s} " + " ".join(
                f"{sum(r[rk][m] for r in allrows) / len(allrows):6.3f}"
                for m in ["mrr"] + [f"r@{k}" for k in KS] + [f"c@{k}" for k in KS]))
        for a, b in (("static+cc", "static"), ("samefile+static+cc", "samefile+static"),
                     ("cochange", "static"), ("cochange", "samefile")):
            d = paired_bootstrap([r[a]["r@10"] for r in allrows], [r[b]["r@10"] for r in allrows], g)
            print(f"  dR@10 {a} - {b}: {fmt_ci(d)}")
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(allres, open(ROOT / "results" / "heldout.json", "w"), indent=1)
