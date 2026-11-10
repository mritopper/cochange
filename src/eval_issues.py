"""Experiment 2: issue-to-symbol localization on SWE-bench Verified.

No oracle here: the only input is the issue text. BM25 over every function,
method and class in the pre-fix snapshot gives a first ranking; its top-M hits
act as seeds whose neighbourhoods are searched for the symbols BM25 missed.
We compare WHICH neighbourhood is worth searching:

  static     call / containment / inheritance neighbours of the seeds
  samefile   symbols next to the seeds in the same file
  cochange   symbols historically edited together with the seeds
  hot        no seeds at all: symbols most often edited in the past (control
             for "history only helps because busy code gets fixed again")

Every expansion list is fused with the BM25 list by reciprocal rank fusion.
All constants (M=10 seeds, RRF k=60) were fixed before any run. Co-change
evidence is restricted to ancestors of each task's base_commit.

Usage: python src/eval_issues.py [repo ...]     (default: all 12 repositories)
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cochange import CoChangeIndex, ancestors, fmt_ci, load_history, paired_bootstrap, rrf  # noqa: E402
from gold import gold_symbols  # noqa: E402
from eval_heldout import rank_samefile, rank_static  # noqa: E402
from static_graph import Snapshots, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
M_SEEDS = 10
TOP = 200
KS = (1, 5, 10, 20, 50)
ALPHAS = (0.05, 0.1, 0.25, 0.5)
RANKERS = ("bm25", "bm25+hot", "bm25+samefile", "bm25+static", "bm25+cc",
           "bm25+static+cc", "bm25+samefile+static", "bm25+samefile+static+cc") + tuple(
    f"bm25+{e}@{a}" for e in ("samefile", "static", "cc") for a in ALPHAS)
COMPLETERS = ("issue", "samefile", "static", "cochange", "samefile+static", "samefile+static+cc",
              "issue+samefile", "issue+static", "issue+cc", "issue+samefile+static",
              "issue+samefile+static+cc")


def bm25(query: list[str], syms: dict, k1: float = 1.2, b: float = 0.75) -> list[str]:
    q = set(query)
    n = len(syms)
    avg = sum(s["len"] for s in syms.values()) / max(n, 1)
    df = defaultdict(int)
    hits = {}
    for sid, info in syms.items():
        toks = info["tokens"]
        found = [(t, toks[t]) for t in q if t in toks] if len(q) < len(toks) else \
                [(t, c) for t, c in toks.items() if t in q]
        if found:
            hits[sid] = found
            for t, _ in found:
                df[t] += 1
    idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    score = {}
    for sid, found in hits.items():
        norm = k1 * (1 - b + b * syms[sid]["len"] / avg)
        score[sid] = sum(idf[t] * tf * (k1 + 1) / (tf + norm) for t, tf in found)
    return [s for s, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP]]


def expand(seeds: list[str], neighbours) -> list[str]:
    """Rank candidates by rank-discounted, weighted votes from the seeds."""
    score = defaultdict(float)
    for i, s in enumerate(seeds):
        for c, w in neighbours(s).items():
            score[c] += w / (60 + i + 1)
    return [s for s, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP]]


def run_repo(repo_full: str) -> list[dict]:
    warnings.simplefilter("ignore")
    name = repo_full.split("/")[1]
    repo = str(ROOT / "data" / "repos" / name)
    d = pd.read_parquet(ROOT / "data" / "swebench" / "verified.parquet")
    d = d[d.repo == repo_full].sort_values("created_at")
    index = CoChangeIndex(load_history(ROOT / "data" / "history_swe" / f"{name}.jsonl"))
    snaps = Snapshots(repo, with_text=True)
    rows = []
    for t in d.itertuples():
        g = gold_symbols(repo, t.base_commit, t.patch)
        syms, file_of, adj = snaps.at(t.base_commit)
        gold = {s for s in g["symbols"] if s in syms}
        row = {"repo": name, "id": t.instance_id, "n_gold": len(gold), "n_symbols": len(syms),
               "new_code": g["new_code"], "module_level": g["module_level"],
               "n_gold_files": len({file_of[s] for s in gold})}
        if not gold:
            rows.append(row)
            continue
        visible = ancestors(repo, t.base_commit)
        row["history_commits"] = len(visible)
        bm = bm25(tokenize(t.problem_statement), syms)
        seeds = bm[:M_SEEDS]

        def nb_static(s):
            return {c: 1.0 / math.log(2 + len(adj[c])) for c in adj.get(s, ())}

        def nb_samefile(s):
            f, line = file_of[s], syms[s]["line"]
            peers = sorted((x for x in syms if file_of[x] == f and x != s),
                           key=lambda x: abs(syms[x]["line"] - line))
            return {c: 1.0 / (1 + i) for i, c in enumerate(peers[:50])}

        def nb_cc(s):
            n = index.support(s, visible)
            return {c: w / (n + 2) for c, w in index.neighbours(s, visible).items() if c in syms}

        ex_static, ex_same, ex_cc = expand(seeds, nb_static), expand(seeds, nb_samefile), expand(seeds, nb_cc)
        hot = sorted(((index.change_count(s, visible), s) for s in index.touched if s in syms), reverse=True)
        hot = [s for c, s in hot[:TOP] if c > 0]
        lists = {
            "bm25": bm, "bm25+hot": rrf([bm, hot]), "bm25+samefile": rrf([bm, ex_same]),
            "bm25+static": rrf([bm, ex_static]), "bm25+cc": rrf([bm, ex_cc]),
            "bm25+static+cc": rrf([bm, ex_static, ex_cc]),
            "bm25+samefile+static": rrf([bm, ex_same, ex_static]),
            "bm25+samefile+static+cc": rrf([bm, ex_same, ex_static, ex_cc]),
        }
        for e, lst in (("samefile", ex_same), ("static", ex_static), ("cc", ex_cc)):
            for a in ALPHAS:
                lists[f"bm25+{e}@{a}"] = rrf([bm, lst], weights=[1.0, a])
        # Edit completion on the same tasks: one edited symbol is known (the
        # seed), the issue text is still available. Which signal finds the rest?
        if len(gold) >= 2:
            universe = set(syms)
            comp, cands, meta = [], [], {}
            for seed in sorted(gold):
                rest = gold - {seed}
                iss = [x for x in bm if x != seed]
                sf = rank_samefile(seed, syms, file_of)[:TOP]
                st = rank_static(seed, adj)[:TOP]
                cc = index.rank(seed, visible, universe)[:TOP]
                cl = {"issue": iss, "samefile": sf, "static": st, "cochange": cc,
                      "samefile+static": rrf([sf, st]), "samefile+static+cc": rrf([sf, st, cc]),
                      "issue+samefile": rrf([iss, sf]), "issue+static": rrf([iss, st]),
                      "issue+cc": rrf([iss, cc]), "issue+samefile+static": rrf([iss, sf, st]),
                      "issue+samefile+static+cc": rrf([iss, sf, st, cc])}
                comp.append({r: sorted(l.index(x) if x in l else -1 for x in rest) for r, l in cl.items()})
                a, b = cl["issue+samefile+static"][:10], cl["issue+samefile+static+cc"][:10]
                cands.append({"seed": seed, "rest": sorted(rest), "A": a, "B": b})
                for x in [seed, *a, *b]:
                    meta[x] = [file_of[x], syms[x]["line"], syms[x]["end"]]
            row["completion"] = comp
            row["candidates"] = cands
            row["meta"] = meta
            row["base_commit"] = t.base_commit
        row["seed_hit"] = bool(gold & set(seeds))
        row["cc_candidates"] = len(ex_cc)
        row["ranks"] = {r: sorted(lst.index(s) if s in lst else -1 for s in gold) for r, lst in lists.items()}
        row["raw"] = {"cc": sorted(ex_cc.index(s) if s in ex_cc else -1 for s in gold),
                      "static": sorted(ex_static.index(s) if s in ex_static else -1 for s in gold),
                      "samefile": sorted(ex_same.index(s) if s in ex_same else -1 for s in gold)}
        rows.append(row)
    return rows


def metrics(ranks: list[int]) -> dict:
    found = [r for r in ranks if r >= 0]
    out = {"mrr": 1.0 / (min(found) + 1) if found else 0.0}
    for k in KS:
        inside = sum(1 for r in found if r < k)
        out[f"r@{k}"] = inside / len(ranks)
        out[f"c@{k}"] = float(inside == len(ranks))
    return out


def table(per: dict, names, n: int, cols) -> None:
    print(f"{'ranker':26s} " + " ".join(f"{c:>6s}" for c in cols))
    for r in names:
        print(f"{r:26s} " + " ".join(f"{sum(m[c] for m in per[r]) / n:6.3f}" for c in cols))


def report(rows: list[dict], title: str) -> None:
    ok = [r for r in rows if r.get("ranks")]
    if not ok:
        return
    print(f"\n== {title}: {len(ok)} tasks ({len(rows) - len(ok)} skipped: no pre-existing symbol edited)")
    cols = ["mrr"] + [f"r@{k}" for k in KS] + [f"c@{k}" for k in (5, 10, 20)]
    per = {r: [metrics(row["ranks"][r]) for row in ok] for r in RANKERS}
    table(per, RANKERS, len(ok), cols)
    ids = [r["id"] for r in ok]
    for a, b in (("bm25+cc", "bm25"), ("bm25+static", "bm25"), ("bm25+samefile", "bm25"),
                 ("bm25+cc@0.1", "bm25"), ("bm25+static@0.1", "bm25"), ("bm25+samefile@0.1", "bm25")):
        d = paired_bootstrap([x["r@10"] for x in per[a]], [x["r@10"] for x in per[b]], ids, n=5000)
        print(f"  dr@10 {a} - {b}: {fmt_ci(d)}")


def report_completion(rows: list[dict], title: str) -> None:
    ok = [r for r in rows if r.get("completion")]
    if not ok:
        return
    seeds = [(r["id"], c) for r in ok for c in r["completion"]]
    print(f"\n== COMPLETION {title}: {len(ok)} multi-symbol tasks, {len(seeds)} seeds")
    cols = ["mrr"] + [f"r@{k}" for k in KS] + [f"c@{k}" for k in (5, 10, 20)]
    per = {r: [metrics(c[r]) for _, c in seeds] for r in COMPLETERS}
    table(per, COMPLETERS, len(seeds), cols)
    ids = [i for i, _ in seeds]
    for a, b in (("cochange", "static"), ("cochange", "samefile"), ("cochange", "issue"),
                 ("samefile+static+cc", "samefile+static"), ("issue+cc", "issue"),
                 ("issue+static", "issue"), ("issue+samefile", "issue"),
                 ("issue+samefile+static+cc", "issue+samefile+static")):
        d = paired_bootstrap([x["r@10"] for x in per[a]], [x["r@10"] for x in per[b]], ids, n=5000)
        print(f"  dr@10 {a} - {b}: {fmt_ci(d)}")


if __name__ == "__main__":
    d = pd.read_parquet(ROOT / "data" / "swebench" / "verified.parquet")
    wanted = sys.argv[1:]
    repos = [r for r in sorted(d.repo.unique()) if not wanted or r.split("/")[1] in wanted]
    with Pool(min(len(repos), 8)) as p:
        allrows = [row for rows in p.imap_unordered(run_repo, repos) for row in rows]
    out = ROOT / "results" / ("issues.json" if not wanted else f"issues_{'_'.join(wanted)}.json")
    json.dump(allrows, open(out, "w"))
    report(allrows, "ALL")
    report([r for r in allrows if r["n_gold"] >= 2], "ALL, fixes touching 2+ symbols")
    report([r for r in allrows if r["n_gold"] == 1], "ALL, single-symbol fixes")
    report([r for r in allrows if r["repo"] == "django"], "django only")
    report([r for r in allrows if r["repo"] != "django"], "all but django")
    report_completion(allrows, "ALL")
    report_completion([r for r in allrows if r["repo"] == "django"], "django only")
    report_completion([r for r in allrows if r["repo"] != "django"], "all but django")
