"""Summarize Experiment 3 (results/gemma_select.jsonl)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cochange import fmt_ci, paired_bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in open(ROOT / "results" / "gemma_select.jsonl")]
by = defaultdict(dict)
for r in rows:
    by[(r["id"], r["seed"])][r["cond"]] = r
pairs = {k: v for k, v in by.items() if "A" in v and "B" in v}
print(f"{len(rows)} calls, {len(pairs)} paired seeds, {len({k[0] for k in pairs})} tasks, model {rows[0]['model']}")


def stats(r):
    rest, picks = set(r["rest"]), r["picks"]
    hit = len(set(picks) & rest)
    return {"recall": hit / len(rest), "any": float(hit > 0), "n_picks": len(picks),
            "precision": hit / len(picks) if picks else 0.0,
            "list_top5": len(set(r["candidates"][:5]) & rest) / len(rest),
            "list_same_n": len(set(r["candidates"][:len(picks)]) & rest) / len(rest),
            "empty": float(not picks),
            "ceiling": len(set(r["candidates"]) & rest) / len(rest)}


def block(keys, title):
    if not keys:
        return
    S = {c: [stats(pairs[k][c]) for k in keys] for c in "AB"}
    ids = [k[0] for k in keys]
    print(f"\n== {title}: {len(keys)} seeds")
    print(f"{'':30s}{'A: no history':>15s}{'B: + history':>15s}   B - A [95% CI]")
    for m, label in (("ceiling", "candidates contain (R@10)"), ("list_top5", "list order, top 5"),
                     ("recall", "Gemma picks: recall"), ("any", "Gemma picks: any correct"),
                     ("precision", "Gemma picks: precision"), ("n_picks", "picks per call"),
                     ("list_same_n", "list order, same n as Gemma"), ("empty", "calls with no pick")):
        a = [x[m] for x in S["A"]]
        b = [x[m] for x in S["B"]]
        d = paired_bootstrap(b, a, ids, n=5000)
        print(f"{label:30s}{sum(a) / len(a):15.3f}{sum(b) / len(b):15.3f}   {fmt_ci(d)}")
    for c in "AB":
        g = [x["recall"] for x in S[c]]
        l = [x["list_top5"] for x in S[c]]
        print(f"  {c}: Gemma recall - list-top-5 recall: {fmt_ci(paired_bootstrap(g, l, ids, n=5000))}")
        s = [x["list_same_n"] for x in S[c]]
        print(f"  {c}: Gemma recall - same-size list prefix: {fmt_ci(paired_bootstrap(g, s, ids, n=5000))}")


keys = sorted(pairs)
block(keys, "ALL")
block([k for k in keys if pairs[k]["A"]["repo"] == "django"], "django")
block([k for k in keys if pairs[k]["A"]["repo"] != "django"], "all but django")
first = {}
for k in keys:
    first.setdefault(k[0], k)
block(sorted(first.values()), "one seed per task")
