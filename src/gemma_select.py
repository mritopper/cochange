"""Experiment 3: does a small local model benefit from history-aware candidates?

For every multi-symbol SWE-bench Verified fix and every edited symbol taken as
"the edit already made", Gemma 4 E4B (4-bit, MLX, on a laptop) sees the issue,
the edited function and TEN candidate symbols, and must say which candidates
also need editing (at most five). The only thing that differs between the two
conditions is how the ten candidates were retrieved:

  A  issue text + same-file + static-graph neighbours
  B  the same, plus co-change history

Output: results/gemma_select.jsonl (one line per model call; resumable).

Usage: python src/gemma_select.py [--limit N] [--model HF_ID]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "gemma_select.jsonl"
MODEL = "lmstudio-community/gemma-4-E4B-it-MLX-4bit"
MAX_PICKS = 5


def show(repo: str, commit: str, path: str) -> list[str]:
    r = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True)
    return r.stdout.decode("utf-8", "replace").splitlines()


def build_prompt(row: dict, cand: dict, ids: list[str], files: dict) -> str:
    repo = str(ROOT / "data" / "repos" / row["repo"])

    def lines_of(sym):
        path, a, b = row["meta"][sym]
        if path not in files:
            files[path] = show(repo, row["base_commit"], path)
        return files[path][a - 1:b]

    seed_src = lines_of(cand["seed"])
    if len(seed_src) > 60:
        seed_src = seed_src[:60] + ["    ..."]
    listing = []
    for i, sym in enumerate(ids, 1):
        head = " ".join(l.strip() for l in lines_of(sym)[:2])[:150]
        listing.append(f"{i}. {sym}\n   {head}")
    issue = row["problem_statement"].strip()
    if len(issue) > 2500:
        issue = issue[:2500] + " ..."
    return (
        f"You are fixing an issue in the Python project `{row['repo']}`.\n\n"
        f"ISSUE:\n{issue}\n\n"
        f"You have already edited this symbol as part of the fix:\n"
        f"{cand['seed']}\n```python\n" + "\n".join(seed_src) + "\n```\n\n"
        "A complete fix may need changes in other places too. Candidates:\n"
        + "\n".join(listing) + "\n\n"
        f"Which candidates must ALSO be edited to fix this issue? Choose at most {MAX_PICKS}. "
        "Finish with one line in exactly this form: ANSWER: <comma-separated candidate numbers, or none>"
    )


def parse(text: str, n: int) -> list[int]:
    final = text.split("<channel|>")[-1]
    m = re.findall(r"ANSWER:\s*([^\n]*)", final) or re.findall(r"ANSWER:\s*([^\n]*)", text)
    if not m:
        return []
    picks = []
    for x in re.findall(r"\d+", m[-1]):
        v = int(x)
        if 1 <= v <= n and v not in picks:
            picks.append(v)
    return picks[:MAX_PICKS]


def main() -> None:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    import pandas as pd

    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    model_id = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else MODEL
    rows = [r for r in json.load(open(ROOT / "results" / "issues.json")) if r.get("candidates")]
    stmt = pd.read_parquet(ROOT / "data" / "swebench" / "verified.parquet").set_index("instance_id").problem_statement
    done = set()
    if OUT.exists():
        done = {(d["id"], d["seed"], d["cond"]) for d in map(json.loads, open(OUT))}
    jobs = []
    # first seed of every task first, so a partial run already covers all tasks
    depth = max(len(r["candidates"]) for r in rows)
    for j in range(depth):
        for r in sorted(rows, key=lambda r: r["id"]):
            if j < len(r["candidates"]):
                for cond in ("A", "B"):
                    if (r["id"], r["candidates"][j]["seed"], cond) not in done:
                        jobs.append((r, r["candidates"][j], cond))
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} calls to run ({len(done)} already done)", flush=True)
    model, tok = load(model_id)
    sampler = make_sampler(temp=0.0)
    t0 = time.time()
    with open(OUT, "a") as fo:
        for n, (r, cand, cond) in enumerate(jobs, 1):
            r["problem_statement"] = stmt[r["id"]]
            ids = cand[cond]
            # Thinking is switched off and the reply is pre-filled with "ANSWER:" so the
            # model commits to a selection directly (greedy decoding, ~3.5 s per call).
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": build_prompt(r, cand, ids, {})}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False) + "ANSWER:"
            text = "ANSWER:" + generate(model, tok, prompt=prompt, max_tokens=24, sampler=sampler, verbose=False)
            picks = [ids[i - 1] for i in parse(text.split("\n")[0], len(ids))]
            fo.write(json.dumps({"id": r["id"], "repo": r["repo"], "seed": cand["seed"], "cond": cond,
                                 "rest": cand["rest"], "candidates": ids, "picks": picks,
                                 "model": model_id, "raw": text[:80]}) + "\n")
            fo.flush()
            if n % 20 == 0:
                print(f"{n}/{len(jobs)}  {(time.time() - t0) / n:.1f}s per call", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
