#!/usr/bin/env python3
"""cochange: "what else usually changes with this?" for coding agents.

Standard library only; works in any git checkout with history. Designed to be
dropped into an agent sandbox (e.g. as a skill script) next to mine_history.py
and cochange.py.

  cochange_tool.py check [-k 10]          symbols historically edited together with
                                          the ones in your uncommitted diff, that
                                          the diff has NOT touched yet
  cochange_tool.py neighbors SYMBOL [-k]  co-change neighbours of one symbol
  cochange_tool.py export -o edges.json   co-change edges in the node-link schema
                                          of the released code graphs

The first call mines the repository history once and caches it under
.git/cochange/ keyed by HEAD, so later calls are instant.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cochange import CoChangeIndex, ancestors, load_history  # noqa: E402
from mine_history import HUNK, SymbolTable, is_test_path, mine, module_name  # noqa: E402


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout


def load_index(repo: str) -> tuple[CoChangeIndex, set[str]]:
    head = git(repo, "rev-parse", "HEAD").strip()
    cache = Path(git(repo, "rev-parse", "--absolute-git-dir").strip()) / "cochange"
    cache.mkdir(exist_ok=True)
    path = cache / f"{head}.jsonl"
    if not path.exists():
        mine(repo, str(path), "HEAD")
    return CoChangeIndex(load_history(str(path))), ancestors(repo, "HEAD")


def diff_symbols(repo: str) -> dict[str, str]:
    """Symbols touched by the uncommitted diff (working tree vs HEAD) -> file."""
    out: dict[str, str] = {}
    old_path = new_path = None
    told = tnew = None
    for line in git(repo, "diff", "HEAD", "-U0", "--no-color", "--no-ext-diff", "--", "*.py").splitlines():
        if line.startswith("diff --git "):
            old_path = new_path = told = tnew = None
        elif line.startswith("--- "):
            old_path = None if line[4:] == "/dev/null" else line[6:]
        elif line.startswith("+++ "):
            new_path = None if line[4:] == "/dev/null" else line[6:]
            if old_path:
                src = git(repo, "show", f"HEAD:{old_path}")
                told = SymbolTable(src, module_name(old_path))
            if new_path and (Path(repo) / new_path).exists():
                tnew = SymbolTable((Path(repo) / new_path).read_text(errors="replace"), module_name(new_path))
        elif line.startswith("@@"):
            m = HUNK.match(line)
            if not m:
                continue
            a, b, c, d = int(m[1]), int(m[2] or 1), int(m[3]), int(m[4] or 1)
            for table, lo, n, path in ((told, a, b, old_path), (tnew, c, d, new_path)):
                if table and n and path and not is_test_path(path):
                    for ln in range(lo, lo + n):
                        sid, kind = table.lookup(ln)
                        if kind != "module":
                            out[sid] = path
    return out


def suggest(index: CoChangeIndex, visible: set[str], seeds: list[str], k: int) -> list[tuple[str, float, dict]]:
    """Rank untouched symbols by summed co-change confidence with the seeds."""
    score: dict[str, float] = defaultdict(float)
    why: dict[str, dict[str, int]] = defaultdict(dict)
    for s in seeds:
        n = index.support(s, visible)
        for c, w in index.neighbours(s, visible).items():
            if c not in seeds:
                score[c] += w / (n + 2)
                why[c][s] = int(w)
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [(c, sc, why[c]) for c, sc in ranked]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("-k", type=int, default=10)
    n = sub.add_parser("neighbors"); n.add_argument("symbol"); n.add_argument("-k", type=int, default=10)
    e = sub.add_parser("export"); e.add_argument("-o", required=True); e.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()
    index, visible = load_index(args.repo)
    if args.cmd == "export":
        edges = index.edges(visible, args.min_count)
        nodes = sorted({x for a, b, _ in edges for x in (a, b)})
        graph = {"directed": False, "multigraph": False,
                 "graph": {"edge_type": "co_changed", "min_count": args.min_count,
                           "head": git(args.repo, "rev-parse", "HEAD").strip()},
                 "nodes": [{"id": x, "name": x} for x in nodes],
                 "edges": [{"source": a, "target": b, "type": "co_changed", "weight": w}
                           for a, b, w in sorted(edges, key=lambda t: -t[2])]}
        json.dump(graph, open(args.o, "w"))
        print(f"{len(edges)} co-change edges over {len(nodes)} symbols -> {args.o}")
        return
    if args.cmd == "neighbors":
        seeds = [args.symbol]
        if args.symbol not in index.by_symbol:
            close = [s for s in index.by_symbol if s.endswith("." + args.symbol.rsplit(".", 1)[-1])][:5]
            print(f"No multi-symbol history for {args.symbol}." + (f" Did you mean: {', '.join(close)}" if close else ""))
            return
    else:
        touched = diff_symbols(args.repo)
        seeds = sorted(touched)
        if not seeds:
            print("No uncommitted changes to existing functions or classes; nothing to check.")
            return
        print(f"Your diff edits {len(seeds)} symbol(s): " + ", ".join(seeds[:8]) + (" ..." if len(seeds) > 8 else ""))
    res = suggest(index, visible, seeds, args.k)
    if not res:
        print("History shows nothing that usually changes together with these symbols.")
        return
    print("Historically edited in the same commits" + (" but not in your diff:" if args.cmd == "check" else ":"))
    for sym, _, why in res:
        ev = "; ".join(f"{cnt}x with {s.rsplit('.', 1)[-1]}" for s, cnt in sorted(why.items(), key=lambda kv: -kv[1])[:3])
        print(f"  {sym}   ({ev})")
    print(f"[{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
