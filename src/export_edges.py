"""Export the released co-change edge files: edges/<repo>.json.gz (node-link JSON).

Uses every mined commit (see README for which revisions were mined), keeps
pairs co-edited at least twice, and writes the same node-link fields as
cochange_tool.py export.
"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cochange import CoChangeIndex, load_history  # noqa: E402
from eval_heldout import ALL_REPOS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIN_COUNT = 2

if __name__ == "__main__":
    out = ROOT / "edges"
    out.mkdir(exist_ok=True)
    te = tn = tc = 0
    for name in ALL_REPOS:
        path = ROOT / "data" / "history_swe" / f"{name}.jsonl"
        if not path.exists():
            path = ROOT / "data" / "history" / f"{name}.jsonl"
        hist = load_history(str(path))
        index = CoChangeIndex(hist)
        edges = sorted(index.edges({c["sha"] for c in hist}, MIN_COUNT), key=lambda t: (-t[2], t[0], t[1]))
        nodes = sorted({x for a, b, _ in edges for x in (a, b)})
        graph = {"directed": False, "multigraph": False,
                 "graph": {"repo": name, "edge_type": "co_changed", "min_count": MIN_COUNT, "commits_mined": len(hist)},
                 "nodes": [{"id": x, "name": x} for x in nodes],
                 "edges": [{"source": a, "target": b, "type": "co_changed", "weight": w} for a, b, w in edges]}
        with gzip.GzipFile(out / f"{name}.json.gz", "wb", mtime=0) as f:
            f.write(json.dumps(graph).encode())
        print(f"{name:14s} commits {len(hist):6d}  edges {len(edges):6d}  symbols {len(nodes):6d}")
        te, tn, tc = te + len(edges), tn + len(nodes), tc + len(hist)
    print(f"TOTAL commits {tc}  edges {te}  symbols {tn}")
