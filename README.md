# cochange

Edit history as an edge type for code graphs: which functions usually change together, mined from `git log`, evaluated against same-file and static-graph neighbours, and packaged as a tool a coding agent can call on its own diff.

Paper: [paper/paper.md](paper/paper.md) (entry for the Kaggle "Gemma 4 Developer Agent Paper Track").

## Use the tool

Standard library only. Copy `src/cochange_tool.py`, `src/cochange.py` and `src/mine_history.py` anywhere and run from inside a git checkout:

```bash
python cochange_tool.py check                 # what usually changes with my uncommitted diff?
python cochange_tool.py neighbors pkg.mod.Class.method -k 10
python cochange_tool.py export -o edges.json  # node-link JSON, type "co_changed"
```

The first call mines history once (seconds to a few minutes) and caches it under `.git/cochange/`.

## Released edges

`edges/<repo>.json.gz` holds co-change edges for 15 Python repositories: 249,739 edges (co-edited in at least two commits) over 61,238 functions, methods and classes. Each file is node-link JSON: `nodes[].id` is the fully qualified symbol path, and each edge has `source`, `target`, `type: "co_changed"` and an integer `weight` (number of commits that edited both).

```python
import gzip, json, networkx as nx
g = json.load(gzip.open("edges/httpx.json.gz"))
G = nx.Graph()
G.add_weighted_edges_from((e["source"], e["target"], e["weight"]) for e in g["edges"])
```

`fastapi`, `rich` and `httpx` were mined at the default branch in November 2026; the other twelve were mined over the history reachable from their SWE-bench Verified base commits. `src/export_edges.py` regenerates the files.

## Reproduce the paper

Needs Python 3.12, git, about 3 GB of disk for clones. Everything except the Gemma run is CPU-only.

```bash
python -m venv .venv && .venv/bin/pip install pandas pyarrow
# 1. data: SWE-bench Verified task list and the 15 repositories
mkdir -p data/swebench data/repos
curl -L -o data/swebench/verified.parquet \
  https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/main/data/test-00000-of-00001.parquet
for r in fastapi/fastapi Textualize/rich encode/httpx psf/requests django/django sympy/sympy \
         sphinx-doc/sphinx matplotlib/matplotlib scikit-learn/scikit-learn astropy/astropy \
         pydata/xarray pytest-dev/pytest pylint-dev/pylint mwaskom/seaborn pallets/flask; do
  git clone https://github.com/$r.git data/repos/$(basename $r)
done
# 2. mine history (about 10 minutes)
.venv/bin/python src/mine_swebench.py
for r in fastapi rich httpx; do .venv/bin/python src/mine_history.py data/repos/$r data/history/$r.jsonl; done
# 3. experiments
.venv/bin/python src/eval_issues.py  > results/issues.txt     # Sections 4.1 and 4.2 (about 3 minutes)
.venv/bin/python src/eval_heldout.py > results/heldout.txt    # Section 4.3 (about 10 minutes)
# 4. Gemma 4 E4B on Apple silicon (about 70 minutes)
.venv/bin/pip install mlx-lm
.venv/bin/python src/gemma_select.py && .venv/bin/python src/gemma_report.py > results/gemma_report.txt
# 5. figure (needs matplotlib)
.venv/bin/pip install matplotlib && .venv/bin/python src/make_figures.py
```

Results depend on the state of each repository's default branch only through `eval_heldout.py` for `fastapi`, `rich` and `httpx` (mined at HEAD); the SWE-bench experiments read history reachable from each task's `base_commit` and are stable.

## Files

| File | What it does |
|---|---|
| `src/mine_history.py` | `git log` to one JSON line per commit listing the functions, methods and classes it edited |
| `src/cochange.py` | time-respecting co-change index, rank fusion, metrics, paired bootstrap |
| `src/static_graph.py` | name-resolved static graph and BM25 token tables for any commit |
| `src/gold.py` | reference patch to gold symbols in the pre-fix snapshot |
| `src/eval_issues.py` | issue localization and edit completion on SWE-bench Verified |
| `src/eval_heldout.py` | edit completion on held-out commits of 15 repositories |
| `src/gemma_select.py`, `src/gemma_report.py` | small-model selection experiment |
| `src/make_figures.py` | Figure 1 from `results/heldout.json` |
| `src/export_edges.py` | writes `edges/*.json.gz` from the mined history |
| `src/cochange_tool.py` | the agent-facing tool |
| `src/download.py`, `src/wait_and_download.sh` | optional: fetch the competition's released graphs (requires joining the main competition) |

## License

Apache 2.0.
