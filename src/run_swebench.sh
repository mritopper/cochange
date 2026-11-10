#!/bin/bash
# Wait for the repository clones, then mine history and run Experiment 2.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "git clone -q https://github.com" >/dev/null; do sleep 15; done
echo "clones finished: $(date)"
for d in data/repos/*/; do
  git -C "$d" rev-parse HEAD >/dev/null 2>&1 || echo "BROKEN CLONE: $d"
done
.venv/bin/python -W ignore src/mine_swebench.py 2>&1 | grep -v Warning
echo "mining finished: $(date)"
.venv/bin/python -W ignore src/eval_issues.py > results/issues.txt 2>&1
echo "eval finished: $(date)"
grep -v Warning results/issues.txt | tail -40
