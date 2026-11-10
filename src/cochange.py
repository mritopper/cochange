"""Co-change index, rank fusion and metrics shared by all experiments."""
from __future__ import annotations

import json
import random
import subprocess
from collections import Counter, defaultdict

MAX_TRAIN = 30  # commits editing more symbols than this carry no co-change evidence


def load_history(path: str) -> list[dict]:
    """Commits in the order written by mine_history.py (parents before children)."""
    return [json.loads(l) for l in open(path)]


def code_symbols(commit: dict, include_modules: bool = False) -> list[str]:
    return [s["id"] for s in commit["symbols"]
            if not s["test"] and (include_modules or s["kind"] != "module")]


def test_symbols(commit: dict) -> list[str]:
    return [s["id"] for s in commit["symbols"] if s["test"] and s["kind"] != "module"]


def ancestors(repo: str, rev: str) -> set[str]:
    out = subprocess.run(["git", "-C", repo, "rev-list", rev], capture_output=True, text=True).stdout
    return set(out.split())


class CoChangeIndex:
    """Inverted index symbol -> commits, queried under a visibility filter.

    `visible` is the set of commit shas the caller is allowed to see (the
    ancestors of the snapshot being queried), which makes every query
    time-respecting by construction.
    """

    def __init__(self, commits: list[dict], include_modules: bool = False):
        self.sha = [c["sha"] for c in commits]
        self.time = [c["time"] for c in commits]
        self.code = [code_symbols(c, include_modules) for c in commits]
        self.by_symbol: dict[str, list[int]] = defaultdict(list)   # multi-symbol commits
        self.touched: dict[str, list[int]] = defaultdict(list)     # every commit <= MAX_TRAIN
        for i, syms in enumerate(self.code):
            if len(syms) <= MAX_TRAIN:
                for s in syms:
                    self.touched[s].append(i)
                    if len(syms) >= 2:
                        self.by_symbol[s].append(i)

    def change_count(self, symbol: str, visible: set[str]) -> int:
        """How many visible commits edited `symbol` (a 'hot spot' prior)."""
        return sum(1 for i in self.touched.get(symbol, ()) if self.sha[i] in visible)

    def support(self, symbol: str, visible: set[str]) -> int:
        return sum(1 for i in self.by_symbol.get(symbol, ()) if self.sha[i] in visible)

    def neighbours(self, seed: str, visible: set[str], now: int | None = None,
                   half_life_days: float | None = None) -> dict[str, float]:
        """Weighted co-change counts for `seed` over visible commits.

        Each shared commit counts 1; with `half_life_days`, a commit's weight
        halves for every half-life of age relative to `now`.
        """
        score: dict[str, float] = Counter()
        for i in self.by_symbol.get(seed, ()):
            if self.sha[i] not in visible:
                continue
            w = 1.0
            if half_life_days and now:
                age = max(now - self.time[i], 0) / 86400
                w = 0.5 ** (age / half_life_days)
            for s in self.code[i]:
                if s != seed:
                    score[s] += w
        return score

    def rank(self, seed: str, visible: set[str], universe: set[str] | None = None, **kw) -> list[str]:
        sc = self.neighbours(seed, visible, **kw)
        return [s for s, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))
                if universe is None or s in universe]

    def edges(self, visible: set[str], min_count: int = 2) -> list[tuple[str, str, int]]:
        """Undirected co-change edges (a < b) with support >= min_count."""
        pair: Counter = Counter()
        for i, syms in enumerate(self.code):
            if self.sha[i] in visible and 2 <= len(syms) <= MAX_TRAIN:
                ss = sorted(set(syms))
                for x in range(len(ss)):
                    for y in range(x + 1, len(ss)):
                        pair[(ss[x], ss[y])] += 1
        return [(a, b, n) for (a, b), n in pair.items() if n >= min_count]


def rrf(lists: list[list[str]], k: int = 60, limit: int = 200,
        weights: list[float] | None = None) -> list[str]:
    """(Weighted) reciprocal rank fusion (Cormack et al., 2009)."""
    score: dict[str, float] = defaultdict(float)
    for j, lst in enumerate(lists):
        w = weights[j] if weights else 1.0
        for r, s in enumerate(lst[:limit]):
            score[s] += w / (k + r + 1)
    return [s for s, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for i, s in enumerate(ranked):
        if s in gold:
            return 1.0 / (i + 1)
    return 0.0


def paired_bootstrap(a: list[float], b: list[float], groups: list, n: int = 10000, seed: int = 0):
    """Mean difference a-b with a 95% CI, resampling whole groups (commits/tasks)."""
    by = defaultdict(list)
    for x, y, g in zip(a, b, groups):
        by[g].append(x - y)
    keys = list(by)
    rng = random.Random(seed)
    flat = [d for g in keys for d in by[g]]
    mean = sum(flat) / len(flat)
    stats = []
    for _ in range(n):
        tot = cnt = 0.0
        for _ in keys:
            d = by[keys[rng.randrange(len(keys))]]
            tot += sum(d)
            cnt += len(d)
        stats.append(tot / cnt)
    stats.sort()
    return mean, stats[int(0.025 * n)], stats[int(0.975 * n)]


def fmt_ci(t) -> str:
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"

