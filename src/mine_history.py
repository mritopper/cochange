"""Mine symbol-level change history from a git repository.

For every non-merge commit that touches Python files, record which
functions / methods / classes / modules were edited. The output (one JSON
line per commit) is the raw material for co-change edges.

Usage: python src/mine_history.py <repo_dir> <out.jsonl> [--rev <commit>]

Only history reachable from --rev (default HEAD) is read, so pointing --rev at
a task's base_commit yields a history that cannot leak the future fix.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from bisect import bisect_right
from pathlib import Path

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DEF = re.compile(r"^(\s*)(?:async\s+def|def|class)\s+([A-Za-z_]\w*)")
NULL = "0" * 40


def module_name(path: str) -> str:
    p = path[:-3] if path.endswith(".py") else path
    parts = [x for x in p.split("/") if x]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
        or "/tests/" in f"/{path}" or "/test/" in f"/{path}"
    )


class SymbolTable:
    """Maps a line number to the innermost module/class/function/method symbol.

    Granularity matches the released code graphs: modules, classes, and
    functions or methods. Nested functions are attributed to their outermost
    enclosing function.
    """

    def __init__(self, src: str, module: str):
        self.module = module
        self.spans: list[tuple[int, int, str, str]] = []  # (start, end, qualname, kind)
        try:
            tree = ast.parse(src)
            self._from_ast(tree)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            self._from_indent(src)
        self.spans.sort()
        self._starts = [s[0] for s in self.spans]

    def _from_ast(self, tree: ast.AST) -> None:
        def start(node):
            decos = getattr(node, "decorator_list", [])
            return min([node.lineno] + [d.lineno for d in decos])

        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.spans.append((start(node), node.end_lineno, node.name, "function"))
            elif isinstance(node, ast.ClassDef):
                self.spans.append((start(node), node.end_lineno, node.name, "class"))
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.spans.append(
                            (start(sub), sub.end_lineno, f"{node.name}.{sub.name}", "method")
                        )

    def _from_indent(self, src: str) -> None:
        """Fallback for files Python 3 cannot parse (e.g. Python 2 history)."""
        lines = src.splitlines()
        heads = []  # (lineno, indent, name)
        for i, line in enumerate(lines, 1):
            m = DEF.match(line)
            if m:
                heads.append((i, len(m.group(1).expandtabs(4)), m.group(2), line.lstrip().startswith("class")))
        n = len(lines)
        for idx, (ln, ind, name, is_cls) in enumerate(heads):
            if ind > 4:
                continue
            end = n
            for ln2, ind2, _, _ in heads[idx + 1:]:
                if ind2 <= ind:
                    end = ln2 - 1
                    break
            if ind == 0:
                self.spans.append((ln, end, name, "class" if is_cls else "function"))
            elif not is_cls:
                owner = next(
                    (h for h in reversed(heads[:idx]) if h[1] == 0), None
                )
                if owner and owner[3]:
                    self.spans.append((ln, end, f"{owner[2]}.{name}", "method"))

    def lookup(self, line: int) -> tuple[str, str]:
        """Return (symbol_id, kind) for the innermost symbol covering `line`."""
        best = None
        i = bisect_right(self._starts, line) - 1
        # walk back over candidates that start at or before `line`
        while i >= 0:
            s, e, q, k = self.spans[i]
            if s <= line <= e:
                if best is None or (e - s) < (best[1] - best[0]):
                    best = (s, e, q, k)
                if k != "method":  # a top-level span cannot nest in another
                    break
            i -= 1
        if best is None:
            return self.module, "module"
        return f"{self.module}.{best[2]}", best[3]

    def all_symbols(self) -> list[tuple[str, int, int, str]]:
        return [(f"{self.module}.{q}", s, e, k) for s, e, q, k in self.spans]


class Blobs:
    def __init__(self, repo: str):
        self.p = subprocess.Popen(
            ["git", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self.cache: dict[tuple[str, str], SymbolTable] = {}

    def table(self, sha: str, path: str) -> SymbolTable | None:
        key = (sha, path)
        if key in self.cache:
            return self.cache[key]
        self.p.stdin.write((sha + "\n").encode())
        self.p.stdin.flush()
        head = self.p.stdout.readline().split()
        if len(head) < 3 or head[1] != b"blob":
            return None
        data = self.p.stdout.read(int(head[2]) + 1)[:-1]
        t = SymbolTable(data.decode("utf-8", "replace"), module_name(path))
        if len(self.cache) > 4000:
            self.cache.clear()
        self.cache[key] = t
        return t

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def mine(repo: str, out: str, rev: str | list[str] = "HEAD") -> int:
    revs = [rev] if isinstance(rev, str) else list(rev)
    blobs = Blobs(repo)
    log = subprocess.Popen(
        ["git", "-C", repo, "log", *revs, "--no-merges", "--topo-order", "--reverse", "-M", "-p", "-U0",
         "--no-color", "--no-ext-diff", "--full-index", "--format=%x00%H %at", "--", "*.py"],
        stdout=subprocess.PIPE,
    )
    n = 0
    with open(out, "w") as fo:
        commit = None
        old_sha = new_sha = old_path = new_path = None
        told = tnew = None

        def flush():
            nonlocal n
            if commit and commit["symbols"]:
                commit["symbols"] = sorted(
                    [{"id": k, "kind": v[0], "file": v[1], "test": v[2]} for k, v in commit["symbols"].items()],
                    key=lambda d: d["id"],
                )
                commit["files"] = sorted(commit["files"])
                fo.write(json.dumps(commit) + "\n")
                n += 1

        for raw in log.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("\x00"):
                flush()
                sha, at = line[1:].split()
                commit = {"sha": sha, "time": int(at), "files": set(), "symbols": {}}
                continue
            if commit is None:
                continue
            if line.startswith("diff --git "):
                old_sha = new_sha = old_path = new_path = told = tnew = None
            elif line.startswith("index ") and ".." in line:
                a, b = line.split()[1].split("..")
                old_sha, new_sha = a, b
            elif line.startswith("--- "):
                old_path = None if line[4:] == "/dev/null" else line[6:]
            elif line.startswith("+++ "):
                new_path = None if line[4:] == "/dev/null" else line[6:]
                if old_path and old_sha and old_sha != NULL:
                    told = blobs.table(old_sha, old_path)
                if new_path and new_sha and new_sha != NULL:
                    tnew = blobs.table(new_sha, new_path)
                commit["files"].add(new_path or old_path)
            elif line.startswith("@@"):
                m = HUNK.match(line)
                if not m:
                    continue
                a, b, c, d = int(m[1]), int(m[2] or 1), int(m[3]), int(m[4] or 1)
                touched = []
                if told and b:
                    touched += [(told, ln, old_path) for ln in range(a, a + b)]
                if tnew and d:
                    touched += [(tnew, ln, new_path) for ln in range(c, c + d)]
                for table, ln, path in touched:
                    sid, kind = table.lookup(ln)
                    commit["symbols"].setdefault(sid, (kind, path, is_test_path(path)))
        flush()
    blobs.close()
    return n


if __name__ == "__main__":
    repo, out = sys.argv[1], sys.argv[2]
    rev = sys.argv[sys.argv.index("--rev") + 1] if "--rev" in sys.argv else "HEAD"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    print(f"{repo}: {mine(repo, out, rev)} commits with Python symbol changes")
