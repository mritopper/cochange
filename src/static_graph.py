"""Repository-wide symbol tables and a light, name-resolved static graph.

This is our stand-in for the released code graphs when evaluating on commits
outside the competition data. Nodes: top-level functions, classes, methods.
Edges: class-contains-method, calls (callee resolved by short name, skipped
when more than 5 symbols share the name) and inheritance. All edges are
treated as undirected for neighbourhood queries.
"""
from __future__ import annotations

import ast
import keyword
import re
import subprocess
from collections import Counter, OrderedDict, defaultdict

from mine_history import is_test_path, module_name

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
STOP = set(keyword.kwlist) | {"self", "cls", "the", "and", "for", "that", "this", "with", "not", "are",
                              "was", "but", "from", "have", "has", "can", "should", "would", "when", "it",
                              "is", "in", "to", "of", "a", "an", "be", "on", "as", "if", "or", "by", "at"}
SKIP_DIRS = ("docs/", "doc/", "docs_src/", "examples/", "benchmarks/", "asv_bench/")


def tokenize(text: str) -> list[str]:
    """Identifier-aware tokens: whole identifiers plus snake/camel parts."""
    out = []
    for ident in IDENT.findall(text):
        low = ident.lower()
        parts = [p.lower() for chunk in ident.split("_") for p in CAMEL.findall(chunk)]
        if len(parts) > 1 and len(low) > 2:
            out.append(low)
        out += [p for p in parts if len(p) > 1 and p not in STOP and not p.isdigit()]
    return out


class Snapshots:
    """Symbols, their text tokens and static adjacency at any commit."""

    def __init__(self, repo: str, with_text: bool = False, cache_size: int = 12000):
        self.repo = repo
        self.with_text = with_text
        self.cache_size = cache_size
        self.cat = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.blob_cache: OrderedDict[tuple[str, str], dict] = OrderedDict()

    def _blob(self, sha: str) -> str:
        self.cat.stdin.write((sha + "\n").encode())
        self.cat.stdin.flush()
        head = self.cat.stdout.readline().split()
        data = self.cat.stdout.read(int(head[2]) + 1)[:-1]
        return data.decode("utf-8", "replace")

    def _parse(self, sha: str, path: str) -> dict:
        """-> {symbol_id: {"line", "end", "calls", "bases", "tokens"}}"""
        key = (sha, path)
        hit = self.blob_cache.get(key)
        if hit is not None:
            self.blob_cache.move_to_end(key)
            return hit
        mod = module_name(path)
        out: dict[str, dict] = {}
        src = self._blob(sha)
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            tree = None
        if tree is not None:
            lines = src.splitlines()

            def names(node):
                s = set()
                for n in ast.walk(node):
                    if isinstance(n, ast.Call):
                        f = n.func
                        if isinstance(f, ast.Name):
                            s.add(f.id)
                        elif isinstance(f, ast.Attribute):
                            s.add(f.attr)
                return s

            def start(node):
                return min([node.lineno] + [d.lineno for d in node.decorator_list])

            def add(sid, node, calls, bases, skip=()):
                info = {"line": node.lineno, "end": node.end_lineno, "calls": calls, "bases": bases}
                if self.with_text:
                    skipped = set()
                    for a, b in skip:
                        skipped.update(range(a, b + 1))
                    body = "\n".join(lines[i - 1] for i in range(start(node), node.end_lineno + 1)
                                     if i not in skipped)
                    toks = Counter(tokenize(body))
                    for t in tokenize(sid.replace(".", " ")):   # the name counts extra
                        toks[t] += 2
                    info["tokens"] = toks
                    info["len"] = sum(toks.values())
                out[sid] = info

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(f"{mod}.{node.name}", node, names(node), set())
                elif isinstance(node, ast.ClassDef):
                    bases = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases}
                    methods = [s for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    # a class's own text is everything outside its methods
                    add(f"{mod}.{node.name}", node, set(), bases - {""},
                        skip=[(start(m), m.end_lineno) for m in methods])
                    for sub in methods:
                        add(f"{mod}.{node.name}.{sub.name}", sub, names(sub), set())
        self.blob_cache[key] = out
        if len(self.blob_cache) > self.cache_size:
            self.blob_cache.popitem(last=False)
        return out

    def at(self, commit: str):
        """-> (symbols, file_of, adjacency) for non-test code in the tree at `commit`."""
        ls = subprocess.run(["git", "-C", self.repo, "ls-tree", "-r", commit],
                            capture_output=True, text=True).stdout
        syms: dict[str, dict] = {}
        file_of: dict[str, str] = {}
        for line in ls.splitlines():
            meta, path = line.split("\t", 1)
            if not path.endswith(".py") or is_test_path(path) or path.startswith(SKIP_DIRS):
                continue
            for sid, info in self._parse(meta.split()[2], path).items():
                syms[sid] = info
                file_of[sid] = path
        short = defaultdict(list)
        for sid in syms:
            short[sid.rsplit(".", 1)[-1]].append(sid)
        adj: dict[str, set[str]] = defaultdict(set)

        def link(a, b):
            if a != b:
                adj[a].add(b)
                adj[b].add(a)

        for sid, info in syms.items():
            parent = sid.rsplit(".", 1)[0]
            if parent in syms:                      # class contains method
                link(sid, parent)
            for name in info["calls"] | info["bases"]:
                targets = short.get(name, ())
                if 0 < len(targets) <= 5:           # skip hopelessly ambiguous names
                    for t in targets:
                        link(sid, t)
        return syms, file_of, adj
