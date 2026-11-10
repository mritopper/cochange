"""Derive gold localization labels: which existing symbols does a patch edit?

A patch line is attributed to the innermost function / method / class that
contains it in the PRE-fix file (the code an agent actually sees). Pure
insertions are attributed to the symbol that contains the insertion point;
insertions between top-level definitions (brand-new functions or classes)
have no pre-fix symbol and are reported separately as `new_code`.
"""
from __future__ import annotations

import re
import subprocess

from mine_history import SymbolTable, is_test_path, module_name

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def blob_at(repo: str, commit: str, path: str) -> str | None:
    r = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True)
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def patch_old_lines(patch: str) -> dict[str, dict]:
    """-> {path: {"removed": [old line numbers], "inserted_after": [(old line, text)]}}"""
    files: dict[str, dict] = {}
    cur = None
    old = 0
    in_hunk = False
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            cur, in_hunk = None, False
        elif line.startswith("--- ") and not in_hunk:
            path = line[4:].strip()
            cur = None if path == "/dev/null" else files.setdefault(
                path[2:] if path.startswith(("a/", "b/")) else path, {"removed": [], "inserted_after": []})
        elif line.startswith("+++ ") and not in_hunk:
            continue
        elif line.startswith("@@"):
            m = HUNK.match(line)
            if m:
                old, in_hunk = int(m[1]), True
        elif in_hunk and cur is not None:
            if line.startswith("-"):
                cur["removed"].append(old)
                old += 1
            elif line.startswith("+"):
                cur["inserted_after"].append((old - 1, line[1:]))   # sits before old line `old`
            elif line.startswith("\\"):
                continue
            else:
                old += 1
    return files


def gold_symbols(repo: str, base_commit: str, patch: str) -> dict:
    """-> {"symbols": {id: kind}, "files": [...], "new_code": bool, "module_level": bool}"""
    out = {"symbols": {}, "files": [], "new_code": False, "module_level": False}
    for path, ch in patch_old_lines(patch).items():
        if not path.endswith(".py") or is_test_path(path):
            continue
        src = blob_at(repo, base_commit, path)
        if src is None:
            out["new_code"] = True
            continue
        out["files"].append(path)
        table = SymbolTable(src, module_name(path))
        for ln in ch["removed"]:
            sid, kind = table.lookup(ln)
            if kind == "module":
                out["module_level"] = True
            else:
                out["symbols"][sid] = kind
        lines = src.splitlines()
        starts = {sid: s for sid, s, _, _ in table.all_symbols()}

        def indent(text: str) -> int:
            return len(text) - len(text.lstrip())

        for ln, text in ch["inserted_after"]:
            if not text.strip():
                continue
            before, after = table.lookup(max(ln, 1)), table.lookup(ln + 1)
            if before == after:
                if before[1] == "module":
                    out["module_level"] = True
                else:
                    out["symbols"][before[0]] = before[1]
                continue
            # Insertion on a boundary between two symbols. Deeper-indented text
            # than the preceding definition's header continues that definition.
            if before[1] != "module":
                head = lines[starts[before[0]] - 1] if starts.get(before[0]) else ""
                if indent(text) > indent(head):
                    out["symbols"][before[0]] = before[1]
                    continue
            if indent(text) > 0:
                # new member at class level: the enclosing class is what gets edited
                owner = next((x[0].rsplit(".", 1)[0] if x[1] == "method" else x[0]
                              for x in (before, after) if x[1] in ("method", "class")), None)
                if owner:
                    out["symbols"][owner] = "class"
                    continue
            if text.lstrip().startswith(("def ", "class ", "async def ", "@")):
                out["new_code"] = True
            else:
                out["module_level"] = True
    return out
