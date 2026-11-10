"""Download the competition files this study needs (everything except the
21 GB of snapshots, plus the smallest snapshot per repo to inspect git history).

Usage: .venv/bin/python src/download.py [--all-snapshots]
Requires `kaggle auth login` (or ~/.kaggle/kaggle.json) and having joined
https://www.kaggle.com/competitions/gemma-4-developer-agent
"""
import json
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "kaggle"
COMP = "gemma-4-developer-agent"
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")


def fetch(name: str) -> str:
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 0:
        return f"skip {name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [KAGGLE, "competitions", "download", "-c", COMP, "-f", name, "-p", str(dest.parent), "-q"],
        capture_output=True, text=True,
    )
    z = dest.parent / (dest.name + ".zip")
    if z.exists():  # the API zips some files
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest.parent)
        z.unlink()
    if not dest.exists():
        return f"FAIL {name}: {(r.stdout + r.stderr).strip()[-200:]}"
    return f"ok   {name}"


def main() -> None:
    rows = json.load(open(OUT / "_filelist.json"))
    names = [r["name"] for r in rows if not r["name"].startswith("snapshots/")]
    snaps = [r for r in rows if r["name"].startswith("snapshots/")]
    if "--all-snapshots" in sys.argv:
        names += [r["name"] for r in snaps]
    else:
        smallest = {}
        for r in snaps:
            repo = Path(r["name"]).name.split("_")[0]
            if repo not in smallest or int(r["size"]) < int(smallest[repo]["size"]):
                smallest[repo] = r
        names += [r["name"] for r in smallest.values()]
    # small files first so tasks.jsonl and the README land immediately
    size = {r["name"]: int(r["size"]) for r in rows}
    names.sort(key=lambda n: size[n])
    fails = 0
    with ThreadPoolExecutor(8) as ex:
        for i, msg in enumerate(ex.map(fetch, names), 1):
            if msg.startswith("FAIL"):
                fails += 1
                print(msg, flush=True)
            elif i % 50 == 0:
                print(f"{i}/{len(names)}", flush=True)
    print(f"done: {len(names)} files, {fails} failed", flush=True)


if __name__ == "__main__":
    main()
