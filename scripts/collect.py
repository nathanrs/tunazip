import json
import subprocess
import sys
from pathlib import Path

RECORDS = Path(".cache/records")
LEDGER = Path("results/ledger.jsonl")


def collect():
    subprocess.run(["modal", "volume", "get", "--force", "tunazip", "records", str(RECORDS.parent)],
                   check=True, stdout=subprocess.DEVNULL)
    known = set()
    if LEDGER.exists():
        known = {json.loads(line)["record"] for line in LEDGER.read_text().splitlines()}
    added = 0
    with LEDGER.open("a") as ledger:
        for path in sorted(RECORDS.glob("*.json"), key=lambda p: p.stat().st_mtime):
            if path.stem in known:
                continue
            ledger.write(json.dumps({"record": path.stem, **json.loads(path.read_text())}) + "\n")
            added += 1
    return added


if __name__ == "__main__":
    print(collect(), "records added", file=sys.stderr)
