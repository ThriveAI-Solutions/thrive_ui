"""Render an eval results JSON into a self-contained HTML report.

Usage:
    uv run python scripts/generate_eval_report.py evals/results/<run_id>.json
    uv run python scripts/generate_eval_report.py results.json --out report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.report import generate_html


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    out = args.out or args.results.with_suffix(".html")
    out.write_text(generate_html(results))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
