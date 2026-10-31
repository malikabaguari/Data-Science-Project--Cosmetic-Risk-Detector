"""Entry point for the deterministic cleaning, EDA, and RF training pipeline.

A fresh run is deterministic but may take several minutes because grouped CV,
Random Forest tuning, grouped OOF threshold selection, and scope sensitivity
are all executed from the supplied data snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import save_eda
from .cleaning import clean_data, save_clean
from .experiments import run_experiments


def _log(root: Path, message: str) -> None:
    root.joinpath("reports").mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with (root / "reports" / "training_progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main(root: Path) -> None:
    root = Path(root).resolve()
    _log(root, "Loading supplied labeled snapshot; no labels will be generated, masked, or altered.")
    original = pd.read_csv(root / "data/processed/products_labeled.csv", dtype={"code": "string"}, low_memory=False)
    data, quality, rejected = clean_data(original)
    _log(root, f"Cleaning completed: {len(data)} retained rows; {int(data['has_risk'].sum())} supplied positive labels.")
    save_clean(data, quality, rejected, root)
    _log(root, "Saving descriptive EDA artifacts.")
    save_eda(data, original, root)
    _log(root, "Starting grouped Random Forest experiment workflow.")
    run_experiments(data, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    main(Path(args.root))
