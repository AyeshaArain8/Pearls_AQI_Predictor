"""Compatibility entry point for building canonical training features.

Use this instead of the retired AOD/dust/UV preprocessing flow.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feature_contract import make_feature_rows
from src.feature_store import load_observations

def main():
    rows = make_feature_rows(load_observations(), include_targets=True)
    output = ROOT / "data" / "processed" / "forecast_dataset.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False)
    print(f"Wrote {len(rows)} canonical Lahore training rows to {output}")

if __name__ == "__main__": main()
