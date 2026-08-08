"""Compatibility command: fetch canonical historical training rows from Feast."""
from pathlib import Path
from src.feature_contract import make_feature_rows
from src.feature_store import historical_features
def main():
    rows=make_feature_rows(historical_features(),include_targets=True)
    print(f"Feast returned {len(rows)} canonical Lahore training rows.")
if __name__=="__main__": main()
