from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

def test_file_exists():
    csvs = list(RAW.glob("*.csv"))
    assert csvs, "No raw CSVs generated"

def test_labels():
    df = pd.read_csv(next(RAW.glob("*.csv")))
    assert set(df["label"].unique()).issubset({-1, 0, 1})
