"""Tests for data ingestion utilities."""

import types
import sys

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

# ``data_ingest`` depends on MetaTrader5 at import time. Provide a dummy module
# so the import succeeds even when the real package is unavailable.
sys.modules.setdefault("MetaTrader5", types.SimpleNamespace())

from src.data.data_ingest import label_tp_sl

def test_label_tp_sl_deterministic():
    """Ensure ``label_tp_sl`` produces expected labels for simple data."""

    df = pd.DataFrame(
        {
            "open": [10, 10, 10, 10, 10],
            "high": [10, 12, 10, 12, 10],
            "low": [10, 10, 8, 10, 9.5],
        }
    )

    result = label_tp_sl(df.copy(), point=0.1, tp_pips=1, sl_pips=1, lookahead=2)

    assert result["label"].tolist() == [1, 0, 1, -1, -1]


def test_label_tp_sl_roundtrip(tmp_path):
    """Write labeled data to CSV and ensure file exists and round-trips."""

    df = pd.DataFrame(
        {
            "open": [1, 1, 1],
            "high": [1.1, 1.2, 1],
            "low": [0.9, 1, 0.8],
        }
    )

    labeled = label_tp_sl(df.copy(), point=0.1, tp_pips=1, sl_pips=1, lookahead=1)
    out_file = tmp_path / "labeled.csv"
    labeled.to_csv(out_file, index=False)

    assert out_file.exists()
    df2 = pd.read_csv(out_file)
    assert df2.equals(labeled)

