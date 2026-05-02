from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_agent_baseline.eval.scorer import ColumnScorer


@pytest.fixture
def scorer():
    return ColumnScorer(lambda_penalty=0.1)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _write_csv(path: Path, headers: list[str], rows: list[list]):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)


def test_exact_match(scorer, tmp_dir):
    pred = tmp_dir / "pred.csv"
    gt = tmp_dir / "gt.csv"
    _write_csv(pred, ["a", "b"], [["1", "2"], ["3", "4"]])
    _write_csv(gt, ["a", "b"], [["1", "2"], ["3", "4"]])
    result = scorer.score(pred, gt)
    assert result["score"] == 1.0
    assert result["recall"] == 1.0


def test_column_order_irrelevant(scorer, tmp_dir):
    pred = tmp_dir / "pred.csv"
    gt = tmp_dir / "gt.csv"
    _write_csv(pred, ["b", "a"], [["2", "1"], ["4", "3"]])
    _write_csv(gt, ["a", "b"], [["1", "2"], ["3", "4"]])
    result = scorer.score(pred, gt)
    assert result["score"] == 1.0


def test_extra_column_penalty(scorer, tmp_dir):
    pred = tmp_dir / "pred.csv"
    gt = tmp_dir / "gt.csv"
    _write_csv(pred, ["a", "b", "c"], [["1", "2", "5"], ["3", "4", "6"]])
    _write_csv(gt, ["a", "b"], [["1", "2"], ["3", "4"]])
    result = scorer.score(pred, gt)
    assert result["recall"] == 1.0
    assert 0.85 <= result["score"] < 1.0


def test_missing_column(scorer, tmp_dir):
    pred = tmp_dir / "pred.csv"
    gt = tmp_dir / "gt.csv"
    _write_csv(pred, ["a"], [["1"], ["3"]])
    _write_csv(gt, ["a", "b"], [["1", "2"], ["3", "4"]])
    result = scorer.score(pred, gt)
    assert result["recall"] == 0.5


def test_empty_ground_truth(scorer, tmp_dir):
    pred = tmp_dir / "pred.csv"
    gt = tmp_dir / "gt.csv"
    _write_csv(pred, ["a"], [["1"]])
    _write_csv(gt, [], [])
    result = scorer.score(pred, gt)
    assert result["score"] == 0.0

def test_null_normalization(scorer, tmp_dir):
    pred = tmp_dir / "pred.csv"
    gt = tmp_dir / "gt.csv"
    _write_csv(pred, ["a"], [["NULL"]])
    _write_csv(gt, ["a"], [["null"]])
    result = scorer.score(pred, gt)
    assert result["score"] == 1.0

def test_numeric_rounding(scorer, tmp_dir):
    pred = tmp_dir / "pred.csv"
    gt = tmp_dir / "gt.csv"
    _write_csv(pred, ["v"], [["3.14159"]])
    _write_csv(gt, ["v"], [["3.14"]])
    result = scorer.score(pred, gt)
    assert result["score"] == 1.0
