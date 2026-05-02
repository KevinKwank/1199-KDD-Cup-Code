from __future__ import annotations

import csv
from pathlib import Path


class ColumnScorer:
    def __init__(self, lambda_penalty: float = 0.1):
        self.lambda_penalty = lambda_penalty

    def _normalize_value(self, v) -> str:
        if v is None or str(v).strip().lower() in ("null", "nan", "none", ""):
            return ""
        try:
            return str(round(float(v), 2))
        except (ValueError, TypeError):
            return str(v).strip()

    def _column_signature(self, values: list) -> tuple:
        return tuple(sorted(self._normalize_value(v) for v in values))

    def score(self, prediction_path: Path, ground_truth_path: Path) -> dict:
        pred_cols, pred_rows = self._load_csv(prediction_path)
        gt_cols, gt_rows = self._load_csv(ground_truth_path)

        if not gt_cols:
            return {"recall": 0.0, "score": 0.0, "details": "empty ground truth"}

        gt_sigs: dict[tuple, int] = {}
        for i in range(len(gt_cols)):
            values = [row[i] if i < len(row) else "" for row in gt_rows]
            sig = self._column_signature(values)
            gt_sigs[sig] = gt_sigs.get(sig, 0) + 1

        pred_sigs: dict[tuple, int] = {}
        for i in range(len(pred_cols)):
            values = [row[i] if i < len(row) else "" for row in pred_rows]
            sig = self._column_signature(values)
            pred_sigs[sig] = pred_sigs.get(sig, 0) + 1

        matched = 0
        remaining = dict(pred_sigs)
        for sig, count in gt_sigs.items():
            if sig in remaining:
                m = min(count, remaining[sig])
                matched += m
                remaining[sig] -= m
                if remaining[sig] == 0:
                    del remaining[sig]

        recall = matched / len(gt_sigs) if gt_sigs else 0.0
        extra = sum(remaining.values())
        total_pred = len(pred_sigs) if pred_sigs else 1
        score = max(0.0, recall - self.lambda_penalty * (extra / total_pred))

        return {
            "recall": round(recall, 4),
            "score": round(score, 4),
            "matched": matched,
            "gt_cols": len(gt_sigs),
            "pred_cols": len(pred_sigs),
            "extra": extra,
        }

    def _load_csv(self, path: Path) -> tuple[list[str], list[list]]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            rows = [row for row in reader]
        return headers, rows
