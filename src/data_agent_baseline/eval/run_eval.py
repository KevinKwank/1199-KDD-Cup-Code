from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from data_agent_baseline.config import load_app_config
from data_agent_baseline.eval.scorer import ColumnScorer
from data_agent_baseline.run.runner import run_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _find_ground_truth(dataset_root: Path, task_id: str) -> Path | None:
    candidates = [
        dataset_root / task_id / "prediction.csv",
        dataset_root / task_id / "gold.csv",
        dataset_root.parent / "output" / task_id / "gold.csv",
        dataset_root.parent / "output" / task_id / "prediction.csv",
        dataset_root.parent.parent / "output" / task_id / "gold.csv",
        dataset_root.parent.parent / "output" / task_id / "prediction.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/dev.yaml")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    config = load_app_config(config_path)
    logger.info(f"Running benchmark with config: {config_path}")
    if limit:
        logger.info(f"Limit: {limit} tasks")

    run_output_dir, artifacts = run_benchmark(config=config, limit=limit)

    scorer = ColumnScorer()
    results = []
    dataset = config.dataset

    for artifact in artifacts:
        pred_path = artifact.prediction_csv_path or run_output_dir / artifact.task_id / "prediction.csv"
        gt_path = _find_ground_truth(dataset.root_path, artifact.task_id)

        if not pred_path.exists():
            results.append({"task_id": artifact.task_id, "error": "prediction not found"})
            continue
        if gt_path is None:
            results.append({"task_id": artifact.task_id, "error": "ground truth not found"})
            continue

        try:
            score_result = scorer.score(pred_path, gt_path)
            results.append({
                "task_id": artifact.task_id,
                "score": score_result["score"],
                "recall": score_result["recall"],
                "succeeded": artifact.succeeded,
            })
        except Exception as exc:
            results.append({"task_id": artifact.task_id, "error": str(exc)})

    scores = [r["score"] for r in results if "score" in r]
    if scores:
        avg = sum(scores) / len(scores)
        logger.info(f"Average score: {avg:.4f} over {len(scores)} tasks")
    else:
        logger.warning("No valid scores computed.")

    with open(run_output_dir / "eval_results.json", "w") as f:
        json.dump({"results": results, "average_score": avg if scores else None}, f, indent=2)

    logger.info(f"Results saved to {run_output_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
