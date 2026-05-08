from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig, load_app_config_from_env
from data_agent_baseline.eval.run_eval import _find_ground_truth
from data_agent_baseline.run.runner import run_benchmark
from data_agent_baseline.tools.filesystem import read_csv_preview
from data_agent_baseline.tools.registry import create_default_tool_registry


def _write_task(root: Path, task_id: str = "task_1") -> None:
    task_dir = root / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": task_id, "difficulty": "easy", "question": "answer"}),
        encoding="utf-8",
    )


def test_env_config_defaults_to_official_io(monkeypatch):
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    monkeypatch.setenv("MODEL_API_URL", "http://model")
    monkeypatch.setenv("MODEL_API_KEY", "key")

    config = load_app_config_from_env()

    assert config.dataset.root_path == Path("/input")
    assert config.run.output_dir == Path("/output")
    assert config.run.nest_run_id is False
    assert config.run.write_summary is False


def test_official_output_layout_has_no_nested_run_id(tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_task(input_root)
    config = AppConfig(
        dataset=DatasetConfig(root_path=input_root),
        agent=AgentConfig(max_steps=2, workflow_mode="off"),
        run=RunConfig(output_dir=output_root, nest_run_id=False, write_summary=False, max_workers=1),
    )
    model = ScriptedModelAdapter(
        [
            "plan",
            '{"thought":"done","action":"answer","action_input":{"columns":["c"],"rows":[["1"]]}}',
        ]
    )

    run_output_dir, artifacts = run_benchmark(
        config=config,
        model=model,
        tools=create_default_tool_registry(),
    )

    assert run_output_dir == output_root
    assert artifacts[0].prediction_csv_path == output_root / "task_1" / "prediction.csv"
    assert artifacts[0].prediction_csv_path.exists()
    assert not (output_root / "summary.json").exists()


def test_read_csv_preview_streams_preview_rows_but_counts_all_rows(tmp_path):
    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    context_dir = tmp_path / "context"
    csv_dir = context_dir / "csv"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "large.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    task = PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="q"),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )

    preview = read_csv_preview(task, "csv/large.csv", max_rows=2)

    assert preview["columns"] == ["a", "b"]
    assert preview["rows"] == [["1", "2"], ["3", "4"]]
    assert preview["row_count"] == 3
    assert preview["truncated"] is True


def test_find_ground_truth_supports_official_gold_layout(tmp_path):
    dataset_root = tmp_path / "data" / "public" / "input"
    official_gt = tmp_path / "data" / "public" / "output" / "task_1" / "gold.csv"
    official_gt.parent.mkdir(parents=True)
    official_gt.write_text("c\n1\n", encoding="utf-8")

    assert _find_ground_truth(dataset_root, "task_1") == official_gt
