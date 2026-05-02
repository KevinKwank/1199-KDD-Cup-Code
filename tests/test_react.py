from __future__ import annotations

from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig, parse_model_step
from data_agent_baseline.tools.registry import create_default_tool_registry


def test_parse_valid_json():
    raw = '```json\n{"thought":"test","action":"answer","action_input":{"columns":["c"],"rows":[["1"]]}}\n```'
    step = parse_model_step(raw)
    assert step.thought == "test"
    assert step.action == "answer"


def test_parse_json_without_fence():
    raw = '{"thought":"test","action":"answer","action_input":{"columns":["c"],"rows":[["1"]]}}'
    step = parse_model_step(raw)
    assert step.thought == "test"


def test_parse_json_invalid_raises():
    with pytest.raises(ValueError):
        parse_model_step("not json at all")


def _make_task(task_id: str = "test", difficulty: str = "easy", question: str = "q") -> "PublicTask":
    from data_agent_baseline.benchmark.schema import PublicTask, TaskRecord, TaskAssets
    return PublicTask(
        record=TaskRecord(task_id=task_id, difficulty=difficulty, question=question),
        assets=TaskAssets(task_dir=Path("."), context_dir=Path(".")),
    )


def test_force_answer_on_timeout():
    tools = create_default_tool_registry()
    model = ScriptedModelAdapter([
        'plan: inspect files, summarize, answer',
        '{"thought":"done","action":"answer","action_input":{"columns":["c"],"rows":[["1"]]}}',
    ])
    config = ReActAgentConfig(max_steps=2, force_answer_on_timeout=True)
    agent = ReActAgent(model=model, tools=tools, config=config)

    task = _make_task()
    result = agent.run(task)
    assert result.answer is not None


def test_max_consecutive_errors_intervention():
    tools = create_default_tool_registry()
    config = ReActAgentConfig(
        max_steps=8, max_consecutive_errors=2,
        enable_planning=False, enable_reflection=False,
    )

    responses = []
    for i in range(8):
        responses.append('{"thought":"x","action":"list_context","action_input":{"max_depth":"invalid"}}')
    model = ScriptedModelAdapter(responses)

    agent = ReActAgent(model=model, tools=tools, config=config)
    task = _make_task()
    result = agent.run(task)

    system_hints = [s for s in result.steps if s.action == "__system_hint__"]
    assert len(system_hints) > 0
