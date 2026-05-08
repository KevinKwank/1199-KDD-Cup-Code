from __future__ import annotations

from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.model import ModelMessage
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig, parse_model_step
from data_agent_baseline.tools.registry import (
    ToolExecutionResult,
    ToolRegistry,
    ToolSpec,
    create_default_tool_registry,
)


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


def _make_policy_tools() -> ToolRegistry:
    specs = {
        "answer": ToolSpec("answer", "Submit final answer", {"columns": ["c"], "rows": [["v"]]}),
        "execute_context_sql": ToolSpec(
            "execute_context_sql",
            "Run SQL",
            {"path": "db/test.sqlite", "sql": "SELECT ...", "limit": 200},
        ),
        "execute_python": ToolSpec("execute_python", "Run Python", {"code": "print('ok')"}),
        "inspect_sqlite_schema": ToolSpec(
            "inspect_sqlite_schema",
            "Inspect schema",
            {"path": "db/test.sqlite"},
        ),
        "list_context": ToolSpec("list_context", "List files", {"max_depth": 1}),
        "summarize_sqlite": ToolSpec("summarize_sqlite", "Summarize DB", {"path": "db/test.sqlite"}),
    }

    def ok_content(content):
        return lambda task, action_input: ToolExecutionResult(ok=True, content=content)

    def answer_handler(task, action_input):
        del task
        from data_agent_baseline.benchmark.schema import AnswerTable

        answer = AnswerTable(columns=list(action_input["columns"]), rows=list(action_input["rows"]))
        return ToolExecutionResult(
            ok=True,
            content={"status": "submitted"},
            is_terminal=True,
            answer=answer,
        )

    handlers = {
        "answer": answer_handler,
        "execute_context_sql": ok_content({"columns": ["c"], "rows": [[1]], "row_count": 1}),
        "execute_python": ok_content({"success": True, "output": "verified\n"}),
        "inspect_sqlite_schema": ok_content({"tables": [{"name": "t"}]}),
        "list_context": ok_content({"entries": [{"path": "db/test.sqlite", "kind": "file"}]}),
        "summarize_sqlite": ok_content({"tables": [{"name": "t", "row_count": 1}]}),
    }
    return ToolRegistry(specs=specs, handlers=handlers)


def test_force_answer_on_timeout():
    tools = create_default_tool_registry()
    model = ScriptedModelAdapter([
        'plan: inspect files, summarize, answer',
        '{"thought":"done","action":"answer","action_input":{"columns":["c"],"rows":[["1"]]}}',
    ])
    config = ReActAgentConfig(max_steps=2, force_answer_on_timeout=True, workflow_mode="off")
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


def test_sql_requires_schema_before_execution():
    tools = _make_policy_tools()
    model = ScriptedModelAdapter([
        '{"thought":"inspect files","action":"list_context","action_input":{"max_depth":1}}',
        '{"thought":"query first","action":"execute_context_sql","action_input":{"path":"db/test.sqlite","sql":"SELECT 1"}}',
        '{"thought":"fallback","action":"answer","action_input":{"columns":["c"],"rows":[["1"]]}}',
    ])
    config = ReActAgentConfig(
        max_steps=3,
        enable_planning=False,
        enable_reflection=False,
        force_answer_on_timeout=False,
    )

    agent = ReActAgent(model=model, tools=tools, config=config)
    result = agent.run(_make_task())

    blocked_step = result.steps[1]
    assert blocked_step.action == "execute_context_sql"
    assert blocked_step.ok is False
    assert blocked_step.observation["error_type"] == "schema_required"


def test_workflow_requires_list_context_before_analysis():
    tools = _make_policy_tools()
    model = ScriptedModelAdapter([
        '{"thought":"jump","action":"execute_python","action_input":{"code":"print(1)"}}',
        '{"thought":"recover","action":"list_context","action_input":{"max_depth":1}}',
    ])
    config = ReActAgentConfig(
        max_steps=2,
        enable_planning=False,
        enable_reflection=False,
        force_answer_on_timeout=False,
    )

    agent = ReActAgent(model=model, tools=tools, config=config)
    result = agent.run(_make_task())

    first_step = result.steps[0]
    assert first_step.ok is False
    assert first_step.observation["error_type"] == "discover_required"
    assert first_step.observation["rewritten_action"] == "list_context"


def test_workflow_blocks_premature_answer_without_verification():
    tools = _make_policy_tools()
    model = ScriptedModelAdapter([
        '{"thought":"inspect","action":"list_context","action_input":{"max_depth":1}}',
        '{"thought":"schema","action":"summarize_sqlite","action_input":{"path":"db/test.sqlite"}}',
        '{"thought":"query","action":"execute_context_sql","action_input":{"path":"db/test.sqlite","sql":"SELECT 1"}}',
        '{"thought":"done","action":"answer","action_input":{"columns":["c"],"rows":[["1"]]}}',
    ])
    config = ReActAgentConfig(
        max_steps=4,
        enable_planning=False,
        enable_reflection=False,
        force_answer_on_timeout=False,
    )

    agent = ReActAgent(model=model, tools=tools, config=config)
    result = agent.run(_make_task())

    blocked_step = result.steps[3]
    assert blocked_step.action == "answer"
    assert blocked_step.ok is False
    assert blocked_step.observation["error_type"] == "verification_required"


def test_workflow_allows_sequential_success_path():
    tools = _make_policy_tools()
    model = ScriptedModelAdapter([
        '{"thought":"inspect","action":"list_context","action_input":{"max_depth":1}}',
        '{"thought":"schema","action":"summarize_sqlite","action_input":{"path":"db/test.sqlite"}}',
        '{"thought":"query","action":"execute_context_sql","action_input":{"path":"db/test.sqlite","sql":"SELECT 1"}}',
        '{"thought":"verify","action":"execute_python","action_input":{"code":"print(\\"verified\\")"}}',
        '{"thought":"submit","action":"answer","action_input":{"columns":["c"],"rows":[["1"]]}}',
    ])
    config = ReActAgentConfig(
        max_steps=5,
        enable_planning=False,
        enable_reflection=False,
        force_answer_on_timeout=False,
    )

    agent = ReActAgent(model=model, tools=tools, config=config)
    result = agent.run(_make_task())

    assert result.answer is not None
    assert result.succeeded


def test_workflow_blocks_repeated_failed_call():
    tools = _make_policy_tools()
    model = ScriptedModelAdapter([
        '{"thought":"jump","action":"execute_python","action_input":{"code":"print(1)"}}',
        '{"thought":"repeat","action":"execute_python","action_input":{"code":"print(1)"}}',
    ])
    config = ReActAgentConfig(
        max_steps=2,
        enable_planning=False,
        enable_reflection=False,
        force_answer_on_timeout=False,
    )

    agent = ReActAgent(model=model, tools=tools, config=config)
    result = agent.run(_make_task())

    repeated_step = result.steps[1]
    assert repeated_step.ok is False
    assert repeated_step.observation["error_type"] == "repeated_failure"


class BadNativeToolCallAdapter:
    def __init__(self) -> None:
        self.complete_calls = 0

    def complete(self, messages: list[ModelMessage]) -> str:
        self.complete_calls += 1
        assert "previous tool call/action was invalid" in messages[-1].content
        return '{"thought":"recover","action":"list_context","action_input":{"max_depth":1}}'

    def complete_with_tools(self, messages, tools=None, tool_choice="auto"):
        del messages, tools, tool_choice
        return "", [{"type": "function", "function": {"name": "", "arguments": "{}"}}], 0


def test_bad_native_tool_call_falls_back_to_text_retry():
    tools = _make_policy_tools()
    model = BadNativeToolCallAdapter()
    config = ReActAgentConfig(
        max_steps=1,
        enable_planning=False,
        enable_reflection=False,
        force_answer_on_timeout=False,
        use_native_tool_calling=True,
    )

    agent = ReActAgent(model=model, tools=tools, config=config)
    result = agent.run(_make_task())

    assert model.complete_calls == 1
    assert result.steps[0].action == "list_context"
    assert result.steps[0].ok is True


def test_tool_registry_caches_read_only_tools():
    calls = {"count": 0}

    def handler(task, action_input):
        del task, action_input
        calls["count"] += 1
        return ToolExecutionResult(ok=True, content={"value": calls["count"]})

    registry = ToolRegistry(
        specs={"list_context": ToolSpec("list_context", "list", {"max_depth": 1})},
        handlers={"list_context": handler},
    )
    task = _make_task()

    first = registry.execute(task, "list_context", {"max_depth": 1})
    second = registry.execute(task, "list_context", {"max_depth": 1})

    assert calls["count"] == 1
    assert first.content == second.content
