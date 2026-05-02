from __future__ import annotations

from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent_baseline.agents.model import ModelMessage, ScriptedModelAdapter


def test_model_message_defaults():
    msg = ModelMessage(role="user", content="hello")
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


def test_model_message_with_tool_calls():
    msg = ModelMessage(role="assistant", content="", tool_calls=[{"id": "1"}])
    assert len(msg.tool_calls) == 1


def test_scripted_adapter_consumes_responses():
    adapter = ScriptedModelAdapter(["resp1", "resp2"])
    assert adapter.complete([]) == "resp1"
    assert adapter.complete([]) == "resp2"


def test_scripted_adapter_exhausted_raises():
    adapter = ScriptedModelAdapter([])
    with pytest.raises(RuntimeError):
        adapter.complete([])
