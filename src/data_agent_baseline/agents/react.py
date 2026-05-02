from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
    PLANNING_SYSTEM_PROMPT,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.context.manager import ContextManager
from data_agent_baseline.tools.registry import ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    force_answer_on_timeout: bool = True
    max_consecutive_errors: int = 3
    enable_planning: bool = True
    enable_reflection: bool = True
    reflection_interval: int = 3
    context_max_tokens: int = 200000
    use_native_tool_calling: bool = True


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)
    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")
    return ModelStep(thought=thought, action=action, action_input=action_input, raw_response=raw_response)


def _tool_to_openai_spec(spec: ToolSpec) -> dict[str, Any]:
    properties = {}
    required = []
    if spec.input_schema:
        for k in spec.input_schema:
            properties[k] = {"type": "string", "description": f"Value for {k}"}
            required.append(k)
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _parse_native_tool_calls(tool_calls: list[dict]) -> ModelStep:
    if not tool_calls:
        raise ValueError("No tool calls in native response")
    tc = tool_calls[0]
    func = tc.get("function", {})
    action = func.get("name", "")
    try:
        action_input = json.loads(func.get("arguments", "{}"))
    except json.JSONDecodeError:
        action_input = {}
    return ModelStep(thought="", action=action, action_input=action_input, raw_response=json.dumps(tc))


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        context_manager: ContextManager | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.context_manager = context_manager or ContextManager(max_tokens=self.config.context_max_tokens)

    def _generate_plan(self, task: PublicTask) -> str:
        plan_messages = [
            ModelMessage(role="system", content=PLANNING_SYSTEM_PROMPT),
            ModelMessage(role="user", content=build_task_prompt(task)),
        ]
        plan = self.model.complete(plan_messages)
        logger.info(f"Plan generated for {task.task_id}: {plan[:200]}...")
        return plan[:2000]

    def _build_tools_spec(self) -> list[dict]:
        return [_tool_to_openai_spec(spec) for spec in self.tools.specs.values()]

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(self.tools.describe_for_prompt(), system_prompt=self.system_prompt)
        self.context_manager.set_system(system_content)
        task_content = build_task_prompt(task)
        if state.plan:
            task_content += f"\n\n## Analysis Plan\n{state.plan}"
        self.context_manager.add_user(task_content)
        for step in state.steps:
            if step.action in ("__reflect__", "__system_hint__"):
                self.context_manager.add_user(build_observation_prompt(step.observation))
            else:
                self.context_manager.add_assistant(step.raw_response)
                self.context_manager.add_tool_result(str(step.step_index), build_observation_prompt(step.observation))
        return self.context_manager.get_messages()

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        consecutive_errors = 0

        if self.config.enable_planning:
            try:
                state.plan = self._generate_plan(task)
            except Exception as exc:
                logger.warning(f"Task {task.task_id}: plan generation failed: {exc}")

        tools_spec = self._build_tools_spec() if self.config.use_native_tool_calling else None
        native_available = hasattr(self.model, 'complete_with_tools')

        for step_index in range(1, self.config.max_steps + 1):
            messages = self._build_messages(task, state)

            if step_index == self.config.max_steps and self.config.force_answer_on_timeout:
                messages.append(ModelMessage(role="user", content="WARNING: This is your last step. You MUST call the `answer` tool now with your best answer. Do NOT call any other tool."))
                raw_response = self.model.complete(messages)
                used_native = False
            elif self.config.use_native_tool_calling and native_available and tools_spec:
                try:
                    content, tool_calls, _ = self.model.complete_with_tools(messages, tools=tools_spec, tool_choice="auto")
                    if tool_calls:
                        model_step = _parse_native_tool_calls(tool_calls)
                        raw_response = model_step.raw_response
                        used_native = True
                    else:
                        raw_response = self.model.complete(messages)
                        model_step = parse_model_step(raw_response)
                        used_native = False
                except Exception as exc:
                    logger.warning(f"Native TC fallback to text: {exc}")
                    raw_response = self.model.complete(messages)
                    model_step = parse_model_step(raw_response)
                    used_native = False
            else:
                raw_response = self.model.complete(messages)
                used_native = False

            try:
                if not used_native:
                    model_step = parse_model_step(raw_response)

                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {"ok": tool_result.ok, "tool": model_step.action, "content": tool_result.content}
                step_record = StepRecord(step_index=step_index, thought=model_step.thought, action=model_step.action, action_input=model_step.action_input, raw_response=raw_response, observation=observation, ok=tool_result.ok)
                state.steps.append(step_record)

                if tool_result.ok:
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1

                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    logger.info(f"Task {task.task_id}: done step {step_index}")
                    break

                if consecutive_errors >= self.config.max_consecutive_errors:
                    state.steps.append(StepRecord(step_index=step_index + 1, thought="System intervention", action="__system_hint__", action_input={}, raw_response="", observation={"ok": True, "hint": "Multiple errors in a row. Please reconsider your approach. Try a different tool or strategy. Review available tools and their descriptions."}, ok=True))
                    consecutive_errors = 0

                if self.config.enable_reflection and step_index % self.config.reflection_interval == 0:
                    state.steps.append(StepRecord(step_index=step_index + 1, thought="Reflection checkpoint", action="__reflect__", action_input={}, raw_response="", observation={"ok": True, "hint": "[REFLECTION CHECKPOINT] Review:\n1. Still on track?\n2. Results consistent?\n3. Strategy needs adjustment?\n4. Enough info for answer?"}, ok=True))

            except Exception as exc:
                consecutive_errors += 1
                state.steps.append(StepRecord(step_index=step_index, thought="", action="__error__", action_input={}, raw_response=raw_response, observation={"ok": False, "error": str(exc)}, ok=False))

                if consecutive_errors >= self.config.max_consecutive_errors:
                    state.steps.append(StepRecord(step_index=step_index + 1, thought="System intervention", action="__system_hint__", action_input={}, raw_response="", observation={"ok": True, "hint": "Multiple errors in a row. Please reconsider your approach. Try a different tool or strategy. Review available tools and their descriptions."}, ok=True))
                    consecutive_errors = 0

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(task_id=task.task_id, answer=state.answer, steps=list(state.steps), failure_reason=state.failure_reason)
