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


ERROR_HINTS = {
    "sqlite_error": (
        "SQL execution failed. First inspect the database schema with "
        "`inspect_sqlite_schema` or `summarize_sqlite`, then check table names, "
        "column names, SQL syntax, and try a simpler query."
    ),
    "empty_result": (
        "The query returned no rows. Check whether WHERE conditions are too strict, "
        "whether joins dropped rows, or whether the requested values use different spelling."
    ),
    "python_error": (
        "Python execution failed. Check data types, null handling, file paths, and column names. "
        "Print intermediate columns or shapes before the final calculation."
    ),
    "file_not_found": "The file path was not found. Call `list_context` to confirm available paths.",
    "timeout": "Execution timed out. Reduce data volume, add SQL LIMITs, or split the work into smaller steps.",
    "key_error": (
        "A referenced key or column does not exist. Inspect the available columns with "
        "`summarize_csv`, `read_csv`, or `inspect_sqlite_schema` before retrying."
    ),
    "schema_required": (
        "Before running SQL on a database, inspect its schema with `inspect_sqlite_schema` "
        "or `summarize_sqlite` for the same database path."
    ),
    "discover_required": (
        "Start by calling `list_context` so you know which files and directories are available."
    ),
    "understand_required": (
        "Inspect or summarize the relevant data source before analysis. Use `summarize_csv` "
        "for CSV, `inspect_sqlite_schema` or `summarize_sqlite` for SQLite, and "
        "`search_doc_keywords` or `read_doc_segment` for long documents."
    ),
    "verification_required": (
        "Before submitting, verify the result with an independent SQL/Python check or a concise "
        "cross-check of row counts, columns, filters, and calculations."
    ),
    "repeated_failure": (
        "This repeats a recently failed tool call with the same input. Change strategy: inspect "
        "the data, simplify the query/code, or use a different tool."
    ),
    "python_recovery_required": (
        "Python failed repeatedly. Inspect columns, file paths, shapes, and sample rows before "
        "running more Python."
    ),
    "document_workflow_required": (
        "For hard or extreme document tasks, locate relevant sections first with "
        "`search_doc_keywords` or read targeted segments with `read_doc_segment`."
    ),
    "tool_input_error": "The tool input is invalid. Review the tool schema and provide all required fields.",
}

SQL_SCHEMA_TOOLS = {"inspect_sqlite_schema", "summarize_sqlite"}
DISCOVER_TOOLS = {
    "get_doc_info",
    "inspect_sqlite_schema",
    "list_context",
    "read_csv",
    "read_doc",
    "read_doc_segment",
    "read_json",
    "search_doc_keywords",
    "summarize_csv",
    "summarize_sqlite",
}
UNDERSTAND_TOOLS = DISCOVER_TOOLS | {"execute_python"}
ANALYZE_TOOLS = {
    "execute_context_sql",
    "execute_python",
    "read_csv",
    "read_json",
    "read_doc_segment",
    "search_doc_keywords",
    "summarize_csv",
    "summarize_sqlite",
    "inspect_sqlite_schema",
}
VERIFY_TOOLS = {"execute_context_sql", "execute_python", "read_csv", "read_json", "summarize_csv", "summarize_sqlite"}
SUBMIT_TOOLS = {"answer"}
ANALYSIS_ACTIONS = {"execute_context_sql", "execute_python"}
VERIFY_ACTIONS = {"execute_context_sql", "execute_python"}
INTERNAL_ACTIONS = {"__error__", "__reflect__", "__system_hint__"}

OPTIONAL_TOOL_FIELDS = {
    "list_context": {"max_depth"},
    "read_csv": {"max_rows"},
    "read_doc": {"max_chars"},
    "read_doc_segment": {"segment_index"},
    "read_json": {"max_chars"},
    "execute_context_sql": {"limit"},
}


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
    workflow_mode: str = "soft"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str
    hint: str
    stage: str
    rewritten_action: str | None = None


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


def _schema_value_to_json_schema(value: Any, *, field_name: str = "") -> dict[str, Any]:
    if field_name == "keywords":
        return {"type": "array", "items": {"type": "string"}}
    if isinstance(value, bool):
        return {"type": "boolean", "default": value}
    if isinstance(value, int):
        return {"type": "integer", "default": value}
    if isinstance(value, float):
        return {"type": "number", "default": value}
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return {"type": "array", "items": {"type": "array"}}
        return {"type": "array"}
    if isinstance(value, dict):
        return {"type": "object"}
    if isinstance(value, str) and value.isdigit():
        return {"type": "integer", "description": str(value)}
    return {"type": "string", "description": str(value)}


def _tool_to_openai_spec(spec: ToolSpec) -> dict[str, Any]:
    properties = {}
    required = []
    if spec.input_schema:
        optional_fields = OPTIONAL_TOOL_FIELDS.get(spec.name, set())
        for key, value in spec.input_schema.items():
            properties[key] = _schema_value_to_json_schema(value, field_name=key)
            if key not in optional_fields:
                required.append(key)
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


def _parse_native_tool_calls(tool_calls: list[dict], valid_tool_names: set[str] | None = None) -> ModelStep:
    if not tool_calls:
        raise ValueError("No tool calls in native response")
    tc = tool_calls[0]
    func = tc.get("function", {})
    action = func.get("name", "")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("Native tool call missing function name.")
    action = action.strip()
    if valid_tool_names is not None and action not in valid_tool_names:
        raise ValueError(f"Native tool call referenced unknown tool: {action}")
    try:
        action_input = json.loads(func.get("arguments", "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Native tool call arguments are not valid JSON: {exc}") from exc
    if not isinstance(action_input, dict):
        raise ValueError("Native tool call arguments must decode to a JSON object.")
    raw_response = json.dumps(
        {"thought": "", "action": action, "action_input": action_input},
        ensure_ascii=False,
    )
    return ModelStep(thought="", action=action, action_input=action_input, raw_response=raw_response)


def _retry_prompt_for_invalid_model_step(error: Exception) -> ModelMessage:
    return ModelMessage(
        role="user",
        content=(
            "Your previous tool call/action was invalid and could not be executed. "
            f"Error: {error}. Return exactly one valid tool call or one JSON action. "
            "If unsure, call `list_context` first."
        ),
    )


def _stringify_error_payload(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False)
    except TypeError:
        return str(payload)


def _classify_error(action: str, payload: object) -> str:
    text = _stringify_error_payload(payload).lower()
    if "schema" in text and "before running sql" in text:
        return "schema_required"
    if "no such file" in text or "missing context asset" in text or "file not found" in text:
        return "file_not_found"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "keyerror" in text or "key error" in text:
        return "key_error"
    if action == "execute_python" or "traceback" in text or "python execution" in text:
        return "python_error"
    if action == "execute_context_sql" or "sqlite" in text or "sql" in text:
        return "sqlite_error"
    if "required" in text or "invalid" in text or "must be" in text:
        return "tool_input_error"
    return "tool_input_error"


def _error_hint(action: str, payload: object) -> tuple[str, str]:
    error_type = _classify_error(action, payload)
    return error_type, ERROR_HINTS.get(error_type, "The operation failed. Try a different tool or strategy.")


def _is_tool_step(step: StepRecord) -> bool:
    return step.action not in INTERNAL_ACTIONS


def _successful_steps(state: AgentRuntimeState, actions: set[str] | None = None) -> list[StepRecord]:
    return [
        step
        for step in state.steps
        if _is_tool_step(step) and step.ok and (actions is None or step.action in actions)
    ]


def _has_successful_action(state: AgentRuntimeState, action: str) -> bool:
    return bool(_successful_steps(state, {action}))


def _failed_signature(step: StepRecord) -> str:
    return json.dumps(
        {"action": step.action, "action_input": step.action_input},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _model_step_signature(model_step: ModelStep) -> str:
    return json.dumps(
        {"action": model_step.action, "action_input": model_step.action_input},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _step_has_empty_result(step: StepRecord) -> bool:
    content = step.observation.get("content")
    return isinstance(content, dict) and content.get("row_count") == 0


class WorkflowPolicy:
    def __init__(self, *, mode: str = "soft") -> None:
        normalized = mode.lower().strip()
        if normalized not in {"off", "soft", "strict"}:
            raise ValueError("workflow_mode must be one of: off, soft, strict.")
        self.mode = normalized

    def decide(self, task: PublicTask, state: AgentRuntimeState, model_step: ModelStep) -> PolicyDecision:
        stage = self.stage(state, model_step)
        if self.mode == "off" or model_step.action in INTERNAL_ACTIONS:
            return PolicyDecision(True, "allowed", "", stage)

        repeated = self._repeated_failed_call(state, model_step)
        if repeated is not None:
            return self._block("repeated_failure", ERROR_HINTS["repeated_failure"], stage, "list_context")

        if not _has_successful_action(state, "list_context") and model_step.action != "list_context":
            return self._block("discover_required", ERROR_HINTS["discover_required"], "DISCOVER", "list_context")

        if self._python_recovery_required(state, model_step):
            return self._block(
                "python_recovery_required",
                ERROR_HINTS["python_recovery_required"],
                stage,
                "summarize_csv",
            )

        if self._document_workflow_required(task, model_step):
            return self._block(
                "document_workflow_required",
                ERROR_HINTS["document_workflow_required"],
                stage,
                "search_doc_keywords",
            )

        if model_step.action == "execute_context_sql" and not self._has_seen_schema(state, model_step.action_input.get("path")):
            return self._block("schema_required", ERROR_HINTS["schema_required"], stage, "inspect_sqlite_schema")

        if model_step.action == "execute_python" and not self._has_data_understanding(state):
            return self._block("understand_required", ERROR_HINTS["understand_required"], "UNDERSTAND", "summarize_csv")

        if model_step.action == "answer" and not self._can_submit(state):
            return self._block("verification_required", ERROR_HINTS["verification_required"], "VERIFY", "execute_python")

        if not self._allowed_in_stage(stage, model_step.action):
            return self._block(
                "workflow_stage_violation",
                self._stage_hint(stage),
                stage,
                self._recommended_action(stage),
            )

        return PolicyDecision(True, "allowed", "", stage)

    def stage(self, state: AgentRuntimeState, model_step: ModelStep | None = None) -> str:
        if model_step is not None and model_step.action == "answer":
            return "SUBMIT"
        if not _has_successful_action(state, "list_context"):
            return "DISCOVER"
        if not self._has_data_understanding(state):
            return "UNDERSTAND"
        if not self._has_successful_analysis(state):
            return "ANALYZE"
        if not self._has_successful_verification(state):
            return "VERIFY"
        return "SUBMIT"

    def _block(self, reason: str, hint: str, stage: str, rewritten_action: str | None = None) -> PolicyDecision:
        strict_hint = hint
        if self.mode == "strict" and rewritten_action:
            strict_hint = f"{hint} Next action should be `{rewritten_action}`."
        return PolicyDecision(False, reason, strict_hint, stage, rewritten_action)

    def _allowed_in_stage(self, stage: str, action: str) -> bool:
        allowed_by_stage = {
            "DISCOVER": DISCOVER_TOOLS,
            "UNDERSTAND": UNDERSTAND_TOOLS,
            "ANALYZE": ANALYZE_TOOLS,
            "VERIFY": VERIFY_TOOLS | SUBMIT_TOOLS,
            "SUBMIT": SUBMIT_TOOLS | VERIFY_TOOLS,
        }
        return action in allowed_by_stage.get(stage, DISCOVER_TOOLS | ANALYZE_TOOLS | SUBMIT_TOOLS)

    def _stage_hint(self, stage: str) -> str:
        if stage == "DISCOVER":
            return ERROR_HINTS["discover_required"]
        if stage == "UNDERSTAND":
            return ERROR_HINTS["understand_required"]
        if stage == "VERIFY":
            return ERROR_HINTS["verification_required"]
        return "Follow the workflow stage before choosing this tool."

    def _recommended_action(self, stage: str) -> str | None:
        return {
            "DISCOVER": "list_context",
            "UNDERSTAND": "summarize_csv",
            "ANALYZE": "execute_python",
            "VERIFY": "execute_python",
            "SUBMIT": "answer",
        }.get(stage)

    def _has_seen_schema(self, state: AgentRuntimeState, db_path: object) -> bool:
        if db_path is None:
            return False
        normalized_path = str(db_path)
        for step in _successful_steps(state, SQL_SCHEMA_TOOLS):
            if str(step.action_input.get("path")) == normalized_path:
                return True
        return False

    def _has_data_understanding(self, state: AgentRuntimeState) -> bool:
        understanding_actions = {
            "get_doc_info",
            "inspect_sqlite_schema",
            "read_csv",
            "read_doc",
            "read_doc_segment",
            "read_json",
            "search_doc_keywords",
            "summarize_csv",
            "summarize_sqlite",
        }
        return bool(_successful_steps(state, understanding_actions))

    def _has_successful_analysis(self, state: AgentRuntimeState) -> bool:
        return any(
            step.action in ANALYSIS_ACTIONS and not _step_has_empty_result(step)
            for step in _successful_steps(state, ANALYSIS_ACTIONS)
        )

    def _has_successful_verification(self, state: AgentRuntimeState) -> bool:
        analyses = _successful_steps(state, ANALYSIS_ACTIONS)
        if len(analyses) >= 2:
            return True
        return any(
            step.action in VERIFY_ACTIONS and bool(step.observation.get("verified"))
            for step in _successful_steps(state, VERIFY_ACTIONS)
        )

    def _can_submit(self, state: AgentRuntimeState) -> bool:
        if not self._has_successful_analysis(state) or not self._has_successful_verification(state):
            return False
        last_tool_steps = [step for step in state.steps if _is_tool_step(step)]
        if not last_tool_steps:
            return False
        last_step = last_tool_steps[-1]
        if not last_step.ok:
            return False
        if last_step.observation.get("warning_type") == "empty_result":
            return False
        return True

    def _repeated_failed_call(self, state: AgentRuntimeState, model_step: ModelStep) -> StepRecord | None:
        signature = _model_step_signature(model_step)
        recent_tool_steps = [step for step in state.steps if _is_tool_step(step)][-3:]
        for step in recent_tool_steps:
            if not step.ok and _failed_signature(step) == signature:
                return step
        return None

    def _python_recovery_required(self, state: AgentRuntimeState, model_step: ModelStep) -> bool:
        if model_step.action != "execute_python":
            return False
        recent_python = [step for step in state.steps if step.action == "execute_python"][-2:]
        if len(recent_python) < 2 or any(step.ok for step in recent_python):
            return False
        return not self._has_data_understanding_since(state, recent_python[0].step_index)

    def _has_data_understanding_since(self, state: AgentRuntimeState, step_index: int) -> bool:
        understanding_actions = {
            "inspect_sqlite_schema",
            "read_csv",
            "read_json",
            "summarize_csv",
            "summarize_sqlite",
        }
        return any(
            step.step_index > step_index and step.ok and step.action in understanding_actions
            for step in state.steps
        )

    def _document_workflow_required(self, task: PublicTask, model_step: ModelStep) -> bool:
        if task.difficulty not in {"hard", "extreme"} or model_step.action != "read_doc":
            return False
        return int(model_step.action_input.get("max_chars", 8000)) > 2500


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
        self.workflow_policy = WorkflowPolicy(mode=self.config.workflow_mode)

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

    def _complete_text_step_with_retry(self, messages: list[ModelMessage]) -> tuple[str, ModelStep]:
        raw_response = self.model.complete(messages)
        try:
            return raw_response, parse_model_step(raw_response)
        except Exception as exc:
            retry_messages = [*messages, _retry_prompt_for_invalid_model_step(exc)]
            retry_raw_response = self.model.complete(retry_messages)
            return retry_raw_response, parse_model_step(retry_raw_response)

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        self.context_manager.clear()
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
                self.context_manager.add_user(build_observation_prompt(step.observation))
        return self.context_manager.get_messages()

    def _has_seen_schema(self, state: AgentRuntimeState, db_path: object) -> bool:
        if db_path is None:
            return False
        normalized_path = str(db_path)
        for step in state.steps:
            if not step.ok or step.action not in SQL_SCHEMA_TOOLS:
                continue
            if str(step.action_input.get("path")) == normalized_path:
                return True
        return False

    def _schema_guard_observation(self, model_step: ModelStep, state: AgentRuntimeState) -> dict[str, Any] | None:
        if self.config.workflow_mode == "off":
            return None
        if model_step.action != "execute_context_sql":
            return None
        db_path = model_step.action_input.get("path")
        if self._has_seen_schema(state, db_path):
            return None
        hint = ERROR_HINTS["schema_required"]
        return {
            "ok": False,
            "tool": model_step.action,
            "error_type": "schema_required",
            "hint": hint,
            "content": {
                "error": hint,
                "required_next_step": "Call inspect_sqlite_schema or summarize_sqlite first.",
                "path": db_path,
            },
        }

    def _augment_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not observation.get("ok"):
            if "policy_reason" in observation:
                observation["error_type"] = observation["policy_reason"]
                return observation
            error_type, hint = _error_hint(str(observation.get("tool", "")), observation.get("content", observation))
            observation["error_type"] = error_type
            observation["hint"] = hint
            return observation

        if observation.get("tool") == "execute_context_sql":
            content = observation.get("content")
            if isinstance(content, dict) and content.get("row_count") == 0:
                observation["warning_type"] = "empty_result"
                observation["hint"] = ERROR_HINTS["empty_result"]
        return observation

    def _policy_observation(self, model_step: ModelStep, decision: PolicyDecision) -> dict[str, Any]:
        return {
            "ok": False,
            "tool": model_step.action,
            "error_type": decision.reason,
            "policy_stage": decision.stage,
            "policy_reason": decision.reason,
            "hint": decision.hint,
            "rewritten_action": decision.rewritten_action,
            "content": {
                "error": decision.hint,
                "blocked_action": model_step.action,
                "blocked_action_input": model_step.action_input,
                "stage": decision.stage,
                "recommended_next_action": decision.rewritten_action,
            },
        }

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
                raw_response, model_step = self._complete_text_step_with_retry(messages)
                used_native = False
            elif self.config.use_native_tool_calling and native_available and tools_spec:
                try:
                    content, tool_calls, _ = self.model.complete_with_tools(messages, tools=tools_spec, tool_choice="auto")
                    if tool_calls:
                        model_step = _parse_native_tool_calls(tool_calls, set(self.tools.specs))
                        raw_response = model_step.raw_response
                        used_native = True
                    else:
                        raw_response, model_step = self._complete_text_step_with_retry(messages)
                        used_native = False
                except Exception as exc:
                    logger.warning(f"Native TC fallback to text: {exc}")
                    retry_messages = [*messages, _retry_prompt_for_invalid_model_step(exc)]
                    raw_response, model_step = self._complete_text_step_with_retry(retry_messages)
                    used_native = False
            else:
                raw_response, model_step = self._complete_text_step_with_retry(messages)
                used_native = False

            try:
                if not used_native:
                    model_step = parse_model_step(raw_response)

                force_timeout_answer = (
                    step_index == self.config.max_steps
                    and self.config.force_answer_on_timeout
                    and model_step.action == "answer"
                )
                policy_decision = (
                    PolicyDecision(True, "timeout_forced_answer", "", "SUBMIT")
                    if force_timeout_answer
                    else self.workflow_policy.decide(task, state, model_step)
                )

                if policy_decision.allowed:
                    observation = self._schema_guard_observation(model_step, state)
                else:
                    observation = self._policy_observation(model_step, policy_decision)

                if observation is None:
                    tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                    observation = {"ok": tool_result.ok, "tool": model_step.action, "content": tool_result.content}
                    is_terminal = tool_result.is_terminal
                    answer = tool_result.answer
                else:
                    is_terminal = False
                    answer = None

                observation = self._augment_observation(observation)
                step_ok = bool(observation["ok"])
                step_record = StepRecord(step_index=step_index, thought=model_step.thought, action=model_step.action, action_input=model_step.action_input, raw_response=raw_response, observation=observation, ok=step_ok)
                state.steps.append(step_record)

                if step_ok:
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1

                if is_terminal:
                    state.answer = answer
                    logger.info(f"Task {task.task_id}: done step {step_index}")
                    break

                if consecutive_errors >= self.config.max_consecutive_errors:
                    state.steps.append(StepRecord(step_index=step_index + 1, thought="System intervention", action="__system_hint__", action_input={}, raw_response="", observation={"ok": True, "hint": "Multiple errors in a row. Please reconsider your approach. Try a different tool or strategy. Review available tools and their descriptions."}, ok=True))
                    consecutive_errors = 0

                if self.config.enable_reflection and step_index % self.config.reflection_interval == 0:
                    state.steps.append(StepRecord(step_index=step_index + 1, thought="Reflection checkpoint", action="__reflect__", action_input={}, raw_response="", observation={"ok": True, "hint": "[REFLECTION CHECKPOINT] Review:\n1. Still on track?\n2. Results consistent?\n3. Strategy needs adjustment?\n4. Enough info for answer?"}, ok=True))

            except Exception as exc:
                consecutive_errors += 1
                error_type, hint = _error_hint("__error__", str(exc))
                state.steps.append(StepRecord(step_index=step_index, thought="", action="__error__", action_input={}, raw_response=raw_response, observation={"ok": False, "error": str(exc), "error_type": error_type, "hint": hint}, ok=False))

                if consecutive_errors >= self.config.max_consecutive_errors:
                    state.steps.append(StepRecord(step_index=step_index + 1, thought="System intervention", action="__system_hint__", action_input={}, raw_response="", observation={"ok": True, "hint": "Multiple errors in a row. Please reconsider your approach. Try a different tool or strategy. Review available tools and their descriptions."}, ok=True))
                    consecutive_errors = 0

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(task_id=task.task_id, answer=state.answer, steps=list(state.steps), failure_reason=state.failure_reason)
