from __future__ import annotations

from data_agent_baseline.agents.model import ModelMessage


class ContextManager:
    def __init__(self, max_tokens: int = 200000):
        self.max_tokens = max_tokens
        self._system_message: ModelMessage | None = None
        self._messages: list[ModelMessage] = []

    def set_system(self, content: str) -> None:
        self._system_message = ModelMessage(role="system", content=content)

    def add_user(self, content: str) -> None:
        self._messages.append(ModelMessage(role="user", content=content))

    def add_assistant(self, content: str, tool_calls: list[dict] | None = None) -> None:
        self._messages.append(ModelMessage(role="assistant", content=content, tool_calls=tool_calls))

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append(ModelMessage(role="tool", content=content, tool_call_id=tool_call_id))

    def get_messages(self) -> list[ModelMessage]:
        all_msgs: list[ModelMessage] = []
        if self._system_message:
            all_msgs.append(self._system_message)
        all_msgs.extend(self._messages)

        total_chars = sum(len(m.content) for m in all_msgs)
        estimated_tokens = total_chars // 2

        if estimated_tokens <= self.max_tokens:
            return all_msgs

        return self._compress(all_msgs)

    def _compress(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        if not self._system_message or len(messages) <= 3:
            return messages

        system = messages[0]
        rest = messages[1:]

        result = [system]
        budget_chars = (self.max_tokens - len(system.content)) * 2

        for msg in reversed(rest):
            msg_chars = len(msg.content)
            if budget_chars - msg_chars < 4000:
                break
            result.insert(1, msg)
            budget_chars -= msg_chars

        if len(result) < len(messages):
            result.insert(1, ModelMessage(
                role="user",
                content="[系统提示：由于上下文长度限制，早期的对话历史已被截断。请基于当前可见的信息继续推理。]"
            ))

        return result

    def clear(self) -> None:
        self._messages = []

    @property
    def message_count(self) -> int:
        return len(self._messages)
