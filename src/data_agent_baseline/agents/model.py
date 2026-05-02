from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, APITimeoutError, RateLimitError, APIConnectionError, OpenAI

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError

    def complete_with_tools(
        self,
        messages: list[ModelMessage],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[str, list[dict] | None, int]:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        max_tokens: int = 16384,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
        )
        self._total_tokens: int = 0

    def complete(self, messages: list[ModelMessage]) -> str:
        content, _, _ = self.complete_with_tools(messages, tools=None, tool_choice="none")
        return content

    def complete_with_tools(
        self,
        messages: list[ModelMessage],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[str, list[dict] | None, int]:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        oa_messages = []
        for msg in messages:
            m: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            oa_messages.append(m)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": oa_messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice

                response = self._client.chat.completions.create(**kwargs)
                choices = response.choices or []
                if not choices:
                    raise RuntimeError("Model response missing choices.")

                choice = choices[0]
                tokens = response.usage.total_tokens if response.usage else 0
                self._total_tokens += tokens

                content = choice.message.content or ""
                tool_calls = None

                if choice.message.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ]

                return content, tool_calls, tokens

            except (APITimeoutError, RateLimitError, APIConnectionError) as exc:
                last_error = exc
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(f"API调用失败(第{attempt+1}次): {exc}, 等待{wait}s重试...")
                time.sleep(wait)
            except APIError as exc:
                raise RuntimeError(f"Model request failed: {exc}") from exc

        raise RuntimeError(
            f"模型调用失败，已重试{self.max_retries}次: {last_error}"
        )

    @property
    def total_tokens(self) -> int:
        return self._total_tokens


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
