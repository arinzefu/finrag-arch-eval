from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        raise NotImplementedError


class CallableLLM(BaseLLM):
    """Adapter for tests or a user-supplied local callable."""

    def __init__(
        self,
        fn: Callable[[str, str], str | LLMResponse],
        *,
        provider: str = "callable",
        model: str = "callable-model",
    ) -> None:
        self.fn = fn
        self.provider = provider
        self.model = model

    def generate(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        result = self.fn(system_prompt, user_prompt)
        if isinstance(result, LLMResponse):
            return result

        text = str(result).strip()
        if not text:
            raise RuntimeError("CallableLLM returned an empty answer.")

        return LLMResponse(text=text, provider=self.provider, model=self.model)


class OpenAIResponsesLLM(BaseLLM):
    """
    OpenAI Responses API adapter.

    temperature is optional because model families differ in supported sampling
    parameters. Whatever policy you choose must remain identical across P0-P3.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        max_output_tokens: int = 256,
        temperature: float | None = None,
        client: Any | None = None,
    ) -> None:
        model = str(model).strip()
        if not model:
            raise ValueError("model must not be empty.")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be > 0.")

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Install the `openai` package first.") from exc
            client = OpenAI(api_key=api_key)

        self.client = client
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = temperature

    def generate(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": str(system_prompt),
            "input": str(user_prompt),
            "max_output_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = float(self.temperature)

        response = self.client.responses.create(**kwargs)
        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            raise RuntimeError("The Responses API returned no output_text.")

        usage = getattr(response, "usage", None)

        def get_usage(name: str) -> int | None:
            if usage is None:
                return None
            value = getattr(usage, name, None)
            return None if value is None else int(value)

        return LLMResponse(
            text=text,
            provider="openai",
            model=self.model,
            input_tokens=get_usage("input_tokens"),
            output_tokens=get_usage("output_tokens"),
            total_tokens=get_usage("total_tokens"),
        )
