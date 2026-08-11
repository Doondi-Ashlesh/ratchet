"""A scripted chat model, for driving the agent offline.

The agent runs on LangGraph's prebuilt loop, so the test double has to be a real
`BaseChatModel` rather than a callable of our own shape. That is the point of
using the framework's seam: the fake and the live model satisfy the same
interface, so a test cannot pass against a double that the provider would reject.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def calls(*specs: tuple[str, dict[str, Any]]) -> AIMessage:
    """An assistant turn that requests tools."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{i}"}
            for i, (name, args) in enumerate(specs)
        ],
    )


def says(text: str = "done") -> AIMessage:
    """An assistant turn with no tool call, which ends the loop."""
    return AIMessage(content=text)


class ScriptedModel(BaseChatModel):
    """Replies with a fixed script, then stops. Raises `explode` if set."""

    replies: list[BaseMessage] = Field(default_factory=list)
    explode_after: int = -1
    seen: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        # The script already names the tools it wants; binding is a no-op here.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.explode_after >= 0 and self.seen >= self.explode_after:
            raise RuntimeError("boom")
        reply = self.replies[self.seen] if self.seen < len(self.replies) else says()
        self.seen += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


class FunctionModel(BaseChatModel):
    """Answers with `fn(prompt_text)`. For tests that assert on what was asked.

    The prompt is flattened to one string because that is what these tests care
    about: whether a rule, a piece of feedback or the file body reached the model.
    """

    fn: Any = None
    prompts: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "function"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = "\n".join(str(m.content) for m in messages)
        self.prompts.append(text)
        reply = self.fn(text) if callable(self.fn) else str(self.fn)
        if isinstance(reply, BaseMessage):
            return ChatResult(generations=[ChatGeneration(message=reply)])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(reply)))])
