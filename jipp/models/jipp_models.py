import asyncio
from collections.abc import Sequence
from typing_extensions import (
    Callable,
    List,
    Optional,
    Required,
    TypedDict,
    Union,
    Literal,
)
from pydantic import BaseModel, Field


class Function(BaseModel):
    name: str
    arguments: str


class Tool:
    definition: BaseModel
    function: Callable

    def __call__(self, *args, **kwargs):
        """Handle both sync and async calls and cast result to string."""
        if asyncio.iscoroutinefunction(self.function):
            # If in an async context, await the function
            if self._is_in_async_context():
                return self._call_async(*args, **kwargs)
            else:
                # Run async function in sync context using asyncio.run()
                return str(asyncio.run(self.function(*args, **kwargs)))
        else:
            # If the function is synchronous, call it and cast to string
            return self.function(*args, **kwargs)

    async def _call_async(self, *args, **kwargs):
        """Await the async function if called in an async context and cast to string."""
        result = await self.function(*args, **kwargs)
        return result

    def _is_in_async_context(self) -> bool:
        """Helper to check if we're in an async context."""
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: Function


class MessageContentText(TypedDict, total=False):
    text: Required[str]
    """The text content."""

    type: Required[Literal["text"]]
    """The type of the content part."""


class ImageURL(TypedDict, total=False):
    url: Required[str]
    """Either a URL of the image or the base64 encoded image data."""

    detail: Literal["auto", "low", "high"]
    """Specifies the detail level of the image."""


class MessageContentImage(TypedDict, total=False):
    image_url: Required[ImageURL]
    type: Required[Literal["image_url"]]


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: Optional[
        Union[str, List[Union[MessageContentText, MessageContentImage]]]
    ] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[Function] = None
    tool_call_id: Optional[str] = Field(
        None, description="For Tool Call response messages"
    )

    class Config:
        extra = "forbid"  # This will raise an error if extra fields are provided


class LLMError(Exception):
    pass


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResponse(BaseModel):
    message: LLMMessage
    usage: CompletionUsage
    model: str
    finish_reason: str


class Conversation(BaseModel, Sequence):
    messages: List[LLMMessage] = Field(default_factory=list)
    usage: CompletionUsage
    model: str
    finish_reason: str
    parsed: Optional[BaseModel] = None  # parsed goes here

    def __getitem__(
        self, index: Union[int, slice, List[int]]
    ) -> Union[LLMMessage, List[LLMMessage]]:
        if isinstance(index, list):
            return [self.messages[i] for i in index]
        return self.messages[index]

    def __len__(self) -> int:
        return len(self.messages)

    def __str__(self) -> str:
        for message in reversed(self.messages):
            if message.role == "assistant":
                if isinstance(message.content, str):
                    return message.content
                elif isinstance(message.content, list):
                    return " ".join(item.text for item in message.content if item.text)
        return ""
