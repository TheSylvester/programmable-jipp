import asyncio
from collections.abc import Sequence
from typing import Type
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


class NotGiven:
    """Sentinel value for distinguishing omitted arguments from None."""

    def __bool__(self) -> Literal[False]:
        """
        Always returns False to indicate the absence of a value.

        Returns:
            Literal[False]: Always False.
        """
        return False

    def __repr__(self) -> str:
        """
        Returns the string representation of the NotGiven instance.

        Returns:
            str: The string "NOT_GIVEN".
        """
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()


class Function(BaseModel):
    """Represents a function with a name and arguments."""

    name: str
    arguments: str


class Tool:
    """
    A class representing a tool with a schema and an associated function.

    This class allows for the creation of tools that can be used in various contexts,
    supporting both synchronous and asynchronous function calls.

    Attributes:
        schema (Type[BaseModel]): The Pydantic model schema for the tool's input.
        function (Callable): The function to be executed when the tool is called.
    """

    schema: Type[BaseModel]
    function: Callable

    def __init__(self, schema: Type[BaseModel], function: Callable):
        """
        Initialize a Tool instance.

        Args:
            schema (Type[BaseModel]): The Pydantic model schema for the tool's input.
            function (Callable): The function to be executed when the tool is called.
        """
        self.schema: Type[BaseModel] = schema
        self.function: Callable = function

    def __call__(self, *args, **kwargs):
        """
        Handle both sync and async calls.

        This method allows the Tool to be called like a function, automatically
        handling both synchronous and asynchronous execution contexts.

        Returns:
            Any: The result of the function call.
        """
        if asyncio.iscoroutinefunction(self.function):
            if self._is_in_async_context():
                return self._call_async(*args, **kwargs)
            else:
                return asyncio.run(self.function(*args, **kwargs))
        else:
            return self.function(*args, **kwargs)

    async def _call_async(self, *args, **kwargs):
        """
        Await the async function if called in an async context.

        This method is used internally to handle asynchronous function calls.

        Returns:
            Any: The result of the asynchronous function call.
        """
        return await self.function(*args, **kwargs)

    def _is_in_async_context(self) -> bool:
        """
        Helper to check if we're in an async context.

        Returns:
            bool: True if in an async context, False otherwise.
        """
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False


class ToolCall(BaseModel):
    """Represents a tool call with an ID, type, and associated function."""

    id: str
    type: Literal["function"]
    function: Function


class MessageContentText(TypedDict, total=False):
    """Represents the text content of a message."""

    text: Required[str]
    """The text content."""

    type: Required[Literal["text"]]
    """The type of the content part."""


class ImageURL(TypedDict, total=False):
    """Represents an image URL with optional detail level."""

    url: Required[str]
    """Either a URL of the image or the base64 encoded image data."""

    detail: Literal["auto", "low", "high"]
    """Specifies the detail level of the image."""


class MessageContentImage(TypedDict, total=False):
    """Represents the image content of a message."""

    image_url: Required[ImageURL]
    type: Required[Literal["image_url"]]


class LLMMessage(BaseModel):
    """
    Represents a message in the context of a language model conversation.

    Attributes:
        role (Literal): The role of the message sender.
        content (Optional[Union[str, List[Union[MessageContentText, MessageContentImage]]]]): The content of the message.
        name (Optional[str]): The name associated with the message.
        tool_calls (Optional[List[ToolCall]]): Any tool calls made in the message.
        function_call (Optional[Function]): Any function call made in the message.
        tool_call_id (Optional[str]): The ID of a tool call response.
    """

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
    """Custom exception class for Language Model related errors."""

    pass


class CompletionUsage(BaseModel):
    """
    Represents the token usage statistics for a completion.

    Attributes:
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the completion.
        total_tokens (int): The total number of tokens used.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResponse(BaseModel):
    """
    Represents a response from a Language Model.

    Attributes:
        message (LLMMessage): The message content of the response.
        usage (CompletionUsage): The token usage statistics.
        model (str): The name or identifier of the model used.
        finish_reason (str): The reason why the model finished generating the response.
    """

    message: LLMMessage
    usage: CompletionUsage
    model: str
    finish_reason: str


class Conversation(BaseModel, Sequence):
    """
    Represents a conversation with a Language Model, including all messages and metadata.

    Attributes:
        messages (List[LLMMessage]): The list of messages in the conversation.
        usage (CompletionUsage): The token usage statistics for the conversation.
        model (str): The name or identifier of the model used.
        finish_reason (str): The reason why the model finished generating the last response.
        parsed (Optional[BaseModel]): Any parsed structured data from the conversation.
    """

    messages: List[LLMMessage] = Field(default_factory=list)
    usage: CompletionUsage
    model: str
    finish_reason: str
    parsed: Optional[BaseModel] = None  # parsed goes here

    def __getitem__(
        self, index: Union[int, slice, List[int]]
    ) -> Union[LLMMessage, List[LLMMessage]]:
        """
        Allows indexing and slicing of the conversation messages.

        Args:
            index (Union[int, slice, List[int]]): The index, slice, or list of indices to retrieve.

        Returns:
            Union[LLMMessage, List[LLMMessage]]: The requested message(s).
        """
        if isinstance(index, list):
            return [self.messages[i] for i in index]
        return self.messages[index]

    def __len__(self) -> int:
        """
        Returns the number of messages in the conversation.

        Returns:
            int: The number of messages.
        """
        return len(self.messages)

    def __str__(self) -> str:
        """
        Returns the content of the last assistant message in the conversation.

        Returns:
            str: The content of the last assistant message, or an empty string if not found.
        """
        for message in reversed(self.messages):
            if message.role == "assistant":
                if isinstance(message.content, str):
                    return message.content
                elif isinstance(message.content, list):
                    # return " ".join(item.text for item in message.content if item.text)
                    return " ".join(
                        item.get("text") for item in message.content if item.get("text")
                    )
        return ""
