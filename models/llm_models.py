from collections.abc import Sequence
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union


class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]


class BaseLLMInput(BaseModel):
    messages: List[Message]
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class OpenAIInput(BaseLLMInput):
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, str]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class AnthropicInput(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    system: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    class Config:
        populate_by_name = True


class LLMResponse(BaseModel, Sequence):
    messages: List[Message]

    def __getitem__(
        self, index: Union[int, slice, List[int]]
    ) -> Union[Message, List[Message]]:
        if isinstance(index, list):
            return [self.messages[i] for i in index]
        return self.messages[index]

    def __len__(self) -> int:
        return len(self.messages)

    def __str__(self) -> str:
        for message in reversed(self.messages):
            if message.role == "assistant":
                return str(message.content)
        return ""
