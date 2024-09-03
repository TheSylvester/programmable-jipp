from collections.abc import Sequence
from typing import List, Optional, Union, Literal
from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: Optional[Union[str, List[Union[str, dict]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    function_call: Optional[dict] = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResponse(BaseModel):
    message: LLMMessage
    usage: CompletionUsage
    model: str
    finish_reason: str
    formatted_response: Optional[str] = None  # parsed goes here


class JippResponse(BaseModel, Sequence):
    messages: List[LLMMessage]

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
