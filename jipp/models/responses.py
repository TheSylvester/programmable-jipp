# from typing import Optional, Any, Union, List
# from pydantic import BaseModel
# from collections.abc import Sequence
# from .llm_models import Message


# class LLMResponse(BaseModel, Sequence):
#     messages: List[Message]
#     raw_response: Any
#     parsed_content: Optional[Any] = None

#     def __getitem__(
#         self, index: Union[int, slice, List[int]]
#     ) -> Union[Message, List[Message]]:
#         if isinstance(index, list):
#             return [self.messages[i] for i in index]
#         return self.messages[index]

#     def __len__(self) -> int:
#         return len(self.messages)

#     def __str__(self) -> str:
#         for message in reversed(self.messages):
#             if message.role == "assistant":
#                 if isinstance(message.content, str):
#                     return message.content
#                 elif isinstance(message.content, list):
#                     return " ".join(item.text for item in message.content if item.text)
#         return ""
