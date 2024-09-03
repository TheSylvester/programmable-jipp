# from typing import List, Union, Optional, Dict, Any, Callable
# from pydantic import BaseModel, Field
# import asyncio


# class MessageContent(BaseModel):
#     type: str
#     text: Optional[str] = None
#     image_url: Optional[Dict[str, str]] = None


# class Function(BaseModel):
#     name: str
#     arguments: Dict[str, Union[str, int, float, bool, List, Dict]]


# class ToolCall(BaseModel):
#     id: str
#     type: str = "function"
#     function: Function


# class Message(BaseModel):
#     role: str
#     content: Union[str, List[MessageContent], None] = None
#     refusal: Optional[str] = None
#     tool_calls: Optional[List[ToolCall]] = None

#     class Config:
#         extra = "allow"


# class Tool(BaseModel):
#     name: str
#     description: str
#     parameters_model: Optional[BaseModel] = None
#     parameters: Optional[Dict] = None
#     function: Callable[..., Any]

#     model_config = {"arbitrary_types_allowed": True}

#     def __call__(self, *args, **kwargs) -> Any:
#         if self.parameters_model:
#             try:
#                 validated_args = self.parameters_model(**kwargs)
#                 kwargs = validated_args.model_dump()
#             except ValueError as e:
#                 return f"Error: Invalid arguments for {self.name}: {str(e)}"

#         if asyncio.iscoroutinefunction(self.function):
#             return self._async_execute(*args, **kwargs)
#         else:
#             return self._sync_execute(*args, **kwargs)

#     def _sync_execute(self, *args, **kwargs) -> str:
#         try:
#             result = self.function(*args, **kwargs)
#             return str(result)
#         except Exception as e:
#             return f"Error executing {self.name}: {str(e)}"

#     async def _async_execute(self, *args, **kwargs) -> str:
#         try:
#             result = await self.function(*args, **kwargs)
#             return str(result)
#         except Exception as e:
#             return f"Error executing {self.name}: {str(e)}"


# class BaseLLMInput(BaseModel):
#     messages: List[Message]
#     model: str
#     max_tokens: Optional[int] = None
#     temperature: Optional[float] = None
#     top_p: Optional[float] = None
#     n: Optional[int] = None
#     stream: Optional[bool] = None
#     stop: Optional[Union[str, List[str]]] = None
#     presence_penalty: Optional[float] = None
#     frequency_penalty: Optional[float] = None
#     logit_bias: Optional[Dict[str, float]] = None
#     user: Optional[str] = None


# class OpenAIInput(BaseLLMInput):
#     tools: Optional[List[Dict[str, Any]]] = None
#     tool_choice: Optional[Union[str, Dict[str, Any]]] = None
#     response_format: Optional[Union[Dict[str, str], BaseModel]] = None


# class AnthropicInput(BaseModel):
#     model: str
#     messages: List[Message]
#     max_tokens: Optional[int] = None
#     system: Optional[str] = None
#     temperature: Optional[float] = None
#     top_k: Optional[int] = None
#     top_p: Optional[float] = None
#     metadata: Optional[Dict[str, Any]] = None
#     stop_sequences: Optional[List[str]] = None
#     stream: Optional[bool] = None
#     tools: Optional[List[Dict[str, Any]]] = None
#     tool_choice: Optional[Union[str, Dict[str, Any]]] = None

#     model_config = {"populate_by_name": True}


# # Alias for consistency
# OpenAIClientInput = OpenAIInput
