# from typing import List, Optional, Union, Type
# from pydantic import BaseModel, Field, field_validator
# from .common import Tool


# class AskLLMInput(BaseModel):
#     """
#     Input model for LLM requests.

#     Attributes:
#         model (str): The name of the LLM model to use.
#         messages (List[dict]): The conversation history or prompt.
#         tools (Optional[List[Tool]]): List of tools available to the LLM.
#         response_format (Optional[Union[dict, Type[BaseModel]]]): Expected response format.
#         temperature (Optional[float]): The sampling temperature to use.
#         max_tokens (Optional[int]): The maximum number of tokens to generate.
#         stream (bool): Whether to stream the response.
#         additional_kwargs (dict): Additional keyword arguments for the LLM API.
#     """

#     model: str
#     messages: List[dict]
#     tools: Optional[List[Tool]] = None
#     response_format: Optional[Union[dict, Type[BaseModel]]] = None
#     temperature: Optional[float] = None
#     max_tokens: Optional[int] = None
#     stream: bool = False
#     additional_kwargs: dict = Field(default_factory=dict)

#     @field_validator("temperature")
#     def validate_temperature(cls, v):
#         if v is not None and (v < 0 or v > 1):
#             raise ValueError("Temperature must be between 0 and 1")
#         return v

#     @field_validator("max_tokens")
#     def validate_max_tokens(cls, v):
#         if v is not None and v <= 0:
#             raise ValueError("max_tokens must be greater than 0")
#         return v

#     @field_validator("messages")
#     def validate_messages(cls, v):
#         if not v:
#             raise ValueError("messages cannot be empty")
#         return v
