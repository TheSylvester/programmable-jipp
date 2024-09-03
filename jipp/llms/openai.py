# from typing import List, Dict, Any
# import json
# import asyncio
# from openai import AsyncOpenAI
# from pydantic import BaseModel
# from jipp.llms.base import BaseLLMClient
# from jipp.models.requests import AskLLMInput
# from jipp.models.responses import LLMResponse
# from jipp.config.settings import settings
# from jipp.utils import log_exceptions


# class OpenAIClient(BaseLLMClient):
#     def __init__(self):
#         self.client = AsyncOpenAI(api_key=settings.openai_api_key)

#     @log_exceptions
#     async def generate_response(self, input_data: AskLLMInput) -> LLMResponse:
#         try:
#             openai_tools = self._convert_tools_to_openai_format(input_data.tools)

#             if isinstance(input_data.response_format, type) and issubclass(
#                 input_data.response_format, BaseModel
#             ):
#                 return await self._structured_completion(input_data, openai_tools)
#             else:
#                 return await self._standard_completion(input_data, openai_tools)
#         except Exception as e:
#             print(f"An error occurred in OpenAIClient: {str(e)}")
#             raise

#     async def _structured_completion(
#         self, input_data: AskLLMInput, openai_tools: List[dict]
#     ) -> LLMResponse:
#         response = await self.client.chat.completions.create(
#             model=input_data.model,
#             messages=input_data.messages,
#             response_format={"type": "json_object"},
#             tools=openai_tools,
#             temperature=input_data.temperature,
#             max_tokens=input_data.max_tokens,
#             stream=input_data.stream,
#             **input_data.additional_kwargs,
#         )
#         content = json.loads(response.choices[0].message.content)
#         parsed_content = input_data.response_format(**content)
#         return LLMResponse(raw_response=response, parsed_content=parsed_content)

#     async def _standard_completion(
#         self, input_data: AskLLMInput, openai_tools: List[dict]
#     ) -> LLMResponse:
#         response = await self.client.chat.completions.create(
#             model=input_data.model,
#             messages=input_data.messages,
#             tools=openai_tools,
#             temperature=input_data.temperature,
#             max_tokens=input_data.max_tokens,
#             stream=input_data.stream,
#             **input_data.additional_kwargs,
#         )

#         if input_data.tools:
#             response = await self._handle_tool_calls(response, input_data)

#         return LLMResponse(raw_response=response)

#     async def _handle_tool_calls(self, response, input_data: AskLLMInput):
#         tool_calls = response.choices[0].message.tool_calls
#         if not tool_calls:
#             return response

#         tool_results = []
#         for tool_call in tool_calls:
#             tool = next(
#                 (t for t in input_data.tools if t.name == tool_call.function.name), None
#             )
#             if tool:
#                 args = json.loads(tool_call.function.arguments)
#                 result = await asyncio.to_thread(tool, **args)
#                 tool_results.append(
#                     {
#                         "tool_call_id": tool_call.id,
#                         "role": "tool",
#                         "name": tool.name,
#                         "content": str(result),
#                     }
#                 )

#         new_messages = input_data.messages + [
#             {"role": "assistant", "content": None, "tool_calls": tool_calls},
#             *tool_results,
#         ]

#         return await self.client.chat.completions.create(
#             model=input_data.model,
#             messages=new_messages,
#             tools=self._convert_tools_to_openai_format(input_data.tools),
#             temperature=input_data.temperature,
#             max_tokens=input_data.max_tokens,
#             stream=input_data.stream,
#             **input_data.additional_kwargs,
#         )

#     def _create_image_message_from_url(self, url: str) -> dict:
#         return {
#             "type": "image_url",
#             "image_url": {"url": url},
#         }

#     def _create_image_message_from_filepath(self, filepath: str) -> dict:
#         with open(filepath, "rb") as image_file:
#             return {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
#                 },
#             }

#     def _download_and_encode_image(self, url: str) -> str:
#         response = requests.get(url)
#         if response.status_code == 200:
#             return base64.b64encode(response.content).decode("utf-8")
#         else:
#             raise ValueError(f"Failed to download image from {url}")
