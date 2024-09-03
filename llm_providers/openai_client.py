import base64
import io
from typing import List, Optional, Dict, Any, Union
from PIL import Image
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from jippity_core.tool_calling import (
    create_function_response_message,
    execute_tool_call,
)
from .llm_client import LLMClient
from models.llm_models import (
    Function,
    Message,
    MessageContent,
    OpenAIInput as OpenAIClientInput,
    ToolCall,
)
from error_handlers.openai_error import handle_openai_error


class OpenAIClient(LLMClient):
    def __init__(self):
        self.client = AsyncOpenAI()

    async def generate_response(self, input_data: OpenAIClientInput) -> Message:
        try:
            # Process tools if any
            if input_data.tools:
                input_data.tool_choice = input_data.tool_choice or "auto"

            # Process messages
            messages = self._process_messages(input_data.messages)

            response = await self._create_completion(input_data, messages)

            # Process tool calls if any
            if input_data.tools:
                messages = await self._process_tool_calls(response, input_data.tools)
                if messages:
                    # generate a new response with the tool calls
                    response = await self._create_completion(input_data, messages)

            return self._create_message_from_response(response)
        except Exception as error:
            handle_openai_error(error)
            return Message(
                role="error",
                content=f"An error occurred: {str(error)}",
            )

    async def _create_completion(
        self, input_data: OpenAIClientInput, messages: List[Dict]
    ) -> Union[ChatCompletion, Any]:
        if input_data.response_format and isinstance(
            input_data.response_format, BaseModel
        ):
            return await self._create_structured_chat_completions(input_data, messages)
        else:
            return await self._create_chat_completions(input_data, messages)

    async def _create_chat_completions(
        self, input_data: OpenAIClientInput, messages: List[Dict]
    ) -> ChatCompletion:
        # Convert Pydantic model to dict, excluding None values
        input_dict = input_data.model_dump(exclude_none=True)
        input_dict["messages"] = messages

        return await self.client.chat.completions.create(**input_dict)

    async def _create_structured_chat_completions(
        self, input_data: OpenAIClientInput, messages: List[Dict]
    ) -> Any:
        # Convert Pydantic model to dict, excluding None values
        input_dict = input_data.model_dump(exclude_none=True)
        input_dict["messages"] = messages

        return await self.client.beta.chat.completions.parse(**input_dict)

    def _process_messages(self, messages: List[Message]) -> List[Dict]:
        processed_messages = []
        for message in messages:
            processed_content = self._process_message_content(message.content)
            processed_messages.append(
                {"role": message.role, "content": processed_content}
            )
        return processed_messages

    def _process_message_content(
        self, content: Union[str, List[MessageContent]]
    ) -> Union[str, List[Dict]]:
        if isinstance(content, str):
            return content

        processed_content = []
        for item in content:
            if not isinstance(item, MessageContent):
                raise ValueError(f"Expected MessageContent, got {type(item)}")
            if item.type == "text":
                processed_content.append({"type": "text", "text": item.text})
            elif item.type == "image_url":
                processed_content.append(self._process_image_url(item.image_url))
            else:
                raise ValueError(f"Unsupported content type: {item.type}")

        return processed_content

    def _process_image_url(self, image_url: Dict[str, str]) -> Dict:
        if "url" in image_url:
            return {"type": "image_url", "image_url": image_url}
        elif "filepath" in image_url:
            return {
                "type": "image_url",
                "image_url": self._create_image_message_from_filepath(
                    image_url["filepath"]
                ).image_url,
            }
        else:
            raise ValueError(
                "Image URL dictionary must contain either 'url' or 'filepath' key"
            )

    def _create_image_message_from_filepath(self, filepath: str) -> MessageContent:
        with open(filepath, "rb") as image_file:
            image_bytes = io.BytesIO(image_file.read())
            image = Image.open(image_bytes)
            format = image.format
            base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
            return MessageContent(
                type="image_url",
                image_url={"url": f"data:image/{format.lower()};base64,{base64_image}"},
            )

    async def _process_tool_calls(
        self,
        response: Union[ChatCompletion, Any],
        available_functions: List[Dict[str, Any]],
    ) -> List[Message]:
        messages = []
        tool_calls = self._extract_tool_calls(response)

        for tool_call in tool_calls:
            function_output = await execute_tool_call(tool_call, available_functions)
            function_response = create_function_response_message(
                function_output, tool_call.id, tool_call.function.name
            )
            messages.append(function_response)

        return messages

    def _extract_tool_calls(
        self, response: Union[ChatCompletion, Any]
    ) -> List[ToolCall]:
        tool_calls = []
        if isinstance(response, ChatCompletion):
            for choice in response.choices:
                if choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        function = Function(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        )
                        tool_calls.append(
                            ToolCall(
                                id=tool_call.id, type=tool_call.type, function=function
                            )
                        )
        # Add handling for structured responses if needed
        return tool_calls

    def _create_message_from_response(
        self, response: Union[ChatCompletion, Any]
    ) -> Message:
        if isinstance(response, ChatCompletion):
            return Message(
                role=response.choices[0].message.role,
                content=response.choices[0].message.content,
            )
        else:
            # For structured responses
            return Message(
                role="assistant",
                content=str(response),  # Convert the structured response to a string
            )
