import base64
import io
from typing import List, Optional, Dict, Any
from PIL import Image
from openai import OpenAI
from .llm_client import LLMClient
from models.llm_models import Message, MessageContent, OpenAIInput as OpenAIClientInput
from error_handlers.openai_error import handle_openai_error


class OpenAIClient(LLMClient):
    def __init__(self):
        self.client = OpenAI()

    def generate_response(self, input_data: OpenAIClientInput) -> Message:
        try:
            # Process tools if any
            if input_data.tools:
                input_data.tool_choice = input_data.tool_choice or "auto"

            # Process and create image content
            self._process_image_content(input_data)

            # Convert Pydantic model to dict, excluding None values
            input_dict = input_data.model_dump(exclude_none=True)

            # Convert messages to the format expected by OpenAI
            input_dict["messages"] = [
                {
                    "role": m["role"],
                    "content": (
                        m["content"]
                        if isinstance(m["content"], str)
                        else [c for c in m["content"]]
                    ),
                }
                for m in input_dict["messages"]
            ]

            response = self.client.chat.completions.create(**input_dict)
            return Message(
                role=response.choices[0].message.role,
                content=response.choices[0].message.content,
            )
        except Exception as error:
            handle_openai_error(error)
            return Message(
                role="error",
                content=f"An error occurred: {str(error)}",
            )

    def _process_image_content(self, input_data: OpenAIClientInput):
        for message in reversed(input_data.messages):
            if message.role == "user":
                if isinstance(message.content, str):
                    message.content = [
                        MessageContent(type="text", text=message.content)
                    ]
                break

    def _create_image_message_from_url(self, url: str) -> MessageContent:
        return MessageContent(type="image_url", image_url={"url": url})

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
