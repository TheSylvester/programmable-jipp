import os
from typing import List, Optional, Dict, Any
from groq import Groq
from .llm_client import LLMClient
from models.llm_models import Message, MessageContent, OpenAIInput as GroqClientInput
from error_handlers.groq_error import handle_groq_error


class GroqClient(LLMClient):
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def generate_response(self, input_data: GroqClientInput) -> Message:
        try:
            # Process tools if any
            if input_data.tools:
                input_data.tool_choice = input_data.tool_choice or "auto"

            # Process and create image content
            self._process_image_content(input_data)

            # Convert Pydantic model to dict, excluding None values
            input_dict = input_data.model_dump(exclude_none=True)

            # Convert messages to the format expected by Groq (OpenAI-compatible)
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
            handle_groq_error(error)
            return Message(
                role="error",
                content=f"An error occurred: {str(error)}",
            )

    def _process_image_content(self, input_data: GroqClientInput):
        for message in reversed(input_data.messages):
            if message.role == "user":
                if isinstance(message.content, str):
                    message.content = [
                        MessageContent(type="text", text=message.content)
                    ]
                break

    def _create_image_message_from_url(self, url: str) -> MessageContent:
        # Groq doesn't support image inputs, so we'll return a text message instead
        return MessageContent(type="text", text=f"Image URL: {url}")

    def _create_image_message_from_filepath(self, filepath: str) -> MessageContent:
        # Groq doesn't support image inputs, so we'll return a text message instead
        return MessageContent(type="text", text=f"Image file: {filepath}")
