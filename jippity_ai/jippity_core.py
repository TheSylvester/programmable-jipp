from typing import Callable, Any, List
import os

import nextcord
from pydantic import BaseModel, ConfigDict, Field
from jipp import ask_llm, is_model_supported, Conversation
from jipp.llms.llm_selector import (
    MODEL_INFO,
    MODEL_ALIASES,
    get_model_names,
    get_model_profile,
)
from jipp.jipp_fu_suite import ask_llms
from jipp.utils.logging_utils import setup_logger
from bot_base.message_chunker import get_full_text_from_message

log = setup_logger()


class PromptManager:
    def __init__(self):
        self._prompts_dir = None

    def load_prompts(self, directory: str) -> None:
        self._prompts_dir = directory

    def __getattr__(self, name: str) -> str:
        if self._prompts_dir is None:
            raise RuntimeError("Prompts directory not set. Call load_prompts first.")

        file_path = os.path.join(self._prompts_dir, f"{name}.md")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        raise AttributeError(f"No prompt named '{name}' found")

    def list_prompts(self) -> List[str]:
        if self._prompts_dir is None:
            raise RuntimeError("Prompts directory not set. Call load_prompts first.")
        return [
            os.path.splitext(f)[0]
            for f in os.listdir(self._prompts_dir)
            if f.endswith(".md")
        ]


PROMPTS = PromptManager()


class RespondToMessage(BaseModel):
    """Prompts an LLM to respond to a message."""

    context: str = Field(
        ...,
        description="The relevant context of this conversation, including only relevant messages from the history and the current message",
    )
    thoughts: str = Field(
        ...,
        description="Thoughts on why you are responding to this message, and ",
    )
    # model_config = ConfigDict(strict=True)
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"required": ["context", "thoughts"]}
    )


class Jippity:
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant",
        model: str = "llama-3.1-8b-instant",
        bot_name: str = "",
    ):
        self.conversation: Conversation | None = None
        self.system_prompt = system_prompt
        self.model = model
        self.bot_name = bot_name

    async def message_listener(
        self, message: str, channel_history: str, send_response: Callable[[str], Any]
    ):

        system_prompt = PROMPTS.MESSAGE_LISTENER_SYSTEM_PROMPT
        prompt = PROMPTS.MESSAGE_LISTENER_PROMPT

        async def respond_to_message(thoughts: str = "", context: str = ""):
            response_string = f"\n\n**message:**\n{message}\n\n**context:**\n{context}\n\n**thoughts:**\n{thoughts}\n"
            await send_response(response_string)
            return response_string

        print("\n\n", RespondToMessage.model_json_schema(), "\n\n")

        response = await ask_llm(
            model="gpt-4o-mini",
            prompt=prompt,
            system=system_prompt,
            tool_choice="auto",
            tools=[
                {
                    "schema": RespondToMessage,
                    "function": respond_to_message,
                }
            ],
            bot_username=self.bot_name,
            history_string=channel_history,
            message_string=message,
        )
        response_text: str = str(response)
        log.debug(f"message_listener: {response_text}\n\n")

    async def chat_response_with_history_and_thoughts(
        self,
        message: nextcord.Message,
        send_response: Callable[[str], Any],
        history: str,
        thoughts: str,
    ):
        """Continues LLM chat with text contents from a nextcord Message, directly uses the send_response fn to reply"""

        content = await get_full_text_from_message(message)
        response = await self.chat(content)
        response_text: str = str(response)
        print(f"Conversation in chat_response: ", response.model_dump_json(indent=2))
        await send_response(str(response_text))
        return response

    async def chat_response(
        self, message: nextcord.Message, send_response: Callable[[str], Any]
    ):
        """Continues LLM chat with text contents from a nextcord Message, directly uses the send_response fn to reply"""

        content = await get_full_text_from_message(message)
        response = await self.chat(content)
        response_text: str = str(response)
        print(f"Conversation in chat_response: ", response.model_dump_json(indent=2))
        await send_response(str(response_text))
        return response

    async def chat(self, message: str) -> Conversation:
        """Continues the conversation"""
        response = await ask_llm(
            model=self.model,
            prompt=str(message),
            system=self.system_prompt,
            conversation=self.conversation,
        )
        self.conversation = response
        return response

    def update_system_prompt(self, new_prompt: str) -> str:
        self.system_prompt = new_prompt
        return f"System prompt updated to: {new_prompt}"

    def update_model(self, new_model: str = None) -> str:
        if not new_model:
            return "Current model: " + self.model
        if not is_model_supported(new_model):
            return f"Model {new_model} not supported"
        self.model = new_model
        return f"Model updated to: {new_model}"

    def clear_context_window(self) -> str:
        self.conversation = None
        return "Context window cleared"

    def list_models(self) -> str:
        model_info_str = "Available models and their details:\n\n"
        for model in MODEL_INFO.keys():
            model_profile = get_model_profile(model)
            model_info_str += f"Model: {model}\n"
            model_info_str += f"  Provider: {model_profile.provider}\n"
            model_info_str += f"  Context Window: {model_profile.context_window}\n"
            model_info_str += (
                f"  Max Output Tokens: {model_profile.max_output_tokens}\n"
            )
            model_info_str += f"  Features: {', '.join(model_profile.features)}\n"
            model_info_str += "\n"
        return model_info_str

    def get_model_aliases(self):
        return MODEL_ALIASES

    def get_model_names(self):
        return get_model_names()

    async def ask_multiple_llms(self, models: List[str], prompt: str):
        try:
            results = await ask_llms(models=models, prompt=prompt)
            return results
        except Exception as e:
            raise Exception(f"Error in ask_multiple_llms: {str(e)}")

    async def ask_llm_with_tools(self, model: str, prompt: str, tools: list):
        response = await ask_llm(
            model=model,
            prompt=prompt,
            system=self.system_prompt,
            conversation=self.conversation,
            tools=tools,
            tool_choice="auto",
        )
        self.conversation = response
        return response

    def load_prompts(self, directory: str) -> None:
        PROMPTS.load_prompts(directory)

    def get_prompt(self, name: str) -> str:
        return getattr(PROMPTS, name)

    def list_prompts(self) -> List[str]:
        return PROMPTS.list_prompts()

    def get_prompt_content(self, name: str) -> str:
        prompt = self.get_prompt(name)
        return prompt.content if prompt else ""

    def set_bot_name(self, bot_name: str):
        self.bot_name = bot_name
