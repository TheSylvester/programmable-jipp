from typing import Callable, Any, List

import nextcord
from jipp import ask_llm, is_model_supported, Conversation
from jipp.llms.llm_selector import (
    MODEL_INFO,
    MODEL_ALIASES,
    get_model_names,
    get_model_profile,
)
from jipp.jipp_fu_suite import ask_llms
from jipp.models.jipp_models import LLMError
from message_chunker import get_full_text_from_message
from task_manager import StopTask, ListTasks, CreateTask


DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"


class Jippity:
    def __init__(
        self, system_prompt: str = DEFAULT_SYSTEM_PROMPT, model: str = DEFAULT_MODEL
    ):
        self.conversation: Conversation | None = None
        self.system_prompt = system_prompt
        self.model = model

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
