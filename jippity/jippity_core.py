from typing import Callable, Any

import nextcord
from jipp import ask_llm, is_model_supported, Conversation
from jipp.llms.llm_selector import MODEL_INFO
from jipp.llms.llm_selector import get_model_profile
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
        response = await self.chat(message.content)
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

    def update_model(self, new_model: str) -> str:
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

    # async def stop_a_task(self, prompt):
    #     task_manager = self.bot.get_cog("TaskManager")
    #     tools = [{"function": task_manager.stop_task, "schema": StopTask}]

    #     response = await ask_llm(
    #         model=self.model,
    #         prompt=prompt,
    #         system=self.system_prompt,
    #         conversation=self.conversation,
    #         tools=tools,
    #     )
    #     self.conversation = response
    #     return response

    # async def list_tasks(self, prompt):
    #     task_manager = self.bot.get_cog("TaskManager")
    #     tools = [{"function": task_manager.list_tasks, "schema": ListTasks}]

    #     response = await ask_llm(
    #         model=self.model,
    #         prompt=prompt,
    #         system=self.system_prompt,
    #         conversation=self.conversation,
    #         tools=tools,
    #     )
    #     self.conversation = response
    #     return response

    async def ask_llm_with_tools(self, model: str, prompt: str, tools: list):
        response = await ask_llm(
            model=model,
            prompt=prompt,
            system=self.system_prompt,
            conversation=self.conversation,
            tools=tools,
        )
        self.conversation = response
        return response
