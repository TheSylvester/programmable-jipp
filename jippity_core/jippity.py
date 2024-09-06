from typing import Callable, Any
import re
from jipp import ChatLLM, AskLLMInput, LLMResponse  # Updated imports
from llm_providers.llm_selector import is_model_supported
from models.message_context import MessageContext

DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"


class Jippity:
    def __init__(
        self, system_prompt: str = DEFAULT_SYSTEM_PROMPT, model: str = DEFAULT_MODEL
    ):
        self.last_response: LLMResponse | None = None
        self.system_prompt = system_prompt
        self.model = model
        self.chat_llm = ChatLLM()  # Initialize ChatLLM instance

    async def handle_message(
        self, context: MessageContext, send_response: Callable[[str], Any]
    ):
        response = await self.get_response(context)
        await self.send_response(response, send_response)

    async def get_response(self, context: MessageContext) -> str:
        input_data = AskLLMInput(
            model=self.model,
            prompt=context.content,
            system=self.system_prompt,
            last_response=self.last_response,
            additional_kwargs={},
        )

        self.last_response = await self.chat_llm.ask_llm(input_data)  # Updated call

        return str(self.last_response) if self.last_response else ""

    async def send_response(self, response: str, send_function: Callable[[str], Any]):
        """
        Send a response while ensuring it doesn't exceed Discord's message length limit.
        The response is first split by newline characters, then by whitespace, and finally by character count if necessary.

        Args:
            response (str): The response to send.
            send_function (Callable[[str], Any]): The function to call to send the response.
        """
        max_length = 2000

        if len(response) <= max_length:
            await send_function(response)
            return

        lines = response.split("\n")
        current_chunk = ""

        for line in lines:
            if len(current_chunk) + len(line) + 1 > max_length:
                if len(line) > max_length:
                    # Further split the line by whitespace if it's too long
                    words = re.split(r"(\s+)", line)
                    for word in words:
                        if len(current_chunk) + len(word) > max_length:
                            await send_function(current_chunk)
                            current_chunk = word
                        else:
                            current_chunk += word

                    # If still too long, split by character count
                    while len(current_chunk) > max_length:
                        await send_function(current_chunk[:max_length])
                        current_chunk = current_chunk[max_length:]
                else:
                    await send_function(current_chunk)
                    current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line

        # Send any remaining text
        while len(current_chunk) > max_length:
            await send_function(current_chunk[:max_length])
            current_chunk = current_chunk[max_length:]

        if current_chunk:
            await send_function(current_chunk)

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
        self.last_response = None
        return "Context window cleared"

    def list_models(self) -> str:
        from jipp.llms.llm_selector import MODEL_INFO
        from jipp.llms.llm_selector import get_model_profile

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
