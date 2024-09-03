from abc import ABC, abstractmethod
from typing import List, Optional
from jipp.models.common import Tool
from jipp.models.requests import AskLLMInput
from jipp.models.responses import LLMResponse


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate_response(self, input_data: AskLLMInput) -> LLMResponse:
        raise NotImplementedError

    def _convert_tools_to_openai_format(
        self, tools: Optional[List[Tool]]
    ) -> List[dict]:
        if not tools:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": (
                        tool.parameters.model_json_schema() if tool.parameters else {}
                    ),
                },
            }
            for tool in tools
        ]
