from jipp.jipp_engine import ask_llm
from jipp.models.jipp_models import LLMError
from typing import Dict, Any

DEFAULT_MODEL = "gpt-4o-mini"


async def execute_llm(
    model: str = DEFAULT_MODEL, prompt: str = "", **kwargs
) -> Dict[str, Any]:
    try:
        conversation = await ask_llm(model=model, prompt=prompt, **kwargs)
        return conversation.model_dump()
    except LLMError as e:
        # Log the error or handle it as needed
        raise LLMError(f"Error executing LLM: {str(e)}")
    except Exception as e:
        # Catch any unexpected errors and wrap them in LLMError
        raise LLMError(f"Unexpected error during LLM execution: {str(e)}")
