# error_handlers/__init__.py
from .openai_error import handle_openai_error, OpenAIErrorHandlers
from .anthropic_error import handle_anthropic_error, AnthropicErrorHandlers
from .groq_error import handle_groq_error, GroqErrorHandlers

# Optionally, you can create a registry for easier imports
error_registry = {
    "openai": handle_openai_error,
    "anthropic": handle_anthropic_error,
    "groq": handle_groq_error,
}

__all__ = [
    "handle_openai_error",
    "OpenAIErrorHandlers",
    "handle_anthropic_error",
    "AnthropicErrorHandlers",
    "handle_llm_error",
    "error_registry",
    "handle_groq_error",
    "GroqErrorHandlers",
]


def handle_llm_error(llm_type, error):
    """Handle errors based on the LLM type by delegating to the appropriate error handler."""
    handler = error_registry.get(llm_type)
    if handler:
        handler(error)
    else:
        print(f"Error with {llm_type}: {error}")
