from .groq import GroqClient
from .openai import OpenAIClient
from .anthropic import AnthropicClient

MODEL_INFO = {
    "gpt-4o": {"provider": "openai", "client": OpenAIClient, "context_window": 128000},
    "gpt-4o-mini": {
        "provider": "openai",
        "client": OpenAIClient,
        "context_window": 128000,
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "client": OpenAIClient,
        "context_window": 4096,
    },
    "claude-3-5-sonnet-20240620": {
        "provider": "anthropic",
        "client": AnthropicClient,
        "context_window": 200000,
        "max_output_tokens": 8192,
    },
    "claude-3-haiku-20240307": {
        "provider": "anthropic",
        "client": AnthropicClient,
        "context_window": 100000,
        "max_output_tokens": 4096,
    },
    "llama3-groq-70b-8192-tool-use-preview": {
        "provider": "groq",
        "client": GroqClient,
        "context_window": 8192,
    },
    "llama3-groq-8b-8192-tool-use-preview": {
        "provider": "groq",
        "client": GroqClient,
        "context_window": 8192,
    },
    "llama-3.1-70b-versatile": {
        "provider": "groq",
        "client": GroqClient,
        "context_window": 131072,
    },
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "client": GroqClient,
        "context_window": 131072,
    },
    "mixtral-8x7b-32768": {
        "provider": "groq",
        "client": GroqClient,
        "context_window": 32768,
    },
}

# ... (rest of the functions remain the same)


def get_model_context_window(model: str) -> int:
    """Get the context window size for a given model."""
    if model in MODEL_INFO:
        return MODEL_INFO[model]["context_window"]
    else:
        raise ValueError(f"Unknown model: {model}")


def is_model_supported(model: str) -> bool:
    """Check if a model is supported."""
    return model in MODEL_INFO


def get_max_tokens(model: str) -> int:
    model_lower = model.lower()
    model_info = MODEL_INFO.get(model_lower)

    if not model_info:
        raise ValueError(f"No information found for model: {model}")

    return model_info.get("max_output_tokens", model_info["context_window"])


def get_model_provider(model: str) -> str:
    model_lower = model.lower()
    model_info = MODEL_INFO.get(model_lower)

    if not model_info:
        raise ValueError(f"No information found for model: {model}")

    return model_info["provider"]


# def get_llm_client(model: str) -> LLMClient:
#     model_lower = model.lower()
#     model_info = MODEL_INFO.get(model_lower)

#     if not model_info:
#         raise ValueError(f"No information found for model: {model}")

#     client_class = model_info["client"]

#     if not client_class:
#         raise ValueError(f"No client found for model: {model}")

#     return client_class()
