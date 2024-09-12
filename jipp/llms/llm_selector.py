# from .openai_client import ask_openai
# from .groq_client import ask_groq
# from .anthropic_client import ask_claude
from typing import Callable, List

from jipp.utils.tokenizers.approximate_tokenizer import count_tokens_approximate
from jipp.utils.tokenizers.gpt_tokenizer import count_tokens_gpt

MODEL_ALIASES = {
    "claude-sonnet": "claude-3-5-sonnet-20240620",
    "claude-haiku": "claude-3-haiku-20240307",
    "gpt-4": "gpt-4o-mini",
    "gpt-4-mini": "gpt-4o-mini",
    "llama-tool-large": "llama3-groq-70b-8192-tool-use-preview",
    "llama-tool-small": "llama3-groq-8b-8192-tool-use-preview",
    "llama-large": "llama-3.1-70b-versatile",
    "llama-small": "llama-3.1-8b-instant",
    "mixtral": "mixtral-8x7b-32768",
}

MODEL_TOKENIZER_MAP = {}


MODEL_INFO = {
    "gpt-4o": {
        "provider": "openai",
        "context_window": 128000,
        "features": ["image", "response_format", "tools"],
        "tokenizer_model": "gpt-3.5-turbo",  # OpenAI uses gpt-3.5-turbo tokenizer for gpt-4
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "context_window": 128000,
        "features": ["image", "response_format", "tools"],
        "tokenizer_model": "gpt-3.5-turbo",
    },
    "claude-3-5-sonnet-20240620": {
        "provider": "anthropic",
        "context_window": 200000,
        "max_output_tokens": 8192,
        "features": ["image", "response_format", "tools"],
        "tokenizer_model": "claude-3-sonnet-20240229",  # Use the Claude model identifier
    },
    "claude-3-haiku-20240307": {
        "provider": "anthropic",
        "context_window": 100000,
        "max_output_tokens": 4096,
        "features": ["image", "response_format", "tools"],
        "tokenizer_model": "claude-3-haiku-20240307",
    },
    "llama3-groq-70b-8192-tool-use-preview": {
        "provider": "groq",
        "context_window": 8192,
        "features": ["response_format", "tools"],
        "tokenizer_model": "meta-llama/Llama-2-70b-hf",  # Use HuggingFace model identifier
    },
    "llama3-groq-8b-8192-tool-use-preview": {
        "provider": "groq",
        "context_window": 8192,
        "features": ["response_format", "tools"],
        "tokenizer_model": "meta-llama/Llama-2-7b-hf",  # Use HuggingFace model identifier
    },
    "llama-3.1-70b-versatile": {
        "provider": "groq",
        "context_window": 131072,
        "features": ["response_format"],
        "tokenizer_model": "meta-llama/Llama-2-70b-hf",
    },
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "context_window": 131072,
        "features": ["response_format"],
        "tokenizer_model": "meta-llama/Llama-2-7b-hf",
    },
    "mixtral-8x7b-32768": {
        "provider": "groq",
        "context_window": 32768,
        "features": ["response_format"],
        "tokenizer_model": "mistralai/Mixtral-8x7B-v0.1",  # Use HuggingFace model identifier
    },
}


class ModelProfile:
    model: str
    provider: str
    context_window: int
    max_output_tokens: int
    features: List[str]
    tokenizer_model: str  # New attribute

    def __init__(self, model: str, model_info: dict):
        self.model = model
        self.provider = model_info["provider"]
        self.context_window = model_info["context_window"]
        self.max_output_tokens = model_info.get(
            "max_output_tokens", self.context_window
        )
        self.features = model_info["features"]
        self.tokenizer_model = model_info.get("tokenizer_model", model)  # New line

    # def __getattr__(self, item):
    #     return item in self.features

    def __str__(self) -> str:
        return ", ".join(self.features)

    def __repr__(self) -> str:
        return self.__str__()

    def has_feature(self, item: str) -> bool:
        return item in self.features

    @property
    def client(self) -> Callable:
        provider_name = self.provider
        if provider_name is None:
            raise ValueError(f"Unknown provider: {provider_name}")
        if provider_name == "openai":
            from .openai_client import ask_openai

            return ask_openai
        elif provider_name == "anthropic":
            from .anthropic_client import ask_claude

            return ask_claude
        elif provider_name == "groq":
            from .groq_client import ask_groq

            return ask_groq


class ModelProfileNotFoundError(Exception):
    pass


def resolve_model_alias(model: str) -> str:
    """Resolve a model alias to its full name."""
    return MODEL_ALIASES.get(model, model)


def get_model_context_window(model: str) -> int:
    """
    Get the context window size for a given model.

    Args:
        model (str): The name of the model.

    Returns:
        int: The context window size for the given model.

    Raises:
        ValueError: If the model is unknown or not supported.
    """
    model = resolve_model_alias(model)
    if model in MODEL_INFO:
        return MODEL_INFO[model]["context_window"]
    else:
        raise ValueError(f"Unknown model: {model}")


def is_model_supported(model: str) -> bool:
    """Check if a model is supported."""
    model = resolve_model_alias(model)
    return model in MODEL_INFO


def get_max_tokens(model: str) -> int:
    model = resolve_model_alias(model.lower())
    model_info = MODEL_INFO.get(model)

    if not model_info:
        raise ValueError(f"No information found for model: {model}")

    return model_info.get("max_output_tokens", model_info["context_window"])


def get_model_profile(model: str) -> ModelProfile:
    model = resolve_model_alias(model.lower())
    model_info = MODEL_INFO.get(model)

    if not model_info:
        raise ModelProfileNotFoundError(f"Model {model} is not supported")

    return ModelProfile(model, model_info)


def get_tokenizer(model: str):
    model_info = MODEL_INFO.get(resolve_model_alias(model))
    if not model_info:
        return count_tokens_approximate

    tokenizer_model = model_info.get("tokenizer_model")
    if tokenizer_model:
        return lambda text: count_tokens_gpt(text, tokenizer_model)
    else:
        return count_tokens_approximate
