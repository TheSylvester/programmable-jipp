# from .openai_client import ask_openai
# from .groq_client import ask_groq
# from .anthropic_client import ask_claude


from typing import Callable, List

MODEL_ALIASES = {
    "claude-sonnet": "claude-3-5-sonnet-20240620",
    "claude-haiku": "claude-3-haiku-20240307",
    # Add more aliases as needed
}

MODEL_INFO = {
    "gpt-4o": {
        "provider": "openai",
        "context_window": 128000,
        "features": ["image", "response_format", "tools"],
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "context_window": 128000,
        "features": ["image", "response_format", "tools"],
    },
    "claude-3-5-sonnet-20240620": {
        "provider": "anthropic",
        "context_window": 200000,
        "max_output_tokens": 8192,
        "features": ["image", "response_format", "tools"],
    },
    "claude-3-haiku-20240307": {
        "provider": "anthropic",
        "context_window": 100000,
        "max_output_tokens": 4096,
        "features": ["image", "response_format", "tools"],
    },
    "llama3-groq-70b-8192-tool-use-preview": {
        "provider": "groq",
        "context_window": 8192,
        "features": ["response_format", "tools"],
    },
    "llama3-groq-8b-8192-tool-use-preview": {
        "provider": "groq",
        "context_window": 8192,
        "features": ["response_format", "tools"],
    },
    "llama-3.1-70b-versatile": {
        "provider": "groq",
        "context_window": 131072,
        "features": ["response_format"],
    },
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "context_window": 131072,
        "features": ["response_format"],
    },
    "mixtral-8x7b-32768": {
        "provider": "groq",
        "context_window": 32768,
        "features": ["response_format"],
    },
}


class ModelProfile:
    model: str
    provider: str
    context_window: int
    max_output_tokens: int
    features: List[str]

    def __init__(self, model: str, model_info: dict):
        self.model = model
        self.provider = model_info["provider"]
        self.context_window = model_info["context_window"]
        self.max_output_tokens = model_info.get(
            "max_output_tokens", self.context_window
        )
        self.features = model_info["features"]

    def __getattr__(self, item):
        return item in self.features

    def __str__(self) -> str:
        return ", ".join(self.features)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def client(self) -> Callable:
        provider_name = self.provider
        if provider_name is None:
            raise ValueError(f"Unknown provider: {provider_name}")
        if provider_name is "openai":
            from .openai_client import ask_openai

            return ask_openai
        elif provider_name is "anthropic":
            from .anthropic_client import ask_claude

            return ask_claude
        elif provider_name is "groq":
            from .groq_client import ask_groq

            return ask_groq


def resolve_model_alias(model: str) -> str:
    """Resolve a model alias to its full name."""
    return MODEL_ALIASES.get(model, model)


def get_model_context_window(model: str) -> int:
    """Get the context window size for a given model."""
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
        raise ValueError(f"Model {model} is not supported")

    return ModelProfile(model, model_info)
