import tiktoken
import logging

logging.basicConfig(level=logging.INFO)


def count_tokens_gpt(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.warning(f"Model '{model}' not found. Using default encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)
