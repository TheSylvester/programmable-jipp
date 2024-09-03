from typing import List
import tiktoken
from transformers import AutoTokenizer

from jipp.models.llm_models import Message, MessageContent
from jipp.llms.llm_selector import get_model_context_window


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a given string using the appropriate tokenizer based on the model name.
    Falls back to an estimated token count if the model is not recognized.

    Args:
        text (str): The input string to count tokens for.
        model (str): The model to use for tokenization (default: "gpt-3.5-turbo").

    Returns:
        int: The number of tokens in the input string.
    """
    TOKENIZER_MODEL_MAP = {
        "llama": "decapoda-research/llama-7b-hf",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "mixtral": "mistralai/Mistral-7B-v0.1",
    }
    if "gpt" in model:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print(f"Warning: model '{model}' not found. Using default encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)

    elif any(name in model.lower() for name in TOKENIZER_MODEL_MAP.keys()):

        tokenizer_model = None
        for name in TOKENIZER_MODEL_MAP.keys():
            if name in model.lower():
                tokenizer_model = TOKENIZER_MODEL_MAP[name]
                break

        if tokenizer_model is None:
            print(
                f"Warning: model '{model}' not found. Using character-to-token estimate."
            )
            estimated_token_count = (
                len(text) // 3
            )  # Estimate: average 1 token per 3 characters
            return estimated_token_count

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            tokens = tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"Error: {e}")
            print(
                f"Error: model '{model}' not found. Using character-to-token estimate.\n{e}"
            )
            estimated_token_count = (
                len(text) // 3
            )  # Estimate: average 1 token per 3 characters
            return estimated_token_count

    elif "claude" in model.lower():
        print(f"Using character-to-token estimate for Claude model '{model}'.")
        estimated_token_count = (
            len(text) // 3
        )  # Rough estimate: average 1 token per 3 characters
        return estimated_token_count

    else:
        print(
            f"Warning: Unrecognized model '{model}'. Using character-to-token estimate."
        )
        estimated_token_count = (
            len(text) // 3
        )  # Estimate: average 1 token per 3 characters
        return estimated_token_count


def get_text_from_message(message: Message) -> str:
    """
    Extract the usable text content from a Message object.

    Args:
        message (Message): A single Message object.

    Returns:
        str: The text content of the message if the role is 'system', 'user', or 'assistant'.
             Returns an empty string for other roles.
    """
    if message.role in {"system", "user", "assistant"}:
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list):
            # If content is a list of MessageContent objects, concatenate the text fields
            return " ".join(
                content.text for content in message.content if content.text is not None
            )
    return ""


def count_tokens_in_messages(messages: list, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the total number of tokens in the text content of multiple messages.

    Args:
        messages (list): A list of message objects.
        model (str): The model to use for tokenization (default: "gpt-3.5-turbo").

    Returns:
        int: The total number of tokens in the combined text content of all messages.
    """
    combined_text = " ".join(
        get_text_from_message(message)
        for message in messages
        if get_text_from_message(message)
    )
    return count_tokens(combined_text, model=model)


def trim_messages_to_context_window(
    messages: List[Message], model: str = "gpt-3.5-turbo"
) -> List[Message]:
    """
    Trim the messages to fit within the context window by removing the oldest non-system messages.

    Args:
        messages (List[Message]): A list of message objects.
        context_window (int): The maximum number of tokens allowed in the context window.
        model (str): The model to use for tokenization (default: "gpt-3.5-turbo").

    Returns:
        List[Message]: The trimmed list of message objects.
    """

    # Calculate the current token count
    total_tokens = count_tokens_in_messages(messages, model)
    context_window = get_model_context_window(model)

    print(f"\nTotal tokens: {total_tokens}, Context window: {context_window}\n")

    # Trim oldest non-system messages until the total tokens fit within the context window
    while total_tokens > context_window:
        print(
            f"Trim Engaged => Total tokens: {total_tokens}, Context window: {context_window}"
        )
        for i, message in enumerate(messages):
            if message.role == "system":
                continue
            messages.pop(i)
            total_tokens = count_tokens_in_messages(messages, model)
            break  # Exit the loop after removing one message

    return messages
