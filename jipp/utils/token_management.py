import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from typing import List, Callable
from jipp.llms.llm_selector import (
    get_model_context_window,
    get_model_profile,
    ModelProfileNotFoundError,
)
from jipp.models.jipp_models import LLMMessage
from jipp.utils.message_utils import (
    get_text_from_message,
    trim_messages,
    print_messages,
)

# Import only the GPT tokenizer and fallback tokenizer
from jipp.utils.tokenizers.gpt_tokenizer import count_tokens_gpt
from jipp.utils.tokenizers.approximate_tokenizer import count_tokens_approximate

from jipp.utils.logging_utils import log

TokenizerFunc = Callable[[str], int]
MessageTrimmerFunc = Callable[[List[LLMMessage], str, int], List[LLMMessage]]

# Update the model name to one that's supported in your project
DEFAULT_MODEL = "gpt-4o-mini"


def count_tokens(
    text: str, model: str, tokenizer_func: TokenizerFunc | None = None
) -> int:
    """
    Count the number of tokens in a given string using the provided tokenizer function or model.

    Args:
        text (str): The input string to count tokens for.
        model (str): The model name (used if tokenizer_func is not provided).
        tokenizer_func (TokenizerFunc): The tokenizer function to use (optional).

    Returns:
        int: The number of tokens in the input string.
    """
    log.info(f"Counting tokens for text: '{text}'")
    result = 0
    if tokenizer_func is not None:
        try:
            result = tokenizer_func(text)
        except Exception as e:
            log.error(f"Error using provided tokenizer function: {e}")
            result = count_tokens_approximate(text)
    elif not model:
        log.info("No model specified, using approximate tokenizer")
        result = count_tokens_approximate(text)
    else:
        log.info(f"Using model '{model}' to count tokens")
        try:
            model_profile = get_model_profile(model)
            tokenizer_model = model_profile.tokenizer_model
            result = count_tokens_gpt(text, tokenizer_model)
        except ModelProfileNotFoundError as e:
            log.error(f"Error: Model profile not found for '{model}': {e}")
            log.warning("Falling back to approximate tokenization.")
            result = count_tokens_approximate(text)
        except ValueError as e:
            log.error(f"Error: Invalid tokenizer model for '{model}': {e}")
            log.warning("Falling back to approximate tokenization.")
            result = count_tokens_approximate(text)
        except Exception as e:
            log.error(f"Unexpected error occurred while counting tokens: {e}")
            log.warning("Falling back to approximate tokenization.")
            result = count_tokens_approximate(text)

    log.info(f"Token count: {result}")
    return result


def count_tokens_in_messages(messages: List[LLMMessage], model: str) -> int:
    """
    Count the total number of tokens in the text content of multiple messages.

    Args:
        messages (List[LLMMessage]): A list of message objects.
        model (str): The model to use for tokenization.

    Returns:
        int: The total number of tokens in the combined text content of all messages.
    """
    try:
        combined_text = " ".join(
            get_text_from_message(message)
            for message in messages
            if get_text_from_message(message)
        )
        return count_tokens(combined_text, model)
    except ModelProfileNotFoundError as e:
        log.error(f"Error: {e}")
        log.warning("Falling back to approximate tokenization.")
        return count_tokens_approximate(combined_text)
    except Exception as e:
        log.error(f"Unexpected error occurred while counting tokens: {e}")
        log.warning("Falling back to approximate tokenization.")
        return count_tokens_approximate(combined_text)


def trim_messages_with_strategy(
    messages: List[LLMMessage],
    model: str,
    context_window: int,
    trimming_strategy: str = "remove_earliest",
) -> List[LLMMessage]:
    """
    Trim the messages to fit within the context window using the provided trimming strategy.

    Args:
        messages (List[LLMMessage]): A list of message objects.
        model (str): The model to use for tokenization.
        context_window (int): The maximum number of tokens allowed.
        trimming_strategy (str): The strategy to use for trimming messages.

    Returns:
        List[LLMMessage]: The trimmed list of message objects.
    """
    total_tokens = count_tokens_in_messages(messages, model)

    log.info(
        f"\nBefore trimming - Total tokens: {total_tokens}, Context window: {context_window}"
    )
    log.info("Messages before trimming:")
    print_messages(messages)

    if total_tokens > context_window:
        log.info(
            f"Trim Engaged => Total tokens: {total_tokens}, Context window: {context_window}"
        )
        trimmed_messages = trim_messages(
            messages,
            context_window,
            lambda x: count_tokens(x, model),
            trimming_strategy,
        )
        log.info("Messages after trimming:")
        print_messages(trimmed_messages)
        return trimmed_messages

    return messages


def main():
    sample_model = DEFAULT_MODEL
    context_window = 50  # Set a small context window for demonstration

    # Test case 1: Original test case
    log.info("\nTest case 1: Original test case")
    messages1 = [
        LLMMessage(role="system", content="System prompt"),
        LLMMessage(role="user", content="A" * 1000),
    ]
    test_trimming(messages1, sample_model, context_window)

    # Test case 2: Multiple short messages
    log.info("\nTest case 2: Multiple short messages")
    messages2 = [
        LLMMessage(role="system", content="System prompt"),
        LLMMessage(role="user", content="Hello, how are you?"),
        LLMMessage(
            role="assistant",
            content="I'm doing well, thank you. How can I help you today?",
        ),
        LLMMessage(role="user", content="I have a question about Python."),
    ]
    test_trimming(messages2, sample_model, context_window)

    # Test case 3: Mix of short and long messages
    log.info("\nTest case 3: Mix of short and long messages")
    messages3 = [
        LLMMessage(role="system", content="System prompt"),
        LLMMessage(role="user", content="Hello"),
        LLMMessage(role="assistant", content="Hi there! How can I assist you today?"),
        LLMMessage(role="user", content="B" * 500),
        LLMMessage(role="assistant", content="C" * 200),
    ]
    test_trimming(messages3, sample_model, context_window)

    # Test case 4: Larger context window
    log.info("\nTest case 4: Larger context window")
    test_trimming(messages3, sample_model, 100)


def test_trimming(messages, model, context_window):
    log.info(f"\nBefore trimming:")
    print_messages(messages)
    total_tokens = count_tokens_in_messages(messages, model)
    log.info(f"Total tokens: {total_tokens}")

    log.info(f"\nTrimming with context window: {context_window}")
    trimmed_messages = trim_messages_with_strategy(messages, model, context_window)

    log.info("\nAfter trimming:")
    print_messages(trimmed_messages)
    log.info(f"Total tokens: {count_tokens_in_messages(trimmed_messages, model)}")


if __name__ == "__main__":
    main()
