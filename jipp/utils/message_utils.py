from typing import Union, List, Callable
from jipp.models.jipp_models import LLMMessage, MessageContentText, MessageContentImage

TokenizerFunc = Callable[[str], int]


def get_text_from_message(message: LLMMessage) -> str:
    if not hasattr(message, "content") or message.content is None:
        return ""

    if isinstance(message.content, str):
        return message.content
    elif isinstance(message.content, list):
        return " ".join(
            content.get("text", "")
            for content in message.content
            if isinstance(content, dict) and "text" in content
        )
    return ""


def truncate_message(
    message: LLMMessage, max_tokens: int, tokenizer_func: TokenizerFunc
) -> LLMMessage:
    text = get_text_from_message(message)
    token_count = tokenizer_func(text)

    if token_count <= max_tokens:
        return message

    # Improved truncation method
    words = text.split()
    truncated_words = []
    current_tokens = 0
    for word in words:
        word_tokens = tokenizer_func(word)
        if current_tokens + word_tokens > max_tokens:
            break
        truncated_words.append(word)
        current_tokens += word_tokens

    truncated_text = " ".join(truncated_words)
    return LLMMessage(role=message.role, content=truncated_text)


def trim_messages(
    messages: List[LLMMessage],
    max_tokens: int,
    tokenizer_func: TokenizerFunc,
    strategy: str = "remove_earliest",
) -> List[LLMMessage]:
    total_tokens = sum(tokenizer_func(get_text_from_message(msg)) for msg in messages)
    print(f"Initial total tokens: {total_tokens}, Max tokens: {max_tokens}")

    if total_tokens <= max_tokens:
        return messages

    if strategy == "remove_earliest":
        while total_tokens > max_tokens and len(messages) > 1:
            for i, msg in enumerate(messages):
                if i > 0 and msg.role != "system":
                    removed_msg = messages.pop(i)
                    removed_tokens = tokenizer_func(get_text_from_message(removed_msg))
                    total_tokens -= removed_tokens
                    print(
                        f"Removed message {i} ({removed_tokens} tokens). New total: {total_tokens}"
                    )
                    break
    elif strategy == "truncate_long":
        for i, msg in enumerate(messages):
            msg_tokens = tokenizer_func(get_text_from_message(msg))
            if msg_tokens > max_tokens:
                old_tokens = msg_tokens
                messages[i] = truncate_message(msg, max_tokens, tokenizer_func)
                new_tokens = tokenizer_func(get_text_from_message(messages[i]))
                total_tokens = sum(
                    tokenizer_func(get_text_from_message(msg)) for msg in messages
                )
                print(
                    f"Truncated message {i} from {old_tokens} to {new_tokens} tokens. New total: {total_tokens}"
                )
            if total_tokens <= max_tokens:
                break

    while total_tokens > max_tokens and len(messages) > 1:
        removed_msg = messages.pop(-1)
        removed_tokens = tokenizer_func(get_text_from_message(removed_msg))
        total_tokens -= removed_tokens
        print(
            f"Removed last message ({removed_tokens} tokens). New total: {total_tokens}"
        )

    print(f"Final total tokens after trimming: {total_tokens}")
    return messages


def print_messages(messages: List[LLMMessage]):
    for i, msg in enumerate(messages):
        print(f"Message {i + 1}:")
        print(f"  Role: {msg.role}")
        print(f"  Content: {get_text_from_message(msg)}")
        print()
