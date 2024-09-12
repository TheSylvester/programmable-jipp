import pytest
from jipp.models.jipp_models import LLMMessage
from jipp.utils.token_management import (
    count_tokens,
    count_tokens_in_messages,
    trim_messages_with_strategy,
    DEFAULT_MODEL,
)


@pytest.fixture
def sample_model():
    return DEFAULT_MODEL


@pytest.fixture
def context_window():
    return 50


def test_count_tokens(sample_model):
    text = "This is a sample text to demonstrate token counting."
    token_count = count_tokens(text, sample_model)
    assert token_count > 0, "Token count should be greater than 0"


@pytest.mark.parametrize(
    "test_case,expected_trimmed_count,expected_context_window",
    [
        ("original", 1, 50),
        ("multiple_short", 4, 50),
        ("mix_short_long", 1, 50),
        ("larger_context", 2, 100),
    ],
)
def test_trim_messages(
    test_case, expected_trimmed_count, expected_context_window, sample_model
):
    if test_case == "original":
        messages = [
            LLMMessage(role="system", content="System prompt"),
            LLMMessage(role="user", content="A" * 1000),
        ]
    elif test_case == "multiple_short":
        messages = [
            LLMMessage(role="system", content="System prompt"),
            LLMMessage(role="user", content="Hello, how are you?"),
            LLMMessage(
                role="assistant",
                content="I'm doing well, thank you. How can I help you today?",
            ),
            LLMMessage(role="user", content="I have a question about Python."),
        ]
    elif test_case == "mix_short_long":
        messages = [
            LLMMessage(role="system", content="System prompt"),
            LLMMessage(role="user", content="Hello"),
            LLMMessage(
                role="assistant", content="Hi there! How can I assist you today?"
            ),
            LLMMessage(role="user", content="B" * 500),
            LLMMessage(role="assistant", content="C" * 200),
        ]
    else:  # larger_context
        messages = [
            LLMMessage(role="system", content="System prompt"),
            LLMMessage(role="user", content="Hello"),
            LLMMessage(
                role="assistant", content="Hi there! How can I assist you today?"
            ),
            LLMMessage(role="user", content="B" * 500),
            LLMMessage(role="assistant", content="C" * 200),
        ]

    total_tokens_before = count_tokens_in_messages(messages, sample_model)

    trimmed_messages = trim_messages_with_strategy(
        messages, sample_model, expected_context_window
    )

    total_tokens_after = count_tokens_in_messages(trimmed_messages, sample_model)

    print(f"Test case: {test_case}")
    print(f"Total tokens before: {total_tokens_before}")
    print(f"Total tokens after: {total_tokens_after}")
    print(f"Context window: {expected_context_window}")
    print(f"Number of messages after trimming: {len(trimmed_messages)}")
    for i, msg in enumerate(trimmed_messages):
        print(f"Message {i + 1}: {msg.role} - {msg.content[:50]}...")

    assert (
        len(trimmed_messages) == expected_trimmed_count
    ), f"Expected {expected_trimmed_count} messages after trimming, but got {len(trimmed_messages)}"
    assert (
        total_tokens_after <= expected_context_window
    ), f"Total tokens after trimming ({total_tokens_after}) should be less than or equal to the context window ({expected_context_window})"

    if total_tokens_before > expected_context_window:
        assert (
            total_tokens_after < total_tokens_before
        ), "Total tokens should decrease after trimming when initial tokens exceed the context window"


def test_count_tokens_in_messages(sample_model):
    messages = [
        LLMMessage(role="system", content="System prompt"),
        LLMMessage(role="user", content="Hello, how are you?"),
        LLMMessage(role="assistant", content="I'm doing well, thank you."),
    ]
    total_tokens = count_tokens_in_messages(messages, sample_model)
    assert total_tokens > 0, "Total token count should be greater than 0"
    assert total_tokens == sum(
        count_tokens(msg.content, sample_model) for msg in messages
    ), "Total tokens should match the sum of individual message tokens"
