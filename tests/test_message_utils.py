import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import List
from nextcord import Attachment, Embed, Message
from message_chunker import (
    get_full_text_from_message,
    get_image_urls_from_message,
    send_chunked_message,
    chunk_message_md_friendly,
)


@pytest.mark.asyncio
async def test_get_full_text_from_message_no_attachments():
    # Mock a message with no attachments
    message = MagicMock()
    message.content = "Hello, this is a test message."
    message.attachments = []

    result = await get_full_text_from_message(message)
    assert result == "Hello, this is a test message."


@pytest.mark.asyncio
async def test_get_full_text_from_message_with_txt_attachment():
    # Mock a message with a .txt attachment
    message = MagicMock()
    message.content = "Hello, this is a test message."

    attachment = MagicMock(spec=Attachment)
    attachment.filename = "test.txt"
    attachment.read = AsyncMock(return_value=b"This is content from the text file.")

    message.attachments = [attachment]

    result = await get_full_text_from_message(message)
    expected_result = (
        "Hello, this is a test message.\nThis is content from the text file."
    )
    assert result == expected_result


@pytest.mark.asyncio
async def test_get_full_text_from_message_with_invalid_txt_attachment():
    # Mock a message with a .txt attachment that has a UnicodeDecodeError
    message = MagicMock()
    message.content = "Hello, this is a test message."

    attachment = MagicMock(spec=Attachment)
    attachment.filename = "test.txt"
    attachment.read = AsyncMock(return_value=b"\xff\xfe")

    # Simulate a UnicodeDecodeError during decoding
    attachment.read.side_effect = UnicodeDecodeError(
        "utf-8", b"\xff\xfe", 0, 1, "invalid start byte"
    )

    message.attachments = [attachment]

    result = await get_full_text_from_message(message)
    expected_result = (
        "Hello, this is a test message.\n_[Could not read attachment: test.txt]_"
    )
    assert result == expected_result


def test_get_image_urls_from_message_no_images():
    # Mock a message with no image attachments or embeds
    message = MagicMock()
    message.attachments = []
    message.embeds = []

    result = get_image_urls_from_message(message)
    assert result == []


def test_get_image_urls_from_message_with_images():
    # Mock a message with image attachments and image embeds
    message = MagicMock()

    attachment_1 = MagicMock()
    attachment_1.content_type = "image/png"
    attachment_1.url = "https://example.com/image1.png"

    attachment_2 = MagicMock()
    attachment_2.content_type = "image/jpeg"
    attachment_2.url = "https://example.com/image2.jpg"

    embed_1 = MagicMock()
    embed_1.type = "image"
    embed_1.url = "https://example.com/embed_image.png"

    message.attachments = [attachment_1, attachment_2]
    message.embeds = [embed_1]

    result = get_image_urls_from_message(message)
    expected_result = [
        "https://example.com/image1.png",
        "https://example.com/image2.jpg",
        "https://example.com/embed_image.png",
    ]

    assert result == expected_result


@pytest.mark.asyncio
async def test_send_chunked_message():
    # Test the send_chunked_message functionality with chunking logic
    send_function = AsyncMock()

    response = (
        "This is a long message that will be chunked. " * 10
    )  # Simulate a long message
    await send_chunked_message(send_function, response, max_length=50)

    # Ensure that the mock send_function is called with the expected number of chunks
    assert send_function.call_count > 1
    for call in send_function.call_args_list:
        assert len(call[0][0]) <= 50  # Ensure that all chunks are within max_length


def test_chunk_message_md_friendly_basic():
    response = "This is a simple test message"
    result = chunk_message_md_friendly(response, max_length=30)  # Increased max_length

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == response  # The message should not be chunked


def test_chunk_message_md_friendly_with_long_message():
    response = "This is a long message that will be chunked."
    result = chunk_message_md_friendly(response, max_length=20)

    assert isinstance(result, list)
    assert len(result) > 1
    assert all(
        len(chunk) <= 20 for chunk in result
    )  # Ensure that all chunks are within max_length


def test_chunk_message_md_friendly_with_code_block():
    response = "Here is a code block:\n```\nprint('Hello, World!')\n```"
    result = chunk_message_md_friendly(response, max_length=50)

    assert isinstance(result, list)
    # Ensure the code block is not split across chunks
    code_block = "```\nprint('Hello, World!')\n```"
    assert any(
        chunk == code_block for chunk in result
    ), "Code block should be in one chunk"
    # Ensure code block is preserved
    assert "```" in "".join(result)


@pytest.mark.asyncio
async def test_send_chunked_message_with_error():
    # Mock a send_function that raises an exception on the second call
    send_function = AsyncMock(side_effect=[None, Exception("Send error")])

    response = "This is a long message that will be chunked. " * 10

    with pytest.raises(Exception, match="Send error"):
        await send_chunked_message(send_function, response, max_length=50)

    # Ensure that the mock send_function is called at least once before the exception
    assert send_function.call_count == 2
