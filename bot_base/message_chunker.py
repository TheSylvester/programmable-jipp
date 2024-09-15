import re
from typing import List
from jipp.utils.logging_utils import setup_logger

log = setup_logger()


async def get_full_text_from_message(message) -> str:
    """
    Extracts the full text content from a message, including any .txt file attachments.

    Args:
        message: The Nextcord message object.

    Returns:
        str: A string containing the full message content with text from .txt attachments.
    """
    full_content = message.content

    if not hasattr(message, "attachments"):
        log.error(f"Message object {message.id} does not have attachments attribute")
        return full_content

    for attachment in message.attachments:
        if attachment.filename.endswith(".txt"):
            try:
                file_content = await attachment.read()
                text_content = file_content.decode("utf-8")
                full_content += f"\n{text_content}"
            except UnicodeDecodeError as e:
                log.error(
                    f"Failed to decode attachment {attachment.filename} in message {message.id}: {e}"
                )
                full_content += (
                    f"\n_[Could not read attachment: {attachment.filename}]_"
                )

    return full_content


def get_image_urls_from_message(message) -> List[str]:
    """
    Extracts a list of image URLs or attachments from a message.

    Args:
        message: The Nextcord message object.

    Returns:
        List[str]: A list of URLs pointing to images in the message.
    """
    image_urls = []

    if not hasattr(message, "attachments"):
        log.error(f"Message object {message.id} does not have attachments attribute")
        return image_urls

    image_urls.extend(
        attachment.url
        for attachment in message.attachments
        if attachment.content_type and attachment.content_type.startswith("image/")
    )

    if not hasattr(message, "embeds"):
        log.error(f"Message object {message.id} does not have embeds attribute")
        return image_urls

    image_urls.extend(
        embed.url for embed in message.embeds if embed.type == "image" and embed.url
    )

    return image_urls


import re
from typing import List, Deque
from collections import deque

import re
from typing import List


def chunk_message_md_friendly(response: str, max_length: int = 2000) -> List[str]:
    """
    Chunk a message in a markdown-friendly way, ensuring no code blocks or markdown formatting is broken.

    Args:
        response (str): The response to chunk.
        max_length (int): The maximum allowed size per chunk. Must be positive.

    Returns:
        List[str]: A list of chunks.

    Raises:
        ValueError: If max_length is not a positive integer.
    """
    if max_length <= 0:
        raise ValueError("max_length must be a positive integer")

    if not response:
        return ["_..._"]

    if len(response) <= max_length:
        return [response]

    chunks: List[str] = []
    current_chunk = ""
    lines = response.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip() == "```":
            code_block_lines = [line]
            i += 1
            while i < len(lines):
                code_line = lines[i]
                code_block_lines.append(code_line)
                if code_line.strip() == "```":
                    i += 1
                    break
                i += 1
            code_block = "\n".join(code_block_lines)

            # Try to add the entire code block to the current chunk
            if len(current_chunk) + len(code_block) + 1 <= max_length:
                current_chunk += ("\n" if current_chunk else "") + code_block
            else:
                # Close the current chunk and start a new one with the code block
                if current_chunk:
                    chunks.append(current_chunk)
                if len(code_block) <= max_length:
                    current_chunk = code_block
                else:
                    # Code block itself exceeds max_length, split it carefully
                    code_lines = code_block_lines
                    code_chunk = ""
                    for code_line in code_lines:
                        if len(code_chunk) + len(code_line) + 1 <= max_length:
                            code_chunk += ("\n" if code_chunk else "") + code_line
                        else:
                            chunks.append(code_chunk)
                            code_chunk = code_line
                    current_chunk = code_chunk
        else:
            if len(current_chunk) + len(line) + 1 <= max_length:
                current_chunk += ("\n" if current_chunk else "") + line
                i += 1
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(line) <= max_length:
                    current_chunk = line
                    i += 1
                else:
                    # Split long line
                    words = re.split(r"(\s+)", line)
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) <= max_length:
                            current_line += word
                        else:
                            chunks.append(current_line)
                            current_line = word
                    current_chunk = current_line
                    i += 1

    if current_chunk:
        chunks.append(current_chunk)

    return [chunk for chunk in chunks if chunk.strip()]  # Remove any empty chunks


async def send_chunked_message(
    send_function, response: str, max_length: int = 2000
) -> None:
    """
    Orchestrate sending a chunked message using the send_function.

    Args:
        send_function: The function to call to send the response.
        response (str): The response to send.
        max_length (int): The maximum chunk size.

    Raises:
        Exception: If sending a chunk fails.
    """
    chunks = chunk_message_md_friendly(response, max_length)

    for chunk in chunks:
        try:
            await send_function(chunk)
        except Exception as e:
            log.error(f"Failed to send chunk: {e}")
            raise  # Re-raise the exception after logging
