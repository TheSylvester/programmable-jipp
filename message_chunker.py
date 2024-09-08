import re
from typing import Callable, Any


async def send_chunked_message(send_function: Callable[[str], Any], response: str):
    """
    Send a response while ensuring it doesn't exceed Discord's message length limit.
    The response is first split by newline characters, then by whitespace, and finally by character count if necessary.

    Args:
        send_function (Callable[[str], Any]): The function to call to send the response.
        response (str): The response to send.
    """
    max_length = 2000

    if len(response) <= max_length:
        await send_function(response)
        return

    lines = response.split("\n")
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            if len(line) > max_length:
                # Further split the line by whitespace if it's too long
                words = re.split(r"(\s+)", line)
                for word in words:
                    if len(current_chunk) + len(word) > max_length:
                        await send_function(current_chunk)
                        current_chunk = word
                    else:
                        current_chunk += word

                # If still too long, split by character count
                while len(current_chunk) > max_length:
                    await send_function(current_chunk[:max_length])
                    current_chunk = current_chunk[max_length:]
            else:
                await send_function(current_chunk)
                current_chunk = line
        else:
            current_chunk += "\n" + line if current_chunk else line

    # Send any remaining text
    while len(current_chunk) > max_length:
        await send_function(current_chunk[:max_length])
        current_chunk = current_chunk[max_length:]

    if current_chunk:
        await send_function(current_chunk)
