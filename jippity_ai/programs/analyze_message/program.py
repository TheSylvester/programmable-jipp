from pydantic import BaseModel, Field

from jipp.jipp_engine import ask_llm
from jippity_ai.programs.md_loader import load_prompts


system_prompt, user_prompt = load_prompts()


class MessageAnalysis(BaseModel):
    """Structured output for message analysis."""

    speaker: str = Field(
        ..., description="The speaker's identity and role in the conversation"
    )
    intent: str = Field(..., description="The intent of the message")
    audience: str = Field(..., description="The intended audience of the message")
    expected_response: str = Field(
        ..., description="The speaker's expectation of a response, if any"
    )
    next_speaker: str = Field(
        ...,
        description="Who might respond next in this conversation based on the incoming message",
    )
    relevant_context: str = Field(
        ..., description="Message from the history that are relevant to this message"
    )


async def analyze_incoming_message(
    message: str, channel_history: str
) -> MessageAnalysis:
    """
    Analyze an incoming message with LLM in the context of the channel history.

    Args:
        message (str): The incoming message to be analyzed.
        channel_history (str): The relevant history of the channel for context.

    Returns:
        MessageAnalysis: A structured analysis of the message, including details about
        the speaker, intent, audience, expected response, and conversation flow.

    Raises:
        Any exceptions raised by the underlying `ask_llm` function or md_loader.
    """

    analysis = await ask_llm(
        model="gpt-4o-mini",
        prompt=user_prompt,
        system=system_prompt,
        response_format=MessageAnalysis,
        history_string=channel_history,
        message_string=message,
    )

    return analysis.parsed if analysis.parsed else analysis
