from typing import Literal

from pydantic import BaseModel, Field

from jipp.jipp_engine import ask_llm
from jippity_ai.programs.md_loader import load_prompts
from jippity_ai.programs.analyze_message.program import MessageAnalysis

system_prompt, user_prompt = load_prompts()


class ActionDecision(BaseModel):
    """Structured output for an Action Decision."""

    action_type: Literal["respond", "no_op", "delegate"] = Field(
        ..., description="The type of action to be taken"
    )
    response_content: str = Field(
        ..., description="The content of the response, if applicable"
    )
    reason: str = Field(..., description="The reason for choosing this action")


async def decide_action(
    message_analysis: MessageAnalysis, bot_name: str
) -> ActionDecision:
    """
    Decide on the next action based on the analysis of an incoming message.

    Args:
        message_analysis (MessageAnalysis): The analysis of the incoming message.
        bot_name (str): The name of the bot making the decision.

    Returns:
        ActionDecision: A structured decision on the next action to take.

    Raises:
        Any exceptions raised by the underlying `ask_llm` function or md_loader.
    """

    decision = await ask_llm(
        model="gpt-4o-mini",
        prompt=user_prompt,
        system=system_prompt,
        response_format=ActionDecision,
        message_analysis=message_analysis.model_dump(),
        bot_name=bot_name,
    )

    return decision.parsed if decision.parsed else decision
