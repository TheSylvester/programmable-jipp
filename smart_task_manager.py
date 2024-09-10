from functools import partial
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Callable, Optional
from nextcord.ext import commands
from pydantic import BaseModel, Field
from jipp import ask_llm
from jipp.models.jipp_models import Tool
from message_chunker import send_chunked_message
from task_manager import TaskManager, StopTask, ListTasks, CreateTask
from utils.logging_utils import setup_logger

DEFAULT_TOOL_MODEL = "gpt-4o-mini"

CREATE_TASK_TOOL_CALL_SYSTEM_PROMPT = """Interpret the user's freeform instruction (provided as {{instruction}}) and map it to the necessary fields for creating a scheduled task.
#### Key Fields to Map:
1. **task_name**: Extract the main action or subject from the user’s input. Keep it short and clear (e.g., "Drink water reminder", "Weekly sales report").
2. **interval**: Parse the recurrence from the input (e.g., "every 30 minutes", "daily") and convert it to an integer in minutes. Default to 60 minutes if not specified.
3. **prompt**: Use the full user instruction as the action for the LLM to execute at the task interval (e.g., "Send a reminder to drink water", "Generate the weekly sales report").
#### Handling Missing Information:
- If no interval is provided, assume a default of 60 minutes.
- If key details are missing, ask for clarification from the user.
### Example Mappings:
1. **Input**: "Remind me to drink water every 30 minutes."  
   - **task_name**: "Drink water reminder"
   - **interval**: 30 minutes
   - **prompt**: "Send a reminder to drink water."
2. **Input**: "Send the weekly sales report every Monday."  
   - **task_name**: "Weekly sales report"
   - **interval**: 10080 minutes (1 week)
   - **prompt**: "Generate the weekly sales report and send it to the team."
3. **Input**: "Call my doctor tomorrow morning."  
   - **task_name**: "Doctor reminder"
   - **interval**: 1440 minutes (1 day)
   - **prompt**: "Remind me to call my doctor tomorrow morning."
Ensure that user input is interpreted accurately and mapped to the correct fields. Apply reasonable defaults where necessary.
"""

log = setup_logger(level="DEBUG", log_file=f"{__name__}.log")


class SmartCreateTask(BaseModel):
    """
    Create a new scheduled task based on natural language instructions.
    The task will pass a prompt instruction to an LLM to execute at defined intervals
    """

    instruction: str = Field(..., description="The name of the task to be created")


async def create_task_with_task_manager(
    task_manager: TaskManager,
    task_name: str,
    interval: int,
    function: Callable,
    **kwargs,
):
    task_manager_response: str = task_manager.create_task(
        task_name=task_name,
        interval=interval,
        function=function,
        prompt="Catch me if you can - Test",
    )


async def execute_prompt_at_context(ctx, prompt):
    """
    Asks an LLM to respond to a prompt, and send that response to the context.
    To execute at Task Trigger time.
    """

    response = await ask_llm(model=DEFAULT_TOOL_MODEL, prompt=prompt, temperature=0.3)
    await ctx.send(response)


class SmartTaskManager(commands.Cog):
    """
    SmartTaskManager is directly interfaced with and will convert natural language instruction to scheduled tasks
    """

    def __init__(self, bot, model: Optional[str] = None):
        self.bot: commands.Bot = bot
        self.model = model or DEFAULT_TOOL_MODEL
        self.task_manager: TaskManager = (
            bot.get_cog("TaskManager") or self.load_task_manager()
        )

    def load_task_manager(self):
        try:
            self.bot.load_extension("task_manager")
            task_manager = self.bot.get_cog("TaskManager")
        except Exception as e:
            raise ("TaskManager Cog not found")
        if not task_manager:
            raise ValueError("TaskManager Cog not found")
        return task_manager

    @commands.command(name="task", brief="Creates a recurring task from instructions")
    async def create_task(self, ctx, *, instruction: str = ""):
        """Converts a natural language task scheduling instruction to a TaskManager API call to schedule the task"""

        if not instruction:
            await send_chunked_message(
                await ctx.send, "No instructions provided to create task"
            )

        # This is the function to run at task time - a wrapped ask_llm call
        async def execute_prompt(prompt):
            return await execute_prompt_at_context(ctx, prompt)

        # Turn creating a task that executes a prompt into a tool function
        async def create_task_fn(task_name: str, interval: int, prompt: str):
            return self.task_manager.create_task(
                task_name=task_name,
                interval=interval,
                function=execute_prompt,
                prompt=prompt,
            )

        create_task_tool = Tool(schema=CreateTask, function=create_task_fn)

        try:
            # ask_llm to use CreateTask, so this will generate llm_task_name, llm_interval, llm_prompt
            response = await ask_llm(
                model=self.model,
                prompt=instruction,
                system=CREATE_TASK_TOOL_CALL_SYSTEM_PROMPT,
                tools=[create_task_tool],
                tool_choice="auto",
            )
        except Exception as e:
            # TODO: tool call error or other error?
            logging.error(f"SmartTaskManager ask_llm error: {e}")
            await send_chunked_message(ctx.send, f"SmartTaskManager ask_llm error: {e}")
            return

        await send_chunked_message(ctx.send, response)

    def export_tools(self) -> list[Tool]:
        """Returns a list of Tools to be imported, including the ones from task manager"""

        task_manager_tools = self.task_manager.export_tools()

        tools = task_manager_tools + [
            Tool(schema=SmartCreateTask, function=self.create_task),
        ]

        return tools


def setup(bot):
    bot.add_cog(SmartTaskManager(bot))
