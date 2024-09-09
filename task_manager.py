import asyncio
from typing import Any, Callable, Coroutine
from nextcord.ext import tasks, commands
from pydantic import BaseModel, Field

from utils.logging_utils import setup_logger


log = setup_logger()

# Schemas for Function Calling


class StopTask(BaseModel):
    """
    Stops a specific scheduled task by providing its name.
    """

    task_name: str = Field(..., description="The name of the task to be stopped")


class ListTasks(BaseModel):
    """
    Lists all scheduled tasks that are currently active.
    """

    # This model is empty as list_tasks doesn't take any parameters
    pass


class CreateTask(BaseModel):
    """
    Create a new scheduled task with a specified name, interval, and prompt to execute.
    The task will pass a prompt instruction to an LLM to execute at defined intervals
    """

    task_name: str = Field(..., description="The name of the task to be created")
    interval: int = Field(
        ..., description="The interval in minutes at which the task should run"
    )
    prompt: str = Field(
        ...,
        description="The prompt message for LLM to follow when this task is triggered",
    )


class TaskManager(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.jobs = {}  # Store dynamically created jobs

    async def create_task(
        self, task_name: str, interval: int, function: Callable[..., Any], **kwargs: Any
    ) -> str:
        """Creates a task with the given interval and function."""
        log.debug(
            f"Creating task: task_name={task_name}, interval={interval}, function={function}, kwargs={kwargs}"
        )
        if task_name in self.jobs:
            return f"Task '{task_name}' already exists."

        # Define the task as a loop that runs at the specified interval
        @tasks.loop(minutes=interval)
        async def dynamic_task():
            # Check if the function is a coroutine (async) or not
            if asyncio.iscoroutinefunction(function):
                # If it's async, use await
                await function(**kwargs)
            else:
                # If it's not async, run it in an executor
                await self.bot.loop.run_in_executor(None, function, **kwargs)

        # Start the task
        dynamic_task.start()

        # Store the task in the jobs dictionary
        self.jobs[task_name] = dynamic_task
        return f"Task '{task_name}' created and scheduled to start with an interval of {interval} minutes."

    def create_task_sync(
        self, task_name: str, interval: int, function: Callable[..., Any], **kwargs: Any
    ) -> str:
        """Synchronous wrapper for create_task."""
        return asyncio.run_coroutine_threadsafe(
            self.create_task(task_name, interval, function, **kwargs), self.bot.loop
        ).result()

    def stop_task(self, task_name: str):
        """Stops a task by name."""
        print(f"Stopping task: task_name={task_name}")
        if task_name in self.jobs:
            self.jobs[task_name].cancel()  # Stop the task loop
            del self.jobs[task_name]
            return f"Task '{task_name}' stopped and removed."
        else:
            return f"Task '{task_name}' does not exist."

    def list_tasks(self):
        """Returns a list of currently running tasks."""
        print("Listing tasks")
        return [name for name in self.jobs]


def setup(bot):
    bot.add_cog(TaskManager(bot))
