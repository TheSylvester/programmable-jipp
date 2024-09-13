import asyncio
import json
from typing import Any, Callable, Coroutine
from nextcord.ext import tasks, commands
from pydantic import BaseModel, Field

from jipp.models.jipp_models import Tool
from message_chunker import send_chunked_message
from jipp.utils.logging_utils import setup_logger


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
        self.jobs_metadata = {}  # Store jobs creation args

    def create_task(
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
        # Store the args metadata
        self.jobs_metadata[task_name] = {
            "task_name": task_name,
            "interval": interval,
            "function_name": function.__name__,
            **kwargs,
        }

        return f"Task '{task_name}' created and scheduled to start with an interval of {interval} minutes."

    @commands.command(
        name="stop_task", aliases=["stop", "stop_tasks"], brief="Stops a recurring task"
    )
    async def stop_task(self, ctx, *, task_name: str):
        if not task_name:
            await send_chunked_message(ctx.send, f"No task name provided to stop.")
            return
        try:
            result = await self._stop_task(task_name=task_name)
            await send_chunked_message(ctx.send, result)
            return
        except Exception as e:
            log.error(f"SmartTaskManager stop_task error: {e}")
            await send_chunked_message(
                ctx.send, f"Failed to stop task '{task_name}': {str(e)}"
            )

    @commands.command(name="list_tasks", brief="Lists all active recurring tasks")
    async def list_tasks(self, ctx):
        try:
            tasks = self._list_tasks()
            if not tasks:
                await ctx.send("No active tasks.")
                return

            output_strings = []
            for task in tasks:
                next_iteration = (
                    task["job"].next_iteration.strftime("%Y-%m-%d %H:%M:%S")
                    if task["job"].next_iteration
                    else "N/A"
                )

                # Prepare args string without 'job' and 'task_name'
                args_str = "\n".join(
                    f"{k}: {v}"
                    for k, v in task.items()
                    if k not in ["job", "task_name", "interval"]
                )

                output_strings.append(
                    f"- Task: {task['task_name']}\n"
                    f"  Interval: {task['interval']} minutes\n"
                    f"  Next run: {next_iteration}\n"
                    f"  {args_str}"
                )

            await send_chunked_message(
                ctx.send, "Active tasks:\n" + "\n\n".join(output_strings)
            )
        except Exception as e:
            log.error(f"TaskManager list_tasks error: {e}")
            await send_chunked_message(ctx.send, f"Failed to list tasks: {str(e)}")

    async def _stop_task(self, task_name: str):
        """Stops a task by name."""
        log.info(f"Stopping task: task_name={task_name}")

        if task_name in self.jobs:
            task = self.jobs[task_name]

            # Cancel the task (this stops the loop)
            task.cancel()

            try:
                # Handle the task cancellation gracefully
                await task._task  # This awaits the internal task if needed
            except asyncio.CancelledError:
                log.info(f"Task '{task_name}' was cancelled successfully.")
            except Exception as e:
                log.error(f"Error while stopping task '{task_name}': {e}")

            # Remove the task from the jobs list and metadata
            del self.jobs[task_name]
            del self.jobs_metadata[task_name]  # Add this line
            return f"Task '{task_name}' stopped and removed."
        else:
            return f"Task '{task_name}' does not exist."

    def _list_tasks(self):
        """Returns a list of currently running tasks."""
        listed_tasks = []

        for job_name, job in self.jobs.items():
            le_job = {**self.jobs_metadata[job_name], "job": job}
            listed_tasks.append(le_job)

        return listed_tasks

    def export_tools(self) -> list[Tool]:
        """Returns a list of Tools to be imported"""

        tools = [
            Tool(schema=StopTask, function=self.stop_task),
            Tool(schema=ListTasks, function=self.list_tasks),
        ]

        return tools


def setup(bot):
    bot.add_cog(TaskManager(bot))
