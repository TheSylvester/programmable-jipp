from typing import Any, Callable
from nextcord.ext import tasks, commands
from pydantic import BaseModel, Field


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

    def create_task(self, task_name: str, interval: int, function, *args):
        """Creates a task with the given interval and function."""
        print(
            f"Creating task: task_name={task_name}, interval={interval}, function={function}, args={args}"
        )
        if task_name in self.jobs:
            return f"Task '{task_name}' already exists."

        # Define the task as a loop that runs at the specified interval
        @tasks.loop(minutes=interval)
        async def dynamic_task():
            await function(*args)  # Call the task function with the provided arguments

        dynamic_task.start()  # Start the task loop

        # Store the task in the jobs dictionary
        self.jobs[task_name] = dynamic_task
        return f"Task '{task_name}' created and started with an interval of {interval} minutes."

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