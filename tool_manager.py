from typing import List
from nextcord.ext import commands
from jipp.models.jipp_models import Tool
from task_manager import TaskManager, CreateTask, StopTask, ListTasks


class ToolManager(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.tools = {}

        self.register_available_tools(bot)

    def register_tool(self, tool: Tool):
        """Register a tool using the schema's name."""
        tool_name = tool.schema.__name__
        self.tools[tool_name] = tool

    def register_tools(self, tools: list[Tool]):
        """Register a list of tools each using the schema's name."""
        for tool in tools:
            self.register_tool(tool)

    def register_available_tools(self, bot=None):
        """Register all available tools."""

        bot = bot or self.bot

        # TODO: Dynamically add Tools here
        self.task_manager = TaskManager(bot)

        # Check if the TaskManager cog is already loaded
        if not self.bot.get_cog("TaskManager"):
            self.bot.add_cog(self.task_manager)

        tools: List[Tool] = [
            Tool(schema=CreateTask, function=self.task_manager.create_task),
            Tool(schema=StopTask, function=self.task_manager.stop_task),
            Tool(schema=ListTasks, function=self.task_manager.list_tasks),
            # Add more tools here as needed
        ]
        self.register_tools(tools)

    def get_tool(self, tool_name: str):
        """Retrieve a tool by name."""
        return self.tools.get(tool_name, None)

    def get_tools(self, tool_names: List[str]) -> List[Tool]:
        """Retrieve multiple tools by name."""
        return [self.tools.get(name) for name in tool_names if name in self.tools]
