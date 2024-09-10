from typing import List
from nextcord.ext import commands
from jipp.models.jipp_models import Tool
from smart_task_manager import SmartCreateTask, SmartTaskManager
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

        self.smart_task_manager: SmartTaskManager = (
            bot.get_cog("SmartTaskManager") or self.load_smart_task_manager()
        )
        smart_task_manager_tools = self.smart_task_manager.export_tools()

        self.register_tools(smart_task_manager_tools)

    def get_tool(self, tool_name: str):
        """Retrieve a tool by name."""
        return self.tools.get(tool_name, None)

    def get_tools(self, tool_names: List[str]) -> List[Tool]:
        """Retrieve multiple tools by name."""
        return [self.tools.get(name) for name in tool_names if name in self.tools]

    def load_smart_task_manager(self):
        try:
            self.bot.load_extension("smart_task_manager")
            smart_task_manager = self.bot.get_cog("SmartTaskManager")
        except Exception as e:
            raise ("SmartTaskManager Cog not found")
        if not smart_task_manager:
            raise ValueError("TaskManager Cog not found")
        return smart_task_manager


def setup(bot):
    bot.add_cog(ToolManager(bot))
