from typing import List, Optional
import nextcord
from nextcord.ext import commands
from nextcord.ext.commands.errors import ExtensionFailed
from jipp.models.jipp_models import Tool
from bot_base.smart_task_manager import SmartTaskManager


class ToolManager(commands.Cog):
    def __init__(self, bot, smart_task_manager: Optional[SmartTaskManager] = None):
        self.bot = bot
        self.tools = {}

        self.smart_task_manager = smart_task_manager or self.load_smart_task_manager()
        self.register_available_tools()

    def register_tool(self, tool: Tool):
        """Register a tool using the schema's name."""
        tool_name = tool.schema.__name__
        self.tools[tool_name] = tool

    def register_tools(self, tools: list[Tool]):
        """Register a list of tools each using the schema's name."""
        for tool in tools:
            self.register_tool(tool)

    def register_available_tools(self):
        """Register all available tools."""
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
            self.bot.load_extension("bot_base.smart_task_manager")
            return self.bot.get_cog("SmartTaskManager")
        except ExtensionFailed:
            raise RuntimeError("SmartTaskManager Cog not found")


def setup(bot):
    bot.add_cog(ToolManager(bot))
