import pytest
from nextcord.ext import commands
from jipp.models.jipp_models import Tool
from smart_task_manager import SmartTaskManager
from task_manager import TaskManager, CreateTask, StopTask, ListTasks
from tool_manager import ToolManager


class MockBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!")


@pytest.fixture
def bot():
    return MockBot()


@pytest.fixture
def tool_manager(bot):
    return ToolManager(bot)


def test_register_tool(tool_manager):
    # Create a mock tool
    mock_tool = Tool(schema=CreateTask, function=lambda: None)
    tool_manager.register_tool(mock_tool)

    assert "CreateTask" in tool_manager.tools
    assert tool_manager.tools["CreateTask"] == mock_tool


def test_get_tool(tool_manager):
    # Register a mock tool
    mock_tool = Tool(schema=CreateTask, function=lambda: None)
    tool_manager.register_tool(mock_tool)

    retrieved_tool = tool_manager.get_tool("CreateTask")
    assert retrieved_tool == mock_tool

    non_existent_tool = tool_manager.get_tool("NonExistentTool")
    assert non_existent_tool is None


def test_get_tools(tool_manager):
    # Register mock tools
    mock_tools = [
        Tool(schema=CreateTask, function=lambda: None),
        Tool(schema=StopTask, function=lambda: None),
        Tool(schema=ListTasks, function=lambda: None),
    ]
    tool_manager.register_tools(mock_tools)

    retrieved_tools = tool_manager.get_tools(
        ["CreateTask", "StopTask", "NonExistentTool"]
    )
    assert len(retrieved_tools) == 2
    assert all(isinstance(tool, Tool) for tool in retrieved_tools)
    assert set(tool.schema.__name__ for tool in retrieved_tools) == set(
        ["CreateTask", "StopTask"]
    )
