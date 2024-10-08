import pytest
from jipp.flow.node_manager import NodeManager
from jipp.flow.node import Node, NodeDefinition
from unittest.mock import patch, MagicMock


@pytest.fixture
def node_manager():
    return NodeManager()


@pytest.mark.asyncio
async def test_load_node(event_loop):
    node_manager = NodeManager()
    node_def = {
        "name": "test_node",
        "type": "Python",
        "function": "test_module.test_function",
        "inputs": {"input1": "string"},
        "outputs": {"output1": "string"},
    }

    with patch.object(NodeManager, "_load_python_function", return_value=lambda: None):
        node_manager.load_node(node_def)

    assert "test_node" in node_manager.nodes
    assert isinstance(node_manager.nodes["test_node"], Node)


def test_load_nodes(node_manager):
    node_definitions = [
        {
            "name": "node1",
            "type": "Python",
            "function": "test_module.test_function",
            "inputs": {"input1": "string"},
            "outputs": {"output1": "string"},
        },
        {
            "name": "node2",
            "type": "LLM",
            "function": "llm_function",
            "inputs": {"prompt": "string"},
            "outputs": {"response": "string"},
        },
        {
            "name": "node3",
            "type": "API",
            "function": "api_function",
            "inputs": {"url": "string"},
            "outputs": {"response": "string"},
        },
    ]

    with patch.object(NodeManager, "_load_python_function", return_value=lambda: None):
        node_manager.load_nodes(node_definitions)

    assert len(node_manager.nodes) == 3
    assert "node1" in node_manager.nodes
    assert "node2" in node_manager.nodes
    assert "node3" in node_manager.nodes

    # Updated assertions to verify the adapted node structures
    assert (
        node_manager.nodes["node2"].definition.function
        == "jipp.flow.adapters.llm_adapter.execute_llm"
    )
    assert (
        node_manager.nodes["node3"].definition.function
        == "jipp.flow.adapters.api_adapter.execute_api"
    )


def test_load_node(node_manager):
    node_def = {
        "name": "test_node",
        "type": "Python",
        "function": "test_module.test_function",
        "inputs": {"input1": "string"},
        "outputs": {"output1": "string"},
    }

    with patch.object(NodeManager, "_load_python_function", return_value=lambda: None):
        node_manager.load_node(node_def)

    assert "test_node" in node_manager.nodes
    assert isinstance(node_manager.nodes["test_node"], Node)


def test_create_node(node_manager):
    node_def = NodeDefinition(
        name="test_node",
        type="Python",  # Changed from "python" to "Python"
        function="test_module.test_function",
        inputs={},
        outputs={},
    )

    with patch.object(NodeManager, "_load_python_function", return_value=lambda: None):
        node = node_manager._create_node(node_def)

    assert isinstance(node, Node)
    assert node.definition == node_def


@patch("importlib.import_module")
def test_load_python_function(mock_import_module, node_manager):
    mock_module = MagicMock()
    mock_function = MagicMock()
    mock_module.test_function = mock_function
    mock_import_module.return_value = mock_module

    function = node_manager._load_python_function("test_module.test_function")

    assert function == mock_function
    mock_import_module.assert_called_once_with("test_module")


def test_adapt_llm_node(node_manager):
    node_def = {"name": "llm_node", "type": "llm", "function": "original_function"}
    adapted_def = node_manager._adapt_llm_node(node_def)

    assert adapted_def["function"] == "jipp.flow.adapters.llm_adapter.execute_llm"
    assert adapted_def["name"] == "llm_node"
    assert adapted_def["type"] == "llm"


def test_adapt_api_node(node_manager):
    node_def = {"name": "api_node", "type": "api", "function": "original_function"}
    adapted_def = node_manager._adapt_api_node(node_def)

    assert adapted_def["function"] == "jipp.flow.adapters.api_adapter.execute_api"
    assert adapted_def["name"] == "api_node"
    assert adapted_def["type"] == "api"


@pytest.mark.asyncio
async def test_execute_node_sync(node_manager):
    mock_function = MagicMock(return_value="test_result")
    node = Node(
        definition=NodeDefinition(
            name="test_node",
            type="python",
            function="test_function",
            inputs={},
            outputs={},
        ),
        callable=mock_function,
    )

    result = await node_manager.execute_node(node, {"input": "test"})

    assert result == "test_result"
    mock_function.assert_called_once_with(input="test")


@pytest.mark.asyncio
async def test_execute_node_async(node_manager):
    async def mock_async_function(**kwargs):
        return "async_result"

    node = Node(
        definition=NodeDefinition(
            name="test_node",
            type="python",
            function="test_function",
            inputs={},
            outputs={},
        ),
        callable=mock_async_function,
    )

    result = await node_manager.execute_node(node, {"input": "test"})

    assert result == "async_result"


def test_get_node(node_manager):
    node_def = NodeDefinition(
        name="test_node", type="python", function="test_function", inputs={}, outputs={}
    )
    node = Node(definition=node_def, callable=lambda: None)
    node_manager.nodes["test_node"] = node

    retrieved_node = node_manager.get_node("test_node")
    assert retrieved_node == node

    non_existent_node = node_manager.get_node("non_existent")
    assert non_existent_node is None
