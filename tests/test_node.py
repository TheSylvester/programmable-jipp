import pytest
from jipp.flow.node import Node, NodeDefinition
from unittest.mock import Mock


def test_node_creation():
    definition = NodeDefinition(
        name="test_node",
        type="Python",
        function="test_function",
        inputs={"input1": "string"},
        outputs={"output1": "string"},
    )
    callable_func = Mock()
    node = Node(definition=definition, callable=callable_func)

    assert node.definition == definition
    assert node.callable == callable_func


def test_node_definition_creation():
    definition = NodeDefinition(
        name="test_node",
        type="Python",
        function="test_function",
        inputs={"input1": "string"},
        outputs={"output1": "string"},
    )

    assert definition.name == "test_node"
    assert definition.type == "Python"
    assert definition.function == "test_function"
    assert definition.inputs == {"input1": "string"}
    assert definition.outputs == {"output1": "string"}
    assert definition.max_retries == 0
    assert definition.on_failure is None
    assert definition.on_success is None


def test_node_definition_with_optional_fields():
    def on_failure_func():
        pass

    def on_success_func():
        pass

    definition = NodeDefinition(
        name="test_node",
        type="Python",
        function="test_function",
        inputs={"input1": "string"},
        outputs={"output1": "string"},
        max_retries=3,
        on_failure=on_failure_func,
        on_success=on_success_func,
    )

    assert definition.max_retries == 3
    assert definition.on_failure == on_failure_func
    assert definition.on_success == on_success_func


def test_node_definition_validation():
    with pytest.raises(ValueError, match="Node name cannot be empty"):
        NodeDefinition(
            name="",  # Empty name should raise an error
            type="Python",
            function="test_function",
            inputs={},
            outputs={},
        )

    with pytest.raises(ValueError, match="Invalid node type: InvalidType"):
        NodeDefinition(
            name="test_node",
            type="InvalidType",  # Invalid type should raise an error
            function="test_function",
            inputs={},
            outputs={},
        )


def test_node_callable_execution():
    definition = NodeDefinition(
        name="test_node",
        type="Python",
        function="test_function",
        inputs={"input1": "string"},
        outputs={"output1": "string"},
    )
    callable_func = Mock(return_value={"output1": "result"})
    node = Node(definition=definition, callable=callable_func)

    result = node.callable(input1="test_input")
    assert result == {"output1": "result"}
    callable_func.assert_called_once_with(input1="test_input")


@pytest.mark.asyncio
async def test_async_node_callable_execution():
    definition = NodeDefinition(
        name="test_node",
        type="Python",
        function="test_function",
        inputs={"input1": "string"},
        outputs={"output1": "string"},
    )

    async def async_callable(input1):
        return {"output1": f"async_{input1}"}

    node = Node(definition=definition, callable=async_callable)

    result = await node.callable(input1="test_input")
    assert result == {"output1": "async_test_input"}


def test_node_str_representation():
    definition = NodeDefinition(
        name="test_node",
        type="Python",
        function="test_function",
        inputs={"input1": "string"},
        outputs={"output1": "string"},
    )
    node = Node(definition=definition, callable=Mock())

    assert str(node) == "Node(name=test_node, type=Python)"


def test_node_repr_representation():
    definition = NodeDefinition(
        name="test_node",
        type="Python",
        function="test_function",
        inputs={"input1": "string"},
        outputs={"output1": "string"},
    )
    node = Node(definition=definition, callable=Mock())

    assert repr(node) == "Node(name=test_node, type=Python, function=test_function)"
