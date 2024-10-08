import pytest
import yaml
from jipp.flow.flow_utils import (
    load_flow_from_yaml,
    Graph,
    adapt_node,
    yaml_to_pydantic_model,
)
from jipp.flow.node import Node, NodeDefinition
from pydantic import BaseModel, Field
from typing import Any


# Helper function to create a temporary YAML file
def create_temp_yaml(tmp_path, content):
    yaml_file = tmp_path / "test_flow.yaml"
    yaml_file.write_text(yaml.dump(content))
    return str(yaml_file)


def test_load_dag_flow(tmp_path):
    dag_content = {
        "$schema": "https://example.com/dag-schema.json",
        "inputs": {"input1": "string"},
        "outputs": {"output1": "string"},
        "nodes": [
            {
                "name": "node1",
                "type": "Python",
                "function": "test_function",
                "inputs": {"input": "${input1}"},
                "outputs": {"output": "string"},
            }
        ],
        "flow": [{"name": "node1", "type": "Python", "dependencies": []}],
    }
    yaml_file = create_temp_yaml(tmp_path, dag_content)

    graph = load_flow_from_yaml(yaml_file)

    assert isinstance(graph, Graph)
    assert graph.flow_type == "DAG"
    assert graph.inputs == {"input1": "string"}
    assert graph.outputs == {"output1": "string"}
    assert len(graph.nodes) == 1
    assert isinstance(graph.nodes["node1"], Node)
    assert graph.nodes["node1"].definition.name == "node1"
    assert graph.nodes["node1"].definition.type == "Python"
    assert graph.flow_steps == [{"name": "node1", "type": "Python", "dependencies": []}]


def test_load_flex_flow(tmp_path):
    flex_content = {
        "$schema": "https://example.com/flex-schema.json",
        "inputs": {"input1": "string"},
        "outputs": {"output1": "string"},
        "nodes": [
            {
                "name": "node1",
                "type": "Python",
                "function": "test_function",
                "inputs": {"input": "${input1}"},
                "outputs": {"output": "string"},
            }
        ],
        "flow": [
            {"when": "${input1} == 'test'", "then": ["node1"]},
            {"output": {"output1": "${node1.output}"}},
        ],
    }
    yaml_file = create_temp_yaml(tmp_path, flex_content)

    graph = load_flow_from_yaml(yaml_file)

    assert isinstance(graph, Graph)
    assert graph.flow_type == "Flex"
    assert graph.inputs == {"input1": "string"}
    assert graph.outputs == {"output1": "string"}
    assert len(graph.nodes) == 1
    assert isinstance(graph.nodes["node1"], Node)
    assert graph.nodes["node1"].definition.name == "node1"
    assert graph.nodes["node1"].definition.type == "Python"
    assert graph.flow_steps == [
        {"when": "${input1} == 'test'", "then": ["node1"]},
        {"output": {"output1": "${node1.output}"}},
    ]


def test_load_flow_with_invalid_schema(tmp_path):
    invalid_content = {
        "$schema": "https://example.com/invalid-schema.json",
        "nodes": [],
        "flow": [],
    }
    yaml_file = create_temp_yaml(tmp_path, invalid_content)

    with pytest.raises(ValueError, match="Unsupported flow type"):
        load_flow_from_yaml(yaml_file)


def test_load_flow_with_missing_file():
    with pytest.raises(FileNotFoundError):
        load_flow_from_yaml("non_existent_file.yaml")


def test_load_flow_with_invalid_yaml(tmp_path):
    invalid_yaml = "invalid: yaml: content:"
    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_text(invalid_yaml)

    with pytest.raises(yaml.YAMLError):
        load_flow_from_yaml(str(yaml_file))


def test_load_flow_with_unsupported_node_type(tmp_path):
    content = {
        "$schema": "https://example.com/dag-schema.json",
        "nodes": [
            {
                "name": "node1",
                "type": "UnsupportedType",
                "function": "test_function",
                "inputs": {},
                "outputs": {},
            }
        ],
        "flow": [{"name": "node1", "type": "UnsupportedType", "dependencies": []}],
    }
    yaml_file = create_temp_yaml(tmp_path, content)

    with pytest.raises(ValueError, match="Unsupported node type: unsupportedtype"):
        load_flow_from_yaml(yaml_file)


def test_adapt_python_node():
    python_node = {
        "name": "python_node",
        "type": "Python",
        "function": "module.function",
        "inputs": {"input1": "value1"},
        "outputs": {"output1": "type1"},
    }
    adapted = adapt_node(python_node)
    assert adapted["name"] == "python_node"
    assert adapted["type"] == "Python"
    assert adapted["function"] == "module.function"
    assert adapted["inputs"] == {"input1": "value1"}
    assert adapted["outputs"] == {"output1": "type1"}
    assert adapted["max_retries"] == 0
    assert "on_failure" not in adapted


def test_adapt_node_with_custom_fields():
    custom_node = {
        "name": "custom_node",
        "type": "Python",
        "function": "module.function",
        "inputs": {"input1": "value1"},
        "outputs": {"output1": "type1"},
        "max_retries": 3,
        "on_failure": "error_handler",
        "custom_field": "custom_value",
    }
    adapted = adapt_node(custom_node)
    assert adapted["max_retries"] == 3
    assert adapted["on_failure"] == "error_handler"
    assert "custom_field" not in adapted
    assert set(adapted.keys()) == {
        "name",
        "type",
        "function",
        "inputs",
        "outputs",
        "max_retries",
        "on_failure",
    }


def test_graph_initialization():
    nodes = {
        "node1": Node(
            definition=NodeDefinition(
                name="node1", type="Python", function="func1", inputs={}, outputs={}
            ),
            callable=lambda: None,
        ),
        "node2": Node(
            definition=NodeDefinition(
                name="node2", type="Python", function="func2", inputs={}, outputs={}
            ),
            callable=lambda: None,
        ),
    }
    flow_steps = [
        {"name": "node1", "dependencies": []},
        {"name": "node2", "dependencies": ["node1"]},
    ]
    graph = Graph(
        nodes=nodes,
        flow_steps=flow_steps,
        flow_type="DAG",
        inputs={"input": "string"},
        outputs={"output": "string"},
    )

    assert graph.nodes == nodes
    assert graph.flow_steps == flow_steps
    assert graph.flow_type == "DAG"
    assert graph.inputs == {"input": "string"}
    assert graph.outputs == {"output": "string"}
    assert graph.dependencies == {"node1": [], "node2": ["node1"]}


def test_graph_dependency_parsing_dag():
    flow_steps = [
        {"name": "node1", "dependencies": []},
        {"name": "node2", "dependencies": ["node1"]},
        {"name": "node3", "dependencies": ["node1", "node2"]},
    ]
    graph = Graph(nodes={}, flow_steps=flow_steps, flow_type="DAG")
    assert graph.dependencies == {
        "node1": [],
        "node2": ["node1"],
        "node3": ["node1", "node2"],
    }


def test_graph_dependency_parsing_flex():
    flow_steps = [
        {"when": "condition1", "then": ["node1", "node2"]},
        {"when": "condition2", "then": ["node3"]},
        {"output": {"result": "${node3.output}"}},
    ]
    graph = Graph(nodes={}, flow_steps=flow_steps, flow_type="Flex")
    assert graph.dependencies == {
        "node1": [],
        "node2": [],
        "node3": [],
    }


def test_graph_invalid_flow_type():
    with pytest.raises(ValueError, match="Unsupported flow type: InvalidType"):
        Graph(nodes={}, flow_steps=[], flow_type="InvalidType")


def test_yaml_to_pydantic_model_basic_types():
    schema = [
        {"name": "string_field", "type": "string", "description": "A string field"},
        {"name": "int_field", "type": "int", "description": "An integer field"},
        {"name": "float_field", "type": "float", "description": "A float field"},
        {"name": "bool_field", "type": "bool", "description": "A boolean field"},
    ]

    DynamicModel = yaml_to_pydantic_model(schema)

    assert issubclass(DynamicModel, BaseModel)
    assert DynamicModel.__annotations__["string_field"] == str
    assert DynamicModel.__annotations__["int_field"] == int
    assert DynamicModel.__annotations__["float_field"] == float
    assert DynamicModel.__annotations__["bool_field"] == bool

    # Check field descriptions
    assert DynamicModel.__fields__["string_field"].description == "A string field"
    assert DynamicModel.__fields__["int_field"].description == "An integer field"
    assert DynamicModel.__fields__["float_field"].description == "A float field"
    assert DynamicModel.__fields__["bool_field"].description == "A boolean field"


def test_yaml_to_pydantic_model_complex_types():
    schema = [
        {"name": "list_field", "type": "list", "description": "A list field"},
        {"name": "dict_field", "type": "dict", "description": "A dictionary field"},
    ]

    DynamicModel = yaml_to_pydantic_model(schema)

    assert issubclass(DynamicModel, BaseModel)
    assert DynamicModel.__annotations__["list_field"] == list
    assert DynamicModel.__annotations__["dict_field"] == dict

    # Check field descriptions
    assert DynamicModel.__fields__["list_field"].description == "A list field"
    assert DynamicModel.__fields__["dict_field"].description == "A dictionary field"


def test_yaml_to_pydantic_model_unknown_type():
    schema = [
        {
            "name": "unknown_field",
            "type": "unknown",
            "description": "An unknown type field",
        },
    ]

    DynamicModel = yaml_to_pydantic_model(schema)

    assert issubclass(DynamicModel, BaseModel)
    assert DynamicModel.__annotations__["unknown_field"] == Any

    # Check field description
    assert (
        DynamicModel.__fields__["unknown_field"].description == "An unknown type field"
    )


def test_yaml_to_pydantic_model_missing_description():
    schema = [
        {"name": "no_description_field", "type": "string"},
    ]

    DynamicModel = yaml_to_pydantic_model(schema)

    assert issubclass(DynamicModel, BaseModel)
    assert DynamicModel.__annotations__["no_description_field"] == str

    # Check that the field name is used as the description
    assert (
        DynamicModel.__fields__["no_description_field"].description
        == "no_description_field"
    )


def test_yaml_to_pydantic_model_custom_model_name():
    schema = [
        {"name": "test_field", "type": "string", "description": "A test field"},
    ]

    CustomModel = yaml_to_pydantic_model(schema, model_name="CustomModel")

    assert issubclass(CustomModel, BaseModel)
    assert CustomModel.__name__ == "CustomModel"
    assert CustomModel.__annotations__["test_field"] == str
    assert CustomModel.__fields__["test_field"].description == "A test field"


def test_yaml_to_pydantic_model_validation():
    schema = [
        {"name": "int_field", "type": "int", "description": "An integer field"},
        {"name": "string_field", "type": "string", "description": "A string field"},
    ]

    DynamicModel = yaml_to_pydantic_model(schema)

    # Valid data
    valid_data = {"int_field": 42, "string_field": "test"}
    valid_instance = DynamicModel(**valid_data)
    assert valid_instance.int_field == 42
    assert valid_instance.string_field == "test"

    # Invalid data (wrong types)
    with pytest.raises(ValueError):
        DynamicModel(int_field="not an int", string_field=123)

    # Missing required field
    with pytest.raises(ValueError):
        DynamicModel(int_field=42)


# Add more tests as needed...
