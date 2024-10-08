# utils.py

import yaml
from typing import Dict, Any, List, Type
from .node import Node, NodeDefinition


class Graph:
    """
    Represents the flow graph containing nodes and flow steps.
    """

    def __init__(
        self,
        nodes: Dict[str, Node],
        flow_steps: List[Any],
        flow_type: str = "DAG",
        inputs: Dict[str, Any] = None,
        outputs: Dict[str, Any] = None,
    ):
        """
        Initializes the graph.

        Args:
            nodes (Dict[str, Node]): The nodes in the graph.
            flow_steps (List[Any]): The flow steps.
            flow_type (str): The type of flow ('DAG' or 'Flex').
            inputs (Dict[str, Any], optional): Flow-level inputs.
            outputs (Dict[str, Any], optional): Flow-level outputs.
        """
        self.nodes = nodes
        self.flow_steps = flow_steps
        self.flow_type = flow_type
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.dependencies = self._parse_dependencies(flow_steps)

    def _parse_dependencies(self, flow_steps: List[Any]) -> Dict[str, List[str]]:
        """
        Parses flow steps to extract dependencies between nodes.

        Args:
            flow_steps (List[Any]): The flow steps from the YAML configuration.

        Returns:
            Dict[str, List[str]]: A mapping of node names to their dependencies.
        """
        dependencies = {}
        if self.flow_type == "DAG":
            for step in flow_steps:
                if isinstance(step, dict):
                    node_name = step["name"]
                    dependencies[node_name] = step.get("dependencies", [])
                elif isinstance(step, list):
                    for node in step:
                        node_name = node["name"]
                        dependencies[node_name] = node.get("dependencies", [])
                else:
                    raise ValueError("Invalid step format in DAG flow")
        elif self.flow_type == "Flex":
            for step in flow_steps:
                if isinstance(step, dict):
                    if "when" in step and "then" in step:
                        for node_name in step["then"]:
                            dependencies[node_name] = (
                                []
                            )  # No explicit dependencies in Flex flow
                    elif "output" in step:
                        # Output steps don't have dependencies
                        pass
                    else:
                        raise ValueError("Invalid step format in Flex flow")
                else:
                    raise ValueError("Invalid step format in Flex flow")
        else:
            raise ValueError(f"Unsupported flow type: {self.flow_type}")
        return dependencies


def adapt_node(node_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapts any node type to a uniform structure.
    """
    adapted_node = {
        "name": node_dict["name"],
        "type": node_dict["type"].capitalize(),  # Normalize the type
        "inputs": node_dict.get("inputs", {}),
        "outputs": node_dict.get("outputs", {}),
        "max_retries": node_dict.get("max_retries", 0),
    }

    node_type = node_dict["type"].lower()  # Use lowercase for comparison

    if node_type == "llm":
        adapted_node["function"] = "jipp.adapters.llm_adapter.execute_llm"
        adapted_node["model"] = node_dict.get("model", "default_model")
        adapted_node["prompt_template"] = node_dict.get("prompt_template", "")
    elif node_type == "api":
        adapted_node["function"] = "jipp.flow.node_manager.NodeManager.api_call"
        if "inputs" not in adapted_node:
            adapted_node["inputs"] = {
                "url": node_dict.get("api", ""),
                "method": node_dict.get("method", "GET"),
            }
    elif node_type == "python":
        if "function" not in node_dict:
            raise ValueError(
                f"Python node '{node_dict['name']}' is missing 'function' field"
            )
        adapted_node["function"] = node_dict["function"]

    # Only include on_failure if it's explicitly set
    if "on_failure" in node_dict:
        adapted_node["on_failure"] = node_dict["on_failure"]

    return adapted_node


def load_flow_from_yaml(yaml_file: str) -> Graph:
    """
    Loads a flow graph from a YAML file.

    Args:
        yaml_file (str): The path to the YAML file.

    Returns:
        Graph: The loaded flow graph.
    """
    with open(yaml_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Determine flow type
    schema_url = config_data.get("$schema", "").lower()
    if "dag" in schema_url:
        flow_type = "DAG"
    elif "flex" in schema_url:
        flow_type = "Flex"
    else:
        raise ValueError(f"Unsupported flow type: {schema_url}")

    # Parse flow-level inputs and outputs
    flow_inputs = config_data.get("inputs", {})
    flow_outputs = config_data.get("outputs", {})

    # Parse nodes
    nodes = {}
    for node_dict in config_data.get("nodes", []):
        node_type = node_dict.get("type", "").lower()
        if node_type not in ["python", "llm", "api"]:
            raise ValueError(f"Unsupported node type: {node_type}")

        adapted_node_dict = adapt_node(node_dict)
        node_definition = NodeDefinition(**adapted_node_dict)
        node = Node(
            definition=node_definition, callable=lambda: None
        )  # placeholder callable
        nodes[node_definition.name] = node

    # Parse flow steps
    flow_steps = config_data.get("flow", [])

    # Create Graph instance
    graph = Graph(
        nodes=nodes,
        flow_steps=flow_steps,
        flow_type=flow_type,
        inputs=flow_inputs,
        outputs=flow_outputs,
    )

    return graph


from pydantic import BaseModel, create_model, Field
from typing import Dict, Any


def yaml_to_pydantic_model(
    schema: List[Dict[str, Any]], model_name: str = "DynamicModel"
) -> Type[BaseModel]:
    """
    Converts a YAML schema dictionary of inputs/outputs into a dynamic Pydantic BaseModel.

    :param schema: The schema dictionary loaded from YAML defining inputs/outputs.
    :param model_name: The name of the dynamic Pydantic model to be created.
    :return: A Pydantic BaseModel derived from the provided schema.
    """

    fields = {}
    for field in schema:
        field_name = field["name"]
        field_type = field["type"]
        description = field.get(
            "description", field_name
        )  # Use name if description is not available

        # Map string types to Python types
        type_mapping = {
            "string": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        python_type = type_mapping.get(
            field_type, Any
        )  # Default to Any if type is not recognized

        # Create a field with description
        fields[field_name] = (python_type, Field(description=description))

    # Dynamically create a Pydantic model
    return create_model(model_name, **fields)
