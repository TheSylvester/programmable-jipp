# utils.py

import yaml
from typing import Dict, Any, List
from .node import BaseNode, PythonNode, LLMNode, APINode


class Graph:
    """
    Represents the flow graph containing nodes and flow steps.
    """

    def __init__(
        self,
        nodes: Dict[str, BaseNode],
        flow_steps: List[Any],
        flow_type: str = "DAG",
        inputs: Dict[str, Any] = None,
        outputs: Dict[str, Any] = None,
    ):
        """
        Initializes the graph.

        Args:
            nodes (Dict[str, BaseNode]): The nodes in the graph.
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


def adapt_node(node_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapts any node type to a uniform structure.
    """
    adapted_node = node_dict.copy()
    node_type = node_dict["type"]

    if node_type == "LLM":
        adapted_node["function"] = "jipp.jipp_engine.ask_llm"
        adapted_node["inputs"] = {
            "prompt": node_dict["prompt_template"],
            "connection": node_dict["connection"],
            # Add any other necessary inputs for ask_llm
        }
    elif node_type == "API":
        adapted_node["function"] = "jipp.flow.node_manager.NodeManager.api_call"
        adapted_node["inputs"] = {
            "url": node_dict["api"],
            "method": node_dict.get("method", "GET"),
            # Add any other necessary inputs for API calls
        }
    elif node_type == "Python":
        # Python nodes should already have a 'function' field, but we can ensure it's present
        if "function" not in adapted_node:
            raise ValueError(
                f"Python node '{node_dict['name']}' is missing 'function' field"
            )
    else:
        raise ValueError(f"Unsupported node type: {node_type}")

    # Ensure common fields are present
    adapted_node["max_retries"] = node_dict.get("max_retries", 0)
    adapted_node["on_failure"] = node_dict.get("on_failure")

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
    flow_type = "DAG" if "dag" in schema_url else "Flex"

    # Parse flow-level inputs and outputs
    flow_inputs = config_data.get("inputs", {})
    flow_outputs = config_data.get("outputs", {})

    # Parse nodes
    nodes = {}
    for node_dict in config_data.get("nodes", []):
        adapted_node_dict = adapt_node(node_dict)
        node_type = adapted_node_dict["type"]
        if node_type == "Python":
            node = PythonNode(**adapted_node_dict)
        elif node_type == "LLM":
            node = LLMNode(**adapted_node_dict)
        elif node_type == "API":
            node = APINode(**adapted_node_dict)
        else:
            raise ValueError(f"Unsupported node type: {node_type}")
        nodes[node.name] = node

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
