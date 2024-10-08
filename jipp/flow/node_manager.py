# node_manager.py

from typing import Any, Callable, Dict, List, Optional
import importlib
import asyncio
from .node import Node, NodeDefinition
from .flow_utils import adapt_node  # Import the adapt_node function


class NodeManager:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.type_adapters: Dict[str, Callable] = {
            "llm": self._adapt_llm_node,
            "api": self._adapt_api_node,
        }

    def load_nodes(self, node_definitions: List[Dict[str, Any]]):
        for node_def in node_definitions:
            self.load_node(node_def)

    def load_node(self, node_def: Dict[str, Any]):
        node_type = node_def.get("type", "python").lower()
        adapted_node_def = adapt_node(node_def)  # Use the adapt_node function

        if node_type in self.type_adapters:
            adapted_node_def = self.type_adapters[node_type](adapted_node_def)

        node = self._create_node(NodeDefinition(**adapted_node_def))
        self.nodes[node.definition.name] = node

    def _create_node(self, node_def: NodeDefinition) -> Node:
        return Node(
            definition=node_def, callable=self._load_python_function(node_def.function)
        )

    def _load_python_function(self, function_path: str) -> Callable:
        module_name, function_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)

    def _adapt_llm_node(self, node_def: Dict[str, Any]) -> Dict[str, Any]:
        adapted_def = node_def.copy()
        adapted_def["function"] = "jipp.adapters.llm_adapter.execute_llm"
        return adapted_def

    def _adapt_api_node(self, node_def: Dict[str, Any]) -> Dict[str, Any]:
        adapted_def = node_def.copy()
        adapted_def["function"] = "jipp.flow.adapters.api_adapter.execute_api"
        return adapted_def

    async def execute_node(self, node: Node, input_data: Dict[str, Any]) -> Any:
        if asyncio.iscoroutinefunction(node.callable):
            return await node.callable(**input_data)
        else:
            return await asyncio.to_thread(node.callable, **input_data)

    def get_node(self, node_name: str) -> Optional[Node]:
        return self.nodes.get(node_name)

    # Add methods for input evaluation, output storage, etc.
