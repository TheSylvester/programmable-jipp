# node_manager.py

from typing import Any, Callable, Dict
from .node import BaseNode, Node
import httpx


class NodeManager:
    """
    Manages the registration and retrieval of node execution handlers.
    """

    _node_handlers: Dict[str, Callable] = {}

    @classmethod
    def register_node_type(
        cls, node_type: str, handler: Callable[[Node, Dict[str, Any]], Any]
    ) -> None:
        """
        Registers a handler for a specific node type.
        """
        cls._node_handlers[node_type] = handler

    @classmethod
    async def execute_node(cls, node: Node, input_data: Dict[str, Any]) -> Any:
        node_callable = node.get_callable()
        return await node_callable(node, input_data)

    @staticmethod
    async def api_call(node: Node, input_data: Dict[str, Any]) -> Any:
        """
        Executes an API call.
        """
        url = input_data.pop("url")
        method = input_data.pop("method", "GET")
        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, **input_data)
            response.raise_for_status()
            return response.json()
