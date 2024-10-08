# node.py

from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Any, Dict, Optional, Callable

if TYPE_CHECKING:
    from .execution_context import ExecutionContext


class NodeDefinition(BaseModel):
    name: str
    type: str
    function: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    max_retries: int = 0
    model: Optional[str] = None  # For LLM nodes
    prompt_template: Optional[str] = None  # For LLM nodes
    on_failure: Optional[Callable[[Exception, "Node", "ExecutionContext"], None]] = None
    on_success: Optional[Callable[["Node", "ExecutionContext"], None]] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.name:
            raise ValueError("Node name cannot be empty")
        if self.type.lower() not in ["python", "llm", "api"]:
            raise ValueError(f"Invalid node type: {self.type}")
        self.type = (
            self.type.capitalize()
        )  # Normalize the type to capitalize first letter


class Node(BaseModel):
    definition: NodeDefinition
    callable: Callable

    def __str__(self):
        return f"Node(name={self.definition.name}, type={self.definition.type})"

    def __repr__(self):
        return f"Node(name={self.definition.name}, type={self.definition.type}, function={self.definition.function})"

    async def execute(self, input_data: Dict[str, Any]) -> Any:
        # Implementation can be added here if needed
        pass
