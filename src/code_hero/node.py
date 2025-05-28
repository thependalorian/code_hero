"""Graph node primitives for the Strategic Framework.

This module defines reusable graph-node primitives that can be composed into workflows.
Each node updates the Pydantic state, calls tools or agents, and returns the new state.
"""

import logging

try:
    from langgraph.func import task
    from langgraph.types import StreamWriter

    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for older LangGraph versions
    LANGGRAPH_AVAILABLE = False

    def task():
        def decorator(func):
            return func

        return decorator

    class StreamWriter:
        async def write(self, data):
            pass


from .state import (
    AgentNode,
    BaseNode,
    HumanLoopNode,
    NodeType,
    ToolNode,
)

logger = logging.getLogger(__name__)


# All node classes are now imported from state.py
# BaseNode, ToolNode, AgentNode, HumanLoopNode are available

# Export all symbols
__all__ = ["NodeType", "BaseNode", "ToolNode", "AgentNode", "HumanLoopNode"]
