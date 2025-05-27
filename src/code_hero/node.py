"""Graph node primitives for the Strategic Framework.

This module defines reusable graph-node primitives that can be composed into workflows.
Each node updates the Pydantic state, calls tools or agents, and returns the new state.
"""

from enum import Enum
import logging
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

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

from .state import StateType, Status
from .utils import call_tool
from .context import managed_state
from .manager import StateManager
from .human_loop import HumanLoopManager

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of nodes available in the graph."""

    AGENT = "agent"
    TOOL = "tool"
    HUMAN_LOOP = "human_loop"
    WORKFLOW = "workflow"


class BaseNode(BaseModel):
    """Base class for all nodes in the graph."""
    name: str
    node_type: NodeType

    model_config = {"arbitrary_types_allowed": True}

    async def run(
        self,
        state: StateType,
        *,
        writer: Optional[StreamWriter] = None
    ) -> StateType:
        """Execute the node's logic on the given state.

        Args:
            state: Current workflow state
            writer: Optional stream writer

        Returns:
            Updated workflow state
        """
        raise NotImplementedError("Subclasses must implement run")
        
    async def __call__(
        self,
        state: StateType,
        *,
        writer: Optional[StreamWriter] = None
    ) -> StateType:
        """Execute the node.
        
        Args:
            state: Current workflow state
            writer: Optional stream writer
            
        Returns:
            Updated workflow state
            
        Raises:
            Exception: If execution fails
        """
        try:
            try:
                return await self.run(state, writer=writer)
            except NotImplementedError as e:
                logger.error(f"Node {self.name} does not implement run: {str(e)}")
                state.status = Status.FAILED
                state.error = f"Node {self.name} does not implement run"
                return state
        except Exception as e:
            error_msg = f"Node {self.name} execution failed: {str(e)}"
            logger.error(error_msg)
            state.status = Status.FAILED
            state.error = error_msg
            return state


class ToolNode(BaseNode):
    """Node that executes a specific tool."""
    tool_name: str

    def __init__(self, **data):
        super().__init__(node_type=NodeType.TOOL, **data)

    async def run(
        self,
        state: StateType,
        *,
        writer: Optional[StreamWriter] = None
    ) -> StateType:
        """Execute the tool and update state.

        Args:
            state: Current workflow state
            writer: Optional stream writer

        Returns:
            Updated workflow state with tool results
        """
        try:
            if writer:
                await writer.write({
                    "type": "tool_start",
                    "data": {"tool": self.tool_name}
                })

            result = await call_tool(self.tool_name, **state)
            state.artifacts[f"{self.name}_result"] = result
            state.status = Status.COMPLETED

            if writer:
                await writer.write({
                    "type": "tool_complete",
                    "data": {"result": result}
                })

            return state

        except Exception as e:
            if writer:
                await writer.write({
                    "type": "tool_error",
                    "data": {"error": str(e)}
                })
            state.status = Status.FAILED
            state.error = str(e)
            return state


class AgentNode(BaseNode):
    """Node that executes an agent."""
    agent_role: str

    def __init__(self, **data):
        super().__init__(node_type=NodeType.AGENT, **data)

    async def run(
        self,
        state: StateType,
        *,
        writer: Optional[StreamWriter] = None
    ) -> StateType:
        """Execute the agent and update state.

        Args:
            state: Current workflow state
            writer: Optional stream writer

        Returns:
            Updated workflow state with agent results
        """
        try:
            if writer:
                await writer.write({
                    "type": "agent_start",
                    "data": {"role": self.agent_role}
                })

            from .agent_expert import execute_agent
            new_state = await execute_agent(self.agent_role, state)

            if writer:
                await writer.write({
                    "type": "agent_complete",
                    "data": {"state": new_state.dict()}
                })

            return new_state

        except Exception as e:
            if writer:
                await writer.write({
                    "type": "agent_error",
                    "data": {"error": str(e)}
                })
            state.status = Status.FAILED
            state.error = str(e)
            return state


class HumanLoopNode(BaseNode):
    """Node that handles human-in-the-loop operations."""
    review_type: str

    def __init__(self, **data):
        super().__init__(node_type=NodeType.HUMAN_LOOP, **data)

    async def run(
        self,
        state: StateType,
        *,
        writer: Optional[StreamWriter] = None
    ) -> StateType:
        """Execute human loop operation and update state.

        Args:
            state: Current workflow state
            writer: Optional stream writer

        Returns:
            Updated workflow state with human feedback
        """
        try:
            if writer:
                await writer.write({
                    "type": "human_loop_start",
                    "data": {"review_type": self.review_type}
                })

            from .human_loop import request_human_feedback
            result = await request_human_feedback(state, writer=writer)

            if writer:
                await writer.write({
                    "type": "human_loop_complete",
                    "data": {"result": result.dict()}
                })

            return result

        except Exception as e:
            if writer:
                await writer.write({
                    "type": "human_loop_error",
                    "data": {"error": str(e)}
                })
            state.status = Status.FAILED
            state.error = str(e)
            return state


# Export all symbols
__all__ = ["NodeType", "BaseNode", "ToolNode", "AgentNode", "HumanLoopNode"]
