"""Workflow execution engine for the Strategic Framework.

This module provides the workflow runner that executes nodes in sequence,
manages state transitions, and handles workflow lifecycle.
"""

from typing import Dict, List, Optional, Type, Any, Tuple
from datetime import datetime
from enum import Enum
import uuid

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticToolsParser

# StreamWriter class for workflow updates
class StreamWriter:
    """Stream writer for workflow updates."""
    async def write(self, data):
        """Write data to stream."""
        pass

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import ToolNode, create_react_agent
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    
    class StateGraph:
        def __init__(self, *args, **kwargs):
            pass
        def add_node(self, *args, **kwargs):
            pass
        def add_edge(self, *args, **kwargs):
            pass
        def add_conditional_edges(self, *args, **kwargs):
            pass
        def compile(self, *args, **kwargs):
            return self
        # Note: No invoke method in fallback - this will raise AttributeError

    class MemorySaver:
        pass
    
    class ToolNode:
        def __init__(self, tools):
            self.tools = tools
    
    START = "start"
    END = "end"
    
    def create_react_agent(*args, **kwargs):
        return None

from langchain.agents import AgentExecutor

from .state import Status, Workflow, AgentState, AgentRole, WorkflowState as StateWorkflowState
from .context import managed_state
from .tools import tool_registry
from .agent_expert import execute_agent, experts
from .prompts import build_prompt
from .human_loop import request_human_feedback
from .utils import generate_id
from .config import WorkflowConfig

# Type aliases for workflow execution
StateType = Dict[str, Any]


class WorkflowRunner:
    """Executes workflows by running nodes in sequence."""

    def __init__(self, workflow: Workflow):
        """Initialize the workflow runner.

        Args:
            workflow: Workflow definition to execute
        """
        self.workflow = workflow
        self._nodes: Dict[str, AgentRole] = {}
        self._tools: List[BaseTool] = []
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        """Initialize tools from registry."""
        self._tools = tool_registry.get_all_tools()

    def register_node(self, node_name: str, agent_role: AgentRole) -> None:
        """Register a node with the workflow.

        Args:
            node_name: Name of the node
            agent_role: Role of the agent for this node
        """
        self._nodes[node_name] = agent_role

    async def invoke(self, initial_state: StateType) -> StateType:
        """Execute the workflow from start to finish.

        Args:
            initial_state: Initial state to begin workflow with

        Returns:
            Final workflow state

        Raises:
            Exception: If workflow execution fails
        """
        async with managed_state(initial_state) as ctx:
            try:
                self.workflow.status = Status.RUNNING
                self.workflow.start_time = datetime.utcnow()
                current_state = initial_state

                for node_name in self.workflow.nodes:
                    agent_role = self._nodes.get(node_name)
                    if not agent_role:
                        raise ValueError(f"Node {node_name} not registered")

                    # Create agent state
                    agent_state = AgentState(
                        id=f"{node_name}_{datetime.utcnow().timestamp()}",
                        agent=agent_role,
                        status=Status.RUNNING,
                        context=current_state.dict() if hasattr(current_state, "dict") else current_state,
                    )

                    # Build prompt
                    prompt = build_prompt(agent_role, agent_state)

                    # Execute agent with tools
                    try:
                        expert = experts.get(agent_role)
                        if not expert:
                            raise ValueError(f"No expert found for role {agent_role}")

                        result = await execute_agent(
                            agent_role=agent_role,
                            state=agent_state,
                            prompt=prompt
                        )

                        # Handle tool calls
                        if isinstance(result, dict) and "tool_calls" in result:
                            tool_results = []
                            for tool_call in result["tool_calls"]:
                                tool = tool_registry.get_tool(tool_call["name"])
                                if tool:
                                    tool_result = await tool(**tool_call["args"])
                                    tool_results.append(tool_result)
                            
                            # Update state with tool results
                            agent_state.artifacts["tool_results"] = tool_results
                            agent_state.status = Status.COMPLETED

                        # Update workflow state
                        if agent_state.status == Status.COMPLETED:
                            current_state = agent_state
                            self.workflow.history.append(node_name)
                        else:
                            self.workflow.status = Status.FAILED
                            self.workflow.error = agent_state.error
                            return current_state

                    except Exception as e:
                        self.workflow.status = Status.FAILED
                        self.workflow.error = str(e)
                        return current_state

                self.workflow.status = Status.COMPLETED
                self.workflow.end_time = datetime.utcnow()
                return current_state

            except Exception as e:
                self.workflow.status = Status.FAILED
                self.workflow.error = str(e)
                raise

    async def resume(self, state: StateType, from_node: str) -> StateType:
        """Resume workflow execution from a specific node.

        Args:
            state: Current workflow state
            from_node: Name of node to resume from

        Returns:
            Updated workflow state
        """
        if from_node not in self.workflow.nodes:
            raise ValueError(f"Node {from_node} not found in workflow")

        start_idx = self.workflow.nodes.index(from_node)
        remaining_nodes = self.workflow.nodes[start_idx:]

        async with managed_state(state) as ctx:
            current_state = state

            for node_name in remaining_nodes:
                agent_role = self._nodes.get(node_name)
                if not agent_role:
                    raise ValueError(f"Node {node_name} not registered")

                # Create agent state
                agent_state = AgentState(
                    id=f"{node_name}_{datetime.utcnow().timestamp()}",
                    agent=agent_role,
                    status=Status.RUNNING,
                    context=current_state.dict() if hasattr(current_state, "dict") else current_state,
                )

                # Build prompt
                prompt = build_prompt(agent_role, agent_state)

                # Execute agent with tools
                try:
                    expert = experts.get(agent_role)
                    if not expert:
                        raise ValueError(f"No expert found for role {agent_role}")

                    result = await execute_agent(
                        agent_role=agent_role,
                        state=agent_state,
                        prompt=prompt
                    )

                    # Handle tool calls
                    if isinstance(result, dict) and "tool_calls" in result:
                        tool_results = []
                        for tool_call in result["tool_calls"]:
                            tool = tool_registry.get_tool(tool_call["name"])
                            if tool:
                                tool_result = await tool(**tool_call["args"])
                                tool_results.append(tool_result)
                        
                        # Update state with tool results
                        agent_state.artifacts["tool_results"] = tool_results
                        agent_state.status = Status.COMPLETED

                    # Update workflow state
                    if agent_state.status == Status.COMPLETED:
                        current_state = agent_state
                        self.workflow.history.append(node_name)
                    else:
                        self.workflow.status = Status.FAILED
                        self.workflow.error = agent_state.error
                        return current_state

                except Exception as e:
                    self.workflow.status = Status.FAILED
                    self.workflow.error = str(e)
                    return current_state

            self.workflow.status = Status.COMPLETED
            self.workflow.end_time = datetime.utcnow()
            return current_state


# Specialized agent node functions with proper tool binding
async def supervisor_node(state: StateType) -> StateType:
    """Supervisor node that routes tasks to appropriate agents."""
    task_description = state.get("context", {}).get("task", "")
    
    # Analyze task to determine agent routing
    if any(keyword in task_description.lower() for keyword in ["research", "search", "find", "analyze"]):
        state["next_agent"] = "research_agent"
        state["task_type"] = "research"
    elif any(keyword in task_description.lower() for keyword in ["write", "document", "content", "blog"]):
        state["next_agent"] = "writer_agent"
        state["task_type"] = "writing"
    elif any(keyword in task_description.lower() for keyword in ["code", "implement", "develop", "program"]):
        state["next_agent"] = "coding_agent"
        state["task_type"] = "coding"
    elif any(keyword in task_description.lower() for keyword in ["review", "check", "validate", "test"]):
        state["next_agent"] = "review_agent"
        state["task_type"] = "review"
    else:
        state["next_agent"] = "research_agent"
        state["task_type"] = "research"
    
    state["status"] = Status.RUNNING
    return state

async def research_agent_node(state: StateType) -> StateType:
    """Research agent specialized for information gathering with tool binding."""
    try:
        # Get available tools for research
        research_tools = tool_registry.get_tools_by_category("research")
        if not research_tools:
            research_tools = [tool_registry.get_tool("search_documents")]
        
        expert = experts.get(AgentRole.RESEARCH)
        if expert:
            agent_state = AgentState(
                id=f"research_{datetime.utcnow().timestamp()}",
                agent=AgentRole.RESEARCH,
                status=Status.RUNNING,
                context=state.get("context", {}),
            )
            
            # Bind tools to the agent
            if hasattr(expert, 'bind_tools'):
                expert.bind_tools(research_tools)
            
            result = await execute_agent(
                agent_role=AgentRole.RESEARCH,
                state=agent_state,
                prompt=build_prompt(AgentRole.RESEARCH, agent_state)
            )
            
            # Handle tool calls if present
            if isinstance(result, AgentState) and result.artifacts:
                state["artifacts"]["research_results"] = result.artifacts
            else:
                state["artifacts"]["research_results"] = result
                
            state["status"] = Status.COMPLETED
            state["needs_review"] = True
        else:
            state["status"] = Status.FAILED
            state["error"] = "Research agent not available"
    except Exception as e:
        state["status"] = Status.FAILED
        state["error"] = str(e)
    
    return state

async def writer_agent_node(state: StateType) -> StateType:
    """Writer agent specialized for content creation with tool binding."""
    try:
        # Get available tools for writing
        writing_tools = tool_registry.get_tools_by_category("content")
        if not writing_tools:
            writing_tools = [tool_registry.get_tool("generate_code")]  # Can be used for content generation
        
        # Use prompt engineer for enhanced writing prompts
        prompt_expert = experts.get(AgentRole.PROMPT_ENGINEER)
        if prompt_expert:
            # Create enhanced writing prompt
            prompt_state = AgentState(
                id=f"prompt_{datetime.utcnow().timestamp()}",
                agent=AgentRole.PROMPT_ENGINEER,
                status=Status.RUNNING,
                context={"task": "Create writing prompt for: " + str(state.get("context", {}))},
            )
            
            enhanced_prompt = await execute_agent(
                agent_role=AgentRole.PROMPT_ENGINEER,
                state=prompt_state,
                prompt=build_prompt(AgentRole.PROMPT_ENGINEER, prompt_state)
            )
            
            # Use enhanced prompt for writing
            state["artifacts"]["enhanced_prompt"] = enhanced_prompt
        
        # Execute writing task (using NextJS expert as writer for now)
        expert = experts.get(AgentRole.NEXTJS_EXPERT)
        if expert:
            # Bind tools to the agent
            if hasattr(expert, 'bind_tools'):
                expert.bind_tools(writing_tools)
                
            agent_state = AgentState(
                id=f"writer_{datetime.utcnow().timestamp()}",
                agent=AgentRole.NEXTJS_EXPERT,
                status=Status.RUNNING,
                context=state.get("context", {}),
            )
            
            result = await execute_agent(
                agent_role=AgentRole.NEXTJS_EXPERT,
                state=agent_state,
                prompt=build_prompt(AgentRole.NEXTJS_EXPERT, agent_state)
            )
            
            # Handle tool calls if present
            if isinstance(result, AgentState) and result.artifacts:
                state["artifacts"]["written_content"] = result.artifacts
            else:
                state["artifacts"]["written_content"] = result
                
            state["status"] = Status.COMPLETED
            state["needs_review"] = True
        else:
            state["status"] = Status.FAILED
            state["error"] = "Writer agent not available"
    except Exception as e:
        state["status"] = Status.FAILED
        state["error"] = str(e)
    
    return state

async def coding_agent_node(state: StateType) -> StateType:
    """Coding agent specialized for software development with tool binding."""
    try:
        # Get available tools for coding
        coding_tools = [
            tool_registry.get_tool("generate_code"),
            tool_registry.get_tool("validate_code"),
            tool_registry.get_tool("analyze_code")
        ]
        coding_tools = [tool for tool in coding_tools if tool is not None]
        
        # Determine coding specialization
        task = state.get("context", {}).get("task", "")
        
        if "fastapi" in task.lower() or "api" in task.lower():
            expert = experts.get(AgentRole.FASTAPI_EXPERT)
            agent_role = AgentRole.FASTAPI_EXPERT
        elif "nextjs" in task.lower() or "frontend" in task.lower():
            expert = experts.get(AgentRole.NEXTJS_EXPERT)
            agent_role = AgentRole.NEXTJS_EXPERT
        else:
            expert = experts.get(AgentRole.LANGCHAIN_EXPERT)
            agent_role = AgentRole.LANGCHAIN_EXPERT
        
        if expert:
            # Bind tools to the agent
            if hasattr(expert, 'bind_tools'):
                expert.bind_tools(coding_tools)
                
            agent_state = AgentState(
                id=f"coding_{datetime.utcnow().timestamp()}",
                agent=agent_role,
                status=Status.RUNNING,
                context=state.get("context", {}),
            )
            
            result = await execute_agent(
                agent_role=agent_role,
                state=agent_state,
                prompt=build_prompt(agent_role, agent_state)
            )
            
            # Handle tool calls if present
            if isinstance(result, AgentState) and result.artifacts:
                state["artifacts"]["code_output"] = result.artifacts
            else:
                state["artifacts"]["code_output"] = result
                
            state["status"] = Status.COMPLETED
            state["needs_review"] = True
        else:
            state["status"] = Status.FAILED
            state["error"] = "Coding agent not available"
    except Exception as e:
        state["status"] = Status.FAILED
        state["error"] = str(e)
    
    return state

async def review_agent_node(state: StateType) -> StateType:
    """Review agent for quality assurance with tool binding."""
    try:
        # Get available tools for review
        review_tools = [
            tool_registry.get_tool("validate_code"),
            tool_registry.get_tool("analyze_code")
        ]
        review_tools = [tool for tool in review_tools if tool is not None]
        
        # Use LangChain expert for review tasks
        expert = experts.get(AgentRole.LANGCHAIN_EXPERT)
        if expert:
            # Bind tools to the agent
            if hasattr(expert, 'bind_tools'):
                expert.bind_tools(review_tools)
                
            # Prepare review context
            review_context = {
                "task": "Review and validate the following work:",
                "artifacts": state.get("artifacts", {}),
                "original_context": state.get("context", {})
            }
            
            agent_state = AgentState(
                id=f"review_{datetime.utcnow().timestamp()}",
                agent=AgentRole.LANGCHAIN_EXPERT,
                status=Status.RUNNING,
                context=review_context,
            )
            
            result = await execute_agent(
                agent_role=AgentRole.LANGCHAIN_EXPERT,
                state=agent_state,
                prompt=build_prompt(AgentRole.LANGCHAIN_EXPERT, agent_state)
            )
            
            # Handle tool calls if present
            if isinstance(result, AgentState) and result.artifacts:
                state["artifacts"]["review_results"] = result.artifacts
            else:
                state["artifacts"]["review_results"] = result
                
            state["status"] = Status.COMPLETED
            state["needs_feedback"] = True  # Request human feedback after review
        else:
            state["status"] = Status.FAILED
            state["error"] = "Review agent not available"
    except Exception as e:
        state["status"] = Status.FAILED
        state["error"] = str(e)
    
    return state

def create_tool_node(tools: List[BaseTool]) -> ToolNode:
    """Create a tool node for LangGraph with proper tool binding.
    
    Args:
        tools: List of tools to bind
        
    Returns:
        Configured tool node
    """
    if LANGGRAPH_AVAILABLE:
        return ToolNode(tools)
    else:
        # Fallback implementation
        class FallbackToolNode:
            def __init__(self, tools):
                self.tools = tools
                
            async def __call__(self, state):
                # Simple fallback - just return the state
                return state
        
        return FallbackToolNode(tools)

def create_workflow_graph() -> StateGraph:
    """Create a multi-agent workflow graph with specialized agents.
    
    Returns:
        Compiled workflow graph with proper agent coordination
    """
    # Create graph with state schema
    graph = StateGraph(StateWorkflowState)

    # Add specialized agent nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research_agent", research_agent_node)
    graph.add_node("writer_agent", writer_agent_node)
    graph.add_node("coding_agent", coding_agent_node)
    graph.add_node("review_agent", review_agent_node)
    graph.add_node("request_feedback", request_human_feedback)

    # Routing functions
    def route_to_agent(state: StateType) -> str:
        """Route to appropriate agent based on supervisor decision."""
        return state.get("next_agent", "research_agent")

    def should_request_feedback(state: StateType) -> str:
        """Determine if human feedback is needed."""
        if state.get("needs_feedback", False):
            return "request_feedback"
        elif state.get("needs_review", False):
            return "review_agent"
        else:
            return "end"

    def should_continue(state: StateType) -> str:
        """Determine workflow continuation."""
        status = state.get("status")
        if status == Status.FAILED:
            return "end"
        elif state.get("needs_review", False):
            return "review_agent"
        elif state.get("needs_feedback", False):
            return "request_feedback"
        else:
            return "end"

    # Add end node
    def end_workflow(state: StateType) -> StateType:
        """End the workflow."""
        state["end_time"] = datetime.utcnow().isoformat()
        if state.get("status") != Status.FAILED:
            state["status"] = Status.COMPLETED
        return state

    graph.add_node("end", end_workflow)

    # Add edges
    graph.add_edge(START, "supervisor")
    
    # Supervisor routes to specialized agents
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "research_agent": "research_agent",
            "writer_agent": "writer_agent",
            "coding_agent": "coding_agent",
            "review_agent": "review_agent"
        }
    )

    # Agent completion routing
    for agent in ["research_agent", "writer_agent", "coding_agent"]:
        graph.add_conditional_edges(
            agent,
            should_request_feedback,
            {
                "request_feedback": "request_feedback",
                "review_agent": "review_agent",
                "end": "end"
            }
        )

    # Review agent routing
    graph.add_conditional_edges(
        "review_agent",
        should_request_feedback,
        {
            "request_feedback": "request_feedback",
            "end": "end"
        }
    )

    # Feedback routing
    graph.add_conditional_edges(
        "request_feedback",
        should_continue,
        {
            "review_agent": "review_agent",
            "request_feedback": "request_feedback",
            "end": "end"
        }
    )

    graph.add_edge("end", END)

    return graph.compile(checkpointer=MemorySaver())


async def workflow_runner(
    initial_state: StateType,
    *,
    previous: Optional[Dict[str, Any]] = None,
    writer: Optional[StreamWriter] = None
) -> Dict[str, Any]:
    """Execute workflow with proper LangGraph invoke method.
    
    Args:
        initial_state: Initial workflow state
        previous: Optional previous state for resumption
        writer: Optional stream writer
        
    Returns:
        Final workflow state
    """
    try:
        # Initialize workflow state
        state = {
            "id": initial_state.get("id", generate_id()),
            "status": Status.PENDING,
            "context": initial_state.get("context", {}),
            "artifacts": initial_state.get("artifacts", {}),
            "start_time": datetime.utcnow(),
            "error": None
        }

        if previous:
            state.update(previous)

        if writer:
            await writer.write({
                "type": "workflow_start",
                "data": {"state": state}
            })

        # Check if LangGraph is available
        if not LANGGRAPH_AVAILABLE:
            # Fallback execution without LangGraph
            return {
                "status": Status.COMPLETED,
                "result": state,
                "message": "LangGraph not available, using fallback execution"
            }

        # Get workflow graph
        graph = create_workflow_graph()
        
        # Execute workflow using invoke method with proper configuration
        try:
            # Prepare configuration for checkpointer
            config = {
                "configurable": {
                    "thread_id": state.get("id", "default")
                }
            }
            
            # Use invoke method which is the standard LangGraph execution method
            final_state = graph.invoke(state, config=config)
            
            if writer:
                await writer.write({
                    "type": "workflow_complete",
                    "data": {"result": final_state}
                })

            return {
                "status": Status.COMPLETED,
                "result": final_state
            }

        except AttributeError as e:
            if "invoke" in str(e):
                # Handle case where invoke method is not available
                if writer:
                    await writer.write({
                        "type": "workflow_error",
                        "data": {"error": f"LangGraph invoke method not available: {str(e)}"}
                    })
                return {
                    "status": Status.FAILED,
                    "error": f"LangGraph invoke method not available: {str(e)}",
                    "fallback_result": state
                }
            else:
                raise
        except Exception as e:
            if writer:
                await writer.write({
                    "type": "workflow_error",
                    "data": {"error": str(e)}
                })
            state["status"] = Status.FAILED
            state["error"] = str(e)
            return {
                "status": Status.FAILED,
                "error": str(e)
            }

    except Exception as e:
        if writer:
            await writer.write({
                "type": "workflow_error",
                "data": {"error": str(e)}
            })
        return {
            "status": Status.FAILED,
            "error": str(e)
        }


# Export all symbols
__all__ = ["WorkflowRunner", "create_workflow_graph", "workflow_runner"]
