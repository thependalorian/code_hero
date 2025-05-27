"""Supervisor Agent for Multi-Agent Backend System.

This module implements the SupervisorAgent class responsible for orchestrating
multi-agent collaboration, managing task execution flow, and coordinating
workflow graphs using contextual awareness and project states.
"""

from typing import Optional, List, Dict, Any, Set
import asyncio
from datetime import datetime
import logging
from enum import Enum

from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticToolsParser
from pydantic import BaseModel, Field

try:
    from langgraph.types import StreamWriter
except ImportError:
    # Fallback for older LangGraph versions
    class StreamWriter:
        async def write(self, data):
            pass

from .state import AgentRole, AgentState, TaskState, ProjectState, Status, GraphState
from .manager import StateManager
from .agent_expert import LangChainExpert, FastAPIExpert, NextJSExpert
from .human_loop import request_human_feedback
from .logger import StructuredLogger
from .interfaces import ServiceInterface
from .tools import tool_registry

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels for the supervisor."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SupervisorState(BaseModel):
    """State model for the supervisor agent."""
    initialized: bool = False
    active_workflows: set[str] = Field(default_factory=set)
    status: Optional[Status] = None
    error: Optional[str] = None


class NodeExecutionResult(BaseModel):
    """Result of a node execution."""
    node_id: str
    status: Status
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: float = 0.0


class SupervisorAgent(ServiceInterface):
    """Supervisor agent for orchestrating multi-agent workflows."""

    def __init__(self, state_manager: StateManager, logger: StructuredLogger):
        """Initialize the supervisor agent.

        Args:
            state_manager: State management service
            logger: Structured logging service
        """
        self.state_manager = state_manager
        self.logger = logger
        self._experts = {
            AgentRole.LANGCHAIN_EXPERT: LangChainExpert(),
            AgentRole.FASTAPI_EXPERT: FastAPIExpert(),
            AgentRole.NEXTJS_EXPERT: NextJSExpert(),
        }
        self.state = SupervisorState()
        self._registered_tools: Set[str] = set()
        
    async def initialize(self) -> None:
        """Initialize the supervisor agent."""
        if self.state.initialized:
            return
            
        try:
            # Initialize each expert
            for expert in self._experts.values():
                if hasattr(expert, 'initialize'):
                    await expert.initialize()
                    
            self.state.initialized = True
            self.logger.info("Supervisor agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize supervisor agent: {e}")
            raise
                
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clean up each expert
            for expert in self._experts.values():
                if hasattr(expert, 'cleanup'):
                    await expert.cleanup()
                    
            self.state = SupervisorState()
            self.logger.info("Supervisor agent cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Failed to clean up supervisor agent: {e}")
            raise

    async def check_health(self) -> dict:
        """Check supervisor health status."""
        return {
            "initialized": self.state.initialized,
            "status": self.state.status,
            "active_workflows": len(self.state.active_workflows),
            "state_manager_connected": self.state_manager is not None,
            "logger_connected": self.logger is not None
        }

    async def execute_node(
        self,
        node: Any,
        project_id: str,
        *,
        writer: Optional[StreamWriter] = None,
        timeout: float = 300.0  # 5 minutes default timeout
    ) -> NodeExecutionResult:
        """Execute a single node with timeout.

        Args:
            node: Node to execute
            project_id: Project identifier
            writer: Optional stream writer
            timeout: Timeout in seconds

        Returns:
            Node execution result
        """
        start_time = datetime.utcnow()
        
        try:
            # Set up execution with timeout
            result = await asyncio.wait_for(
                self._execute_node_internal(node, project_id, writer=writer),
                timeout=timeout
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            return NodeExecutionResult(
                node_id=node.name,
                status=Status.COMPLETED,
                result=result,
                duration=duration
            )
            
        except asyncio.TimeoutError:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error = f"Node {node.name} execution timed out after {duration:.2f}s"
            self.logger.error(error)
            return NodeExecutionResult(
                node_id=node.name,
                status=Status.FAILED,
                error=error,
                duration=duration
            )

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error = f"Node {node.name} execution failed: {str(e)}"
            self.logger.error(error)
            return NodeExecutionResult(
                node_id=node.name,
                status=Status.FAILED,
                error=error,
                duration=duration
            )

    async def _execute_node_internal(
        self,
        node: Any,
        project_id: str,
        *,
        writer: Optional[StreamWriter] = None
    ) -> Dict[str, Any]:
        """Internal node execution logic.
        
        Args:
            node: Node to execute
            project_id: Project identifier
            writer: Optional stream writer
            
        Returns:
            Node execution result
        """
        if node.type == "agent":
            if writer:
                await writer.write({
                    "type": "agent_start",
                    "data": {"role": node.agent_role}
                })

            from .agent_expert import execute_agent
            result = await execute_agent(node.agent_role, node.state)

            if writer:
                await writer.write({
                    "type": "agent_complete",
                    "data": {"state": result.dict()}
                })

            return {"result": result.dict()}

        elif node.type == "tool":
            if writer:
                await writer.write({
                    "type": "tool_start",
                    "data": {"tool": node.tool_name}
                })

            result = await self.execute_tool(node)

            if writer:
                await writer.write({
                    "type": "tool_complete",
                    "data": {"result": result}
                })

            return {"result": result}

        elif node.type == "human":
            await request_human_feedback(
                project_id=project_id,
                task=node.task,
                reason="Human input required by workflow",
            )
            return {"status": "pending_human_feedback"}

        else:
            error = f"Unknown node type: {node.type}"
            self.logger.warning(error, extra={"project_id": project_id, "node_id": node.name})
            return {"error": error}

    def _register_tools(self, tools: List[BaseTool]) -> None:
        """Register tools, deduplicating by name.
        
        Args:
            tools: Tools to register
        """
        for tool in tools:
            if tool.name not in self._registered_tools:
                self._registered_tools.add(tool.name)
                for expert in self._experts.values():
                    expert.tools.append(tool)

    def _extract_primary_output(self, result: AgentState) -> str:
        """Extract primary output from agent state.
        
        Args:
            result: Agent state to extract from
            
        Returns:
            Primary output string
        """
        if isinstance(result.artifacts, dict):
            return (
                result.artifacts.get("response") or
                result.artifacts.get("implementation") or
                result.artifacts.get("search_results") or
                result.artifacts.get("analysis") or
                "Task completed successfully, but no specific response was generated."
            )
        return "Task completed but no response was generated."

    async def __call__(self, state: AgentState, writer: Optional[StreamWriter] = None) -> Dict[str, Any]:
        """Execute the supervisor agent.
        
        Args:
            state: Current agent state
            writer: Optional stream writer
            
        Returns:
            Task result with optional tool calls
        """
        try:
            # Get expert agent
            expert = self._experts.get(state.agent)
            if not expert:
                raise ValueError(f"No expert found for role {state.agent}")
            
            # Execute expert
            result = await expert(state, writer=writer)
            
            # Handle tool calls in result
            if isinstance(result, dict) and "tool_calls" in result:
                return result
            
            # Handle string responses
            if isinstance(result, str):
                return {"content": result}
            
            # Handle agent state responses
            if isinstance(result, AgentState):
                if result.status == Status.COMPLETED:
                    # Try to get response from different possible locations
                    if isinstance(result.artifacts, dict):
                        response = (
                            result.artifacts.get("response") or
                            result.artifacts.get("implementation") or
                            result.artifacts.get("search_results") or
                            result.artifacts.get("analysis") or
                            "Task completed successfully, but no specific response was generated."
                        )
                        return {"content": str(response)}
                    else:
                        return {"content": "Task completed but no response was generated."}
                else:
                    return {"content": result.error if result.error else "Task execution failed"}
                
            # Handle unexpected response types
            self.logger.log_warning(
                "Unexpected response type from agent",
                {"response_type": str(type(result))}
            )
            return {"content": "Unexpected response from agent"}

        except Exception as e:
            error_msg = f"Error executing {state.agent}: {str(e)}"
            self.logger.log_error(error_msg)
            return {"content": error_msg}

    async def dispatch_task(
        self,
        project_id: str,
        task: AgentState,
        tools: Optional[List[BaseTool]] = None,
        *,
        writer: Optional[StreamWriter] = None
    ) -> Dict[str, Any]:
        """Dispatch a task to the appropriate expert agent.

        Args:
            project_id: Project identifier
            task: Agent state to execute
            tools: Optional list of tools to use
            writer: Optional stream writer

        Returns:
            Task result with optional tool calls
        """
        try:
            self.logger.log_event(
                "task_dispatch",
                {"project_id": project_id, "agent": str(task.agent)},
                task
            )
            
            # Get expert agent
            expert = self._experts.get(task.agent)
            if not expert:
                raise ValueError(f"No expert found for role {task.agent}")
            
            # Register tools if provided
            if tools:
                self._register_tools(tools)
            
            # Execute agent using __call__
            result = await expert(task, writer=writer)
            
            # Handle tool calls in result
            if isinstance(result, dict) and "tool_calls" in result:
                return result
            
            # Handle string responses
            if isinstance(result, str):
                return {"content": result}
            
            # Handle agent state responses
            if isinstance(result, AgentState):
                if result.status == Status.COMPLETED:
                    return {"content": self._extract_primary_output(result)}
                else:
                    return {"content": result.error if result.error else "Task execution failed"}
                
            # Handle unexpected response types
            self.logger.log_warning(
                "Unexpected response type from agent",
                {"response_type": str(type(result))}
            )
            return {"content": "Unexpected response from agent"}

        except Exception as e:
            error_msg = f"Error executing {task.agent}: {str(e)}"
            self.logger.log_error(error_msg)
            return {"content": error_msg}

    async def reassign_task(
        self,
        project_id: str,
        task: TaskState,
        tools: Optional[List[BaseTool]] = None
    ) -> Optional[Dict[str, Any]]:
        """Reassign a failed or unassigned task.

        Args:
            project_id: Project identifier
            task: Task to reassign
            tools: Optional list of tools to use

        Returns:
            Optional task result
        """
        project_state = await self.state_manager.get_project_state(project_id)

        # Try to find an idle agent
        for role, agent_state in self.state_manager.agent_states.get(
            project_id, {}
        ).items():
            if agent_state.status == Status.IDLE:
                task.assigned_to = role
                task.reassignment_count = (task.reassignment_count or 0) + 1

                self.logger.info(
                    f"Reassigning task to {role}",
                    extra={
                        "project_id": project_id,
                        "task_id": task.id,
                        "new_agent": role,
                        "reassignment_count": task.reassignment_count,
                    },
                )

                return await self.dispatch_task(project_id, task, tools=tools)

        # No available agent - escalate to human
        self.logger.warning(
            "No available agents. Escalating to human.",
            extra={"project_id": project_id, "task_id": task.id},
        )

        await request_human_feedback(
            project_id=project_id,
            task=task,
            reason="No available agents for task execution",
        )
        return None

    async def coordinate_multi_agent_task(
        self,
        project_id: str,
        task_description: str,
        *,
        writer: Optional[StreamWriter] = None
    ) -> Dict[str, Any]:
        """Coordinate a multi-agent task using the enhanced workflow.

        Args:
            project_id: Project identifier
            task_description: Description of the task to execute
            writer: Optional stream writer for real-time updates

        Returns:
            Task execution results
        """
        try:
            from .workflow import create_workflow_graph
            from .state import WorkflowState
            
            # Create initial workflow state
            initial_state = WorkflowState(
                id=f"multi_agent_{project_id}_{datetime.utcnow().timestamp()}",
                workflow_id=f"multi_agent_{project_id}_{datetime.utcnow().timestamp()}",
                name="Multi-Agent Coordination",
                status=Status.PENDING,
                context={"task": task_description, "project_id": project_id},
                artifacts={},
                start_time=datetime.utcnow()
            )
            
            # Get the multi-agent workflow graph
            workflow_graph = create_workflow_graph()
            
            # Execute the workflow
            if writer:
                await writer.write({
                    "type": "multi_agent_start",
                    "data": {
                        "project_id": project_id,
                        "task": task_description,
                        "workflow_id": initial_state.id
                    }
                })
            
            # Run the workflow
            try:
                # Convert initial state to dict for LangGraph
                state_dict = initial_state.dict()
                
                # Use the standard LangGraph invoke method
                final_state = workflow_graph.invoke(
                    state_dict,
                    config={"configurable": {"thread_id": project_id}}
                )
                    
            except Exception as e:
                error_msg = f"Workflow execution failed: {str(e)}"
                self.logger.error(error_msg, extra={"project_id": project_id})
                final_state = {
                    "status": Status.FAILED,
                    "error": error_msg,
                    "artifacts": {}
                }
            
            if writer:
                await writer.write({
                    "type": "multi_agent_complete",
                    "data": {
                        "project_id": project_id,
                        "workflow_id": initial_state.id,
                        "final_state": final_state
                    }
                })
            
            self.logger.info(
                "Multi-agent task completed",
                extra={
                    "project_id": project_id,
                    "workflow_id": initial_state.id,
                    "status": final_state.get("status"),
                    "artifacts_count": len(final_state.get("artifacts", {}))
                }
            )
            
            return {
                "status": final_state.get("status"),
                "artifacts": final_state.get("artifacts", {}),
                "workflow_id": initial_state.id,
                "execution_time": final_state.get("end_time")
            }
            
        except Exception as e:
            error_msg = f"Multi-agent coordination failed: {str(e)}"
            self.logger.error(error_msg, extra={"project_id": project_id})
            
            if writer:
                await writer.write({
                    "type": "multi_agent_error",
                    "data": {
                        "project_id": project_id,
                        "error": error_msg
                    }
                })
            
            return {
                "status": Status.FAILED,
                "error": error_msg,
                "artifacts": {}
            }

    async def execute_workflow(self, project_id: str) -> None:
        """Execute a workflow graph for a project.

        Args:
            project_id: Project identifier
        """
        project_state = await self.state_manager.get_project_state(project_id)
        graph_state = project_state.graph_state
        active_graph = graph_state.graphs.get(graph_state.active_graph)

        if not active_graph:
            self.logger.error(f"No active graph found for project {project_id}")
            return

        self.logger.info(
            f"Starting workflow execution",
            extra={
                "project_id": project_id,
                "graph_id": graph_state.active_graph,
                "node_count": len(active_graph.nodes),
            },
        )

        try:
            # Group nodes by their dependencies
            independent_nodes = []
            dependent_nodes = {}

            for node in active_graph.nodes:
                if not node.dependencies:
                    independent_nodes.append(node)
                else:
                    dependent_nodes[node.name] = node
            
            # Execute independent nodes concurrently
            node_results = await asyncio.gather(
                *[
                    self.execute_node(node, project_id)
                    for node in independent_nodes
                ],
                return_exceptions=True
            )
            
            # Process results and update graph state
            completed_nodes = set()
            failed_nodes = set()
            
            for node, result in zip(independent_nodes, node_results):
                if isinstance(result, Exception):
                    self.logger.error(
                        f"Node {node.name} failed with exception: {str(result)}",
                        extra={
                            "project_id": project_id,
                            "node": node.name,
                            "error": str(result)
                        }
                    )
                    failed_nodes.add(node.name)
                    continue
                    
                if result.status == Status.COMPLETED:
                    completed_nodes.add(node.name)
                    node.result = result.result
                else:
                    failed_nodes.add(node.name)
                    node.error = result.error
                    
                # Update graph state after each node
                await self.state_manager.update_state(graph_state)
            
            # Execute dependent nodes in order of dependencies
            while dependent_nodes:
                ready_nodes = [
                    node for node in dependent_nodes.values()
                    if all(dep in completed_nodes for dep in node.dependencies)
                    and not any(dep in failed_nodes for dep in node.dependencies)
                ]
                
                if not ready_nodes:
                    break
                    
                # Execute ready nodes concurrently
                node_results = await asyncio.gather(
                    *[
                        self.execute_node(node, project_id)
                        for node in ready_nodes
                    ],
                    return_exceptions=True
                )
                
                # Process results
                for node, result in zip(ready_nodes, node_results):
                    del dependent_nodes[node.name]
                    
                    if isinstance(result, Exception):
                        self.logger.error(
                            f"Node {node.name} failed with exception: {str(result)}",
                            extra={
                                "project_id": project_id,
                                "node": node.name,
                                "error": str(result)
                            }
                        )
                        failed_nodes.add(node.name)
                        continue
                        
                    if result.status == Status.COMPLETED:
                        completed_nodes.add(node.name)
                        node.result = result.result
                    else:
                        failed_nodes.add(node.name)
                        node.error = result.error
                        
                    # Update graph state after each node
                    await self.state_manager.update_state(graph_state)
            
            # Check if any nodes are still pending
            remaining_nodes = set(dependent_nodes.keys())
            if remaining_nodes:
                self.logger.warning(
                    f"Some nodes could not be executed due to failed dependencies",
                    extra={
                        "project_id": project_id,
                        "remaining_nodes": list(remaining_nodes),
                        "failed_nodes": list(failed_nodes)
                    }
                )

            # Mark graph as completed if all nodes succeeded
            if not failed_nodes and not remaining_nodes:
                graph_state.status = Status.COMPLETED
            else:
                graph_state.status = Status.FAILED
                graph_state.error = f"Failed nodes: {list(failed_nodes)}; Unexecuted nodes: {list(remaining_nodes)}"
                
            graph_state.end_time = datetime.utcnow()
            await self.state_manager.update_state(graph_state)

            self.logger.info(
                "Workflow execution completed",
                extra={
                    "project_id": project_id,
                    "graph_id": graph_state.active_graph,
                    "duration": (
                        graph_state.end_time - graph_state.start_time
                    ).total_seconds(),
                    "status": graph_state.status,
                    "completed_nodes": list(completed_nodes),
                    "failed_nodes": list(failed_nodes),
                    "remaining_nodes": list(remaining_nodes)
                }
            )

        except Exception as e:
            error = f"Workflow execution failed: {str(e)}"
            self.logger.error(
                error,
                extra={
                    "project_id": project_id,
                    "graph_id": graph_state.active_graph
                }
            )
            graph_state.status = Status.FAILED
            graph_state.error = error
            graph_state.end_time = datetime.utcnow()
            await self.state_manager.update_state(graph_state)

    async def execute_tool(self, node: Any) -> str:
        """Execute a tool node in the workflow.

        Args:
            node: Tool node to execute

        Returns:
            Tool execution result
        """
        self.logger.info(f"Executing tool: {node.tool_name}")
        # TODO: Implement tool execution logic
        return "Tool executed successfully"

    async def monitor_project(self, project_id: str) -> None:
        """Monitor project progress and handle failures.

        Args:
            project_id: Project identifier
        """
        project_state = await self.state_manager.get_project_state(project_id)

        # Check agent states
        for agent_role, agent_state in self.state_manager.agent_states.get(
            project_id, {}
        ).items():
            if agent_state.status == Status.FAILED:
                self.logger.warning(
                    f"Agent {agent_role} failed",
                    extra={"project_id": project_id, "agent": agent_role},
                )

                # Reassign incomplete tasks
                for task in agent_state.tasks:
                    if task.status not in [Status.COMPLETED, Status.CANCELLED]:
                        await self.reassign_task(project_id, task)

        # Check for stalled tasks
        current_time = datetime.utcnow()
        for task in project_state.tasks:
            if task.status == Status.RUNNING:
                duration = (current_time - task.start_time).total_seconds()
                if duration > project_state.config.task_timeout_seconds:
                    self.logger.warning(
                        f"Task {task.id} has stalled",
                        extra={
                            "project_id": project_id,
                            "task_id": task.id,
                            "duration": duration,
                        },
                    )
                    await self.handle_stalled_task(project_id, task)

    async def handle_stalled_task(self, project_id: str, task: TaskState) -> None:
        """Handle a stalled task by reassigning or escalating.

        Args:
            project_id: Project identifier
            task: Stalled task
        """
        if task.reassignment_count < 3:
            await self.reassign_task(project_id, task)
        else:
            self.logger.error(
                f"Task {task.id} failed after multiple reassignments",
                extra={
                    "project_id": project_id,
                    "task_id": task.id,
                    "reassignment_count": task.reassignment_count,
                },
            )
            await request_human_feedback(
                project_id=project_id,
                task=task,
                reason="Task failed after multiple reassignments",
            )

    async def should_continue_analysis(self, state: AgentState) -> bool:
        """Check if analysis should continue.

        Args:
            state: Current agent state

        Returns:
            True if analysis should continue
        """
        # Check if we have enough insights
        if not state.artifacts.get("insights"):
            return True

        # Check if we need more analysis based on agent role
        if (
            state.agent == AgentRole.RESEARCH
            and len(state.artifacts.get("insights", [])) < 3
        ):
            return True

        # Check if we need implementation details
        if state.agent == AgentRole.IMPLEMENTATION and not state.artifacts.get(
            "technical_specs"
        ):
            return True

        return False

    async def should_generate_code(self, state: AgentState) -> bool:
        """Check if code generation should proceed.

        Args:
            state: Current agent state

        Returns:
            True if code should be generated
        """
        # Check prerequisites
        if not state.artifacts.get("technical_specs"):
            return False

        # Check if code needs to be generated
        if not state.artifacts.get("generated_code"):
            return True

        # Check if code needs to be regenerated based on review
        if state.artifacts.get("review_feedback") and not state.artifacts.get(
            "review_addressed"
        ):
            return True

        return False

    async def should_review_code(self, state: AgentState) -> bool:
        """Check if code review should proceed.

        Args:
            state: Current agent state

        Returns:
            True if code should be reviewed
        """
        # Check if there's code to review
        if not state.artifacts.get("generated_code"):
            return False

        # Check if code needs initial review
        if not state.artifacts.get("review_feedback"):
            return True

        # Check if code needs re-review after changes
        if state.artifacts.get("review_addressed") and not state.artifacts.get(
            "review_passed"
        ):
            return True

        return False

    async def should_enforce_standards(self, state: AgentState) -> bool:
        """Check if standards enforcement should proceed.

        Args:
            state: Current agent state

        Returns:
            True if standards should be enforced
        """
        # Check if code has been reviewed
        if not state.artifacts.get("review_passed"):
            return False

        # Check if standards need to be checked
        if not state.artifacts.get("standards_checked"):
            return True

        # Check if standards need to be re-checked
        if state.artifacts.get("standards_violations") and not state.artifacts.get(
            "standards_passed"
        ):
            return True

        return False

    async def should_generate_docs(self, state: AgentState) -> bool:
        """Check if documentation generation should proceed.

        Args:
            state: Current agent state

        Returns:
            True if documentation should be generated
        """
        # Check if code has passed standards
        if not state.artifacts.get("standards_passed"):
            return False

        # Check if docs need to be generated
        if not state.artifacts.get("generated_docs"):
            return True

        # Check if docs need to be regenerated
        if state.artifacts.get("doc_review_feedback") and not state.artifacts.get(
            "doc_review_addressed"
        ):
            return True

        return False

    async def should_review_docs(self, state: AgentState) -> bool:
        """Check if documentation review should proceed.

        Args:
            state: Current agent state

        Returns:
            True if documentation should be reviewed
        """
        # Check if there are docs to review
        if not state.artifacts.get("generated_docs"):
            return False

        # Check if docs need initial review
        if not state.artifacts.get("doc_review_feedback"):
            return True

        # Check if docs need re-review
        if state.artifacts.get("doc_review_addressed") and not state.artifacts.get(
            "doc_review_passed"
        ):
            return True

        return False

    async def validate_graph(self, graph_id: str) -> bool:
        """Validate graph structure and connections.
        
        Args:
            graph_id: Graph ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self.state.initialized:
            await self.initialize()

        try:
            # Get graph state
            graph_state = await self.state_manager.get_state(graph_id)
            if not graph_state or not isinstance(graph_state, GraphState):
                raise ValueError(f"Invalid graph state: {graph_id}")
                
            # Get active graph
            active_graph = graph_state.active_graph
            if not active_graph:
                raise ValueError(f"No active graph in state: {graph_id}")
                
            # Validate nodes
            for node in active_graph.nodes:
                # Check node has valid ID
                if not node.id:
                    raise ValueError(f"Node missing ID in graph: {graph_id}")
                    
                # Check node has valid type
                if not node.type:
                    raise ValueError(f"Node missing type in graph: {graph_id}")
                    
                # Check node has valid connections
                if not node.inputs and not node.outputs:
                    raise ValueError(f"Node has no connections in graph: {graph_id}")
                    
            # Validate edges
            for edge in active_graph.edges:
                # Check edge has valid source and target
                if not edge.source or not edge.target:
                    raise ValueError(f"Edge missing source/target in graph: {graph_id}")
                    
                # Check edge connects existing nodes
                source_exists = any(n.id == edge.source for n in active_graph.nodes)
                target_exists = any(n.id == edge.target for n in active_graph.nodes)
                
                if not source_exists or not target_exists:
                    raise ValueError(f"Edge connects non-existent nodes in graph: {graph_id}")
                    
            return True
            
        except Exception as e:
            error_msg = f"Failed to validate graph {graph_id}: {str(e)}"
            self.logger.error(error_msg)
            return False


# Export SupervisorAgent
__all__ = ["SupervisorAgent", "TaskPriority"]
