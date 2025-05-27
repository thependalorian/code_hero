"""State management models and enums for the backend system.

This module defines all state-related models and enums used throughout the system,
including agent states, task states, workflow states, and more.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import json
from pydantic import validator

# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────


class Status(str, Enum):
    """Status states for various components."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class AgentRole(str, Enum):
    """Available agent roles in the system."""

    SUPERVISOR = "supervisor"
    RESEARCH = "research"
    IMPLEMENTATION = "implementation"
    DOCUMENTATION = "documentation"
    TRD_CONVERTER = "trd_converter"
    CODE_GENERATOR = "code_generator"
    CODE_REVIEWER = "code_reviewer"
    STANDARDS_ENFORCER = "standards_enforcer"
    STRATEGIC_EXPERT = "strategic_expert"
    LANGCHAIN_EXPERT = "langchain_expert"
    LANGGRAPH_EXPERT = "langgraph_expert"
    LLAMAINDEX_EXPERT = "llamaindex_expert"
    FASTAPI_EXPERT = "fastapi_expert"
    NEXTJS_EXPERT = "nextjs_expert"
    PYDANTIC_EXPERT = "pydantic_expert"
    AGNO_EXPERT = "agno_expert"
    CREWAI_EXPERT = "crewai_expert"
    DOCUMENT_ANALYZER = "document_analyzer"
    PROMPT_ENGINEER = "prompt_engineer"


class TaskPriority(int, Enum):
    """Task priority levels."""

    LOW = 1
    MEDIUM = 5
    HIGH = 10


class ReviewAction(str, Enum):
    """Available review actions."""

    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    FEEDBACK = "feedback"


# ─────────────────────────────────────────────────────────────────────────────
# BASE STATE
# ─────────────────────────────────────────────────────────────────────────────


class BaseState(BaseModel):
    """Base state model with common fields."""

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True
    )

    id: str = Field(..., description="Unique identifier for this state")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# TASK & WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────


class Task(BaseModel):
    """Task model for workflow nodes."""

    task_id: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = Field(default_factory=list)
    status: Status = Status.PENDING
    assigned_to: Optional[AgentRole] = None
    deadline: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    reassignment_count: int = 0


class Workflow(BaseModel):
    """Workflow model for graph execution."""

    workflow_id: str
    name: str
    nodes: List[str] = Field(default_factory=list)
    edges: List[Tuple[str, str]] = Field(default_factory=list)  # (from, to)
    current_node: Optional[str] = None
    history: List[str] = Field(default_factory=list)
    status: Status = Status.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# AGENT STATE
# ─────────────────────────────────────────────────────────────────────────────


class AgentInfo(BaseModel):
    """Static description of each agent."""

    name: AgentRole
    description: str
    persona: str
    tools: List[str] = Field(default_factory=list)
    expertise: List[str] = Field(default_factory=list)
    collections: List[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """Agent state during execution."""

    id: str = Field(default_factory=lambda: f"agent_{datetime.utcnow().timestamp()}")
    parent_run_id: Optional[str] = Field(default=None, description="ID of the parent run")
    agent: AgentRole
    status: Status = Status.PENDING
    context: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    handlers: Dict[str, Any] = Field(default_factory=dict, description="Event handlers for this agent")
    inheritable_handlers: Dict[str, Any] = Field(default_factory=dict, description="Handlers that can be inherited by child agents")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing and filtering agents")
    error: Optional[str] = None
    error_count: int = Field(default=0, description="Number of errors encountered during execution")
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True
        
    @validator('context', 'artifacts', 'handlers', 'inheritable_handlers', pre=True)
    def ensure_dict(cls, v):
        """Ensure context and artifacts are dictionaries."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return {"value": v}
        return {"value": str(v)}
        
    @validator('tags', pre=True)
    def ensure_list(cls, v):
        """Ensure tags is a list."""
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return [v]
        return [str(v)]


# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIZED STATES
# ─────────────────────────────────────────────────────────────────────────────


class Message(BaseModel):
    """Message model for chat communication."""
    role: Literal["user", "assistant", "system"] = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatState(BaseState):
    """Chat conversation state."""
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    participants: List[AgentRole] = Field(default_factory=lambda: [AgentRole.SUPERVISOR])
    messages: List[Message] = Field(default_factory=list)
    active_agent: AgentRole = Field(default=AgentRole.SUPERVISOR)
    status: Status = Field(default=Status.RUNNING)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("messages", pre=True)
    def validate_messages(cls, v):
        """Ensure messages are properly formatted."""
        if isinstance(v, list):
            return [
                Message(**msg) if isinstance(msg, dict) else msg
                for msg in v
            ]
        return []


class DocumentState(AgentState):
    """Document processing state."""

    document_id: str
    content: str
    format: str
    processing_status: Status = Status.PENDING
    review_status: Optional[Status] = None
    review_feedback: Optional[str] = None
    version: str = "1.0.0"


class CodeState(AgentState):
    """Code generation and review state."""

    code_blocks: List[str] = Field(default_factory=list)
    language: str
    lint_results: Optional[Dict[str, Any]] = None
    test_results: Optional[Dict[str, Any]] = None
    review_status: Status = Status.PENDING
    standards_compliance: Dict[str, Any] = Field(default_factory=dict)
    review_feedback: Optional[str] = None
    review_addressed: bool = False
    review_passed: bool = False
    standards_checked: bool = False
    standards_violations: List[str] = Field(default_factory=list)
    standards_passed: bool = False


class TRDState(DocumentState):
    """Technical Requirements Document state."""

    requirements: List[Dict[str, Any]] = Field(default_factory=list)
    specs: List[Dict[str, Any]] = Field(default_factory=list)
    validation_status: Status = Status.PENDING
    dependencies: List[str] = Field(default_factory=list)
    technical_requirements: Optional[Dict[str, Any]] = None
    market_position: Optional[Dict[str, Any]] = None
    strategic_insights: List[Dict[str, Any]] = Field(default_factory=list)


class HumanLoopState(BaseState):
    """Human-in-the-loop review state."""

    review_id: str
    review_type: ReviewAction
    content: Any
    feedback: Optional[str] = None
    status: Status = Status.PENDING
    deadline: Optional[datetime] = None
    reviewer: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH STATE
# ─────────────────────────────────────────────────────────────────────────────


class GraphState(BaseState):
    """Graph execution state."""

    graphs: Dict[str, Workflow] = Field(default_factory=dict)
    active_graph: Optional[str] = None
    execution_history: List[str] = Field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: Status = Status.PENDING


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT STATE
# ─────────────────────────────────────────────────────────────────────────────


class ProjectState(BaseState):
    """Overall project state."""

    project_name: str
    description: Optional[str]
    agents: Dict[AgentRole, AgentInfo] = Field(default_factory=dict)
    team_state: Dict[AgentRole, AgentState] = Field(default_factory=dict)
    graph_state: GraphState
    status: Status = Status.PENDING
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    current_phase: str = "initialization"
    completed_phases: List[str] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL STATE
# ─────────────────────────────────────────────────────────────────────────────


class ToolState(BaseState):
    """Tool execution state."""

    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    execution_status: Status = Status.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY TYPES
# ─────────────────────────────────────────────────────────────────────────────

StateType = Union[
    AgentState,
    ChatState,
    DocumentState,
    CodeState,
    TRDState,
    GraphState,
    ProjectState,
    ToolState,
    HumanLoopState,
]


class DatabaseState(BaseState):
    """Database operation state."""

    collection_name: str
    operation: str = "unknown"
    affected_records: int = 0
    status: Status = Status.PENDING
    error: Optional[str] = None


class TaskState(BaseState):
    """Task state model."""

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "task_id": "t1",
                "description": "Research frameworks",
                "priority": "MEDIUM",
                "status": "pending",
                "assigned_to": "research",
                "result": None
            }
        }
    )

    task_id: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = Field(default_factory=list)
    status: Status = Status.PENDING
    assigned_to: Optional[AgentRole] = None
    deadline: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    reassignment_count: int = 0


class WorkflowState(BaseState):
    """Workflow execution state model.
    
    This model represents the state of a workflow during execution,
    tracking nodes, edges, history, and execution status.
    """
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "workflow_id": "wf1",
                "name": "Research Pipeline",
                "nodes": ["research", "analysis", "report"],
                "status": "running",
                "current_node": "analysis"
            }
        }
    )
    
    workflow_id: str
    name: str
    description: Optional[str] = None
    nodes: List[str] = Field(default_factory=list)
    edges: List[Tuple[str, str]] = Field(default_factory=list)  # (from, to)
    current_node: Optional[str] = None
    history: List[str] = Field(default_factory=list)
    status: Status = Status.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


# Export all symbols
__all__ = [
    # Enums
    "Status",
    "AgentRole",
    "TaskPriority",
    "ReviewAction",
    # Base Models
    "BaseState",
    "Task",
    "TaskState",
    "Workflow",
    "WorkflowState",
    # Agent Models
    "AgentInfo",
    "AgentState",
    # Specialized States
    "ChatState",
    "DocumentState",
    "CodeState",
    "TRDState",
    "HumanLoopState",
    "DatabaseState",
    # Graph and Project
    "GraphState",
    "ProjectState",
    # Tool State
    "ToolState",
    # Types
    "StateType",
]
