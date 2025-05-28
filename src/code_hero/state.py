"""State management models and enums for the backend system.

This module defines all state-related models and enums used throughout the system,
including agent states, task states, workflow states, and more.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, validator

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


class AgentType(str, Enum):
    """Agent types for frontend compatibility."""

    RESEARCH = "research"
    CODING = "coding"
    STRATEGIC = "strategic"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    LLAMAINDEX = "llamaindex"
    FASTAPI = "fastapi"
    NEXTJS = "nextjs"
    PYDANTIC = "pydantic"
    AGNO = "agno"
    CREWAI = "crewai"
    SUPERVISOR = "supervisor"
    PROMPT_ENGINEER = "prompt_engineer"
    IMPLEMENTATION = "implementation"
    DOCUMENTATION = "documentation"
    TRD_CONVERTER = "trd_converter"
    CODE_GENERATOR = "code_generator"
    CODE_REVIEWER = "code_reviewer"
    STANDARDS_ENFORCER = "standards_enforcer"
    DOCUMENT_ANALYZER = "document_analyzer"


class AgentStatus(str, Enum):
    """Agent status for frontend compatibility."""

    ACTIVE = "active"
    PROCESSING = "processing"
    IDLE = "idle"
    ERROR = "error"


# ─────────────────────────────────────────────────────────────────────────────
# BASE STATE
# ─────────────────────────────────────────────────────────────────────────────


class BaseState(BaseModel):
    """Base state model with common fields."""

    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)

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
    parent_run_id: Optional[str] = Field(
        default=None, description="ID of the parent run"
    )
    agent: AgentRole
    status: Status = Status.PENDING
    context: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    handlers: Dict[str, Any] = Field(
        default_factory=dict, description="Event handlers for this agent"
    )
    inheritable_handlers: Dict[str, Any] = Field(
        default_factory=dict,
        description="Handlers that can be inherited by child agents",
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorizing and filtering agents"
    )
    error: Optional[str] = None
    error_count: int = Field(
        default=0, description="Number of errors encountered during execution"
    )

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    @validator("context", "artifacts", "handlers", "inheritable_handlers", pre=True)
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

    @validator("tags", pre=True)
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


class AgentPerformance(BaseModel):
    """Agent performance metrics."""

    tasks_completed: int = Field(default=0, description="Number of tasks completed")
    success_rate: float = Field(default=100.0, description="Success rate percentage")
    avg_response_time: str = Field(default="0.5s", description="Average response time")
    uptime: str = Field(default="100%", description="Uptime percentage")


class AgentInfoExtended(BaseModel):
    """Extended agent information for frontend display."""

    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent display name")
    type: AgentType = Field(..., description="Agent type")
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Current status")
    description: str = Field(..., description="Agent description")
    capabilities: List[str] = Field(
        default_factory=list, description="Agent capabilities"
    )
    tools: List[str] = Field(
        default_factory=list, description="Available tools for this agent"
    )
    team: Optional[str] = Field(default=None, description="Team this agent belongs to")
    performance: AgentPerformance = Field(
        default_factory=AgentPerformance, description="Performance metrics"
    )
    current_task: Optional[str] = Field(
        default=None, description="Current task description"
    )
    last_active: datetime = Field(
        default_factory=datetime.utcnow, description="Last active timestamp"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR STATE
# ─────────────────────────────────────────────────────────────────────────────


class SupervisorState(BaseModel):
    """State model for the supervisor agent."""

    initialized: bool = False
    active_workflows: Set[str] = Field(default_factory=set)
    status: Optional[Status] = None
    error: Optional[str] = None


class NodeExecutionResult(BaseModel):
    """Result of a node execution."""

    node_id: str
    status: Status
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH WORKFLOW STATES
# ─────────────────────────────────────────────────────────────────────────────


class AgentResponse(BaseModel):
    """Structured response from an agent."""

    content: str = Field(..., description="Response content")
    agent_role: str = Field(..., description="Role of the agent")
    confidence: float = Field(default=1.0, description="Confidence in response (0-1)")
    reasoning: Optional[str] = Field(None, description="Reasoning behind the response")
    next_steps: List[str] = Field(
        default_factory=list, description="Suggested next steps"
    )
    tools_used: List[str] = Field(
        default_factory=list, description="Tools used in response"
    )


class CodeGenerationRequest(BaseModel):
    """Request for code generation."""

    language: str = Field(..., description="Programming language")
    framework: Optional[str] = Field(None, description="Framework to use")
    requirements: str = Field(..., description="Code requirements")
    style: str = Field(default="clean", description="Code style preference")


class CodeGenerationResponse(BaseModel):
    """Response from code generation."""

    code: str = Field(..., description="Generated code")
    explanation: str = Field(..., description="Explanation of the code")
    dependencies: List[str] = Field(
        default_factory=list, description="Required dependencies"
    )
    usage_example: Optional[str] = Field(None, description="Usage example")


class ResearchRequest(BaseModel):
    """Request for research."""

    topic: str = Field(..., description="Research topic")
    depth: str = Field(
        default="medium", description="Research depth (shallow, medium, deep)"
    )
    sources: List[str] = Field(default_factory=list, description="Preferred sources")


class ResearchResponse(BaseModel):
    """Response from research."""

    summary: str = Field(..., description="Research summary")
    key_findings: List[str] = Field(default_factory=list, description="Key findings")
    sources_used: List[str] = Field(
        default_factory=list, description="Sources consulted"
    )
    confidence_level: float = Field(default=0.8, description="Confidence in findings")


# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIZED STATES
# ─────────────────────────────────────────────────────────────────────────────


class Message(BaseModel):
    """Message model for chat communication."""

    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Role of the message sender"
    )
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatState(BaseState):
    """Chat conversation state."""

    conversation_id: str = Field(
        ..., description="Unique identifier for the conversation"
    )
    participants: List[AgentRole] = Field(
        default_factory=lambda: [AgentRole.SUPERVISOR]
    )
    messages: List[Message] = Field(default_factory=list)
    active_agent: AgentRole = Field(default=AgentRole.SUPERVISOR)
    status: Status = Field(default=Status.RUNNING)
    context: Dict[str, Any] = Field(default_factory=dict)

    @validator("messages", pre=True)
    def validate_messages(cls, v):
        """Ensure messages are properly formatted."""
        if isinstance(v, list):
            return [Message(**msg) if isinstance(msg, dict) else msg for msg in v]
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
                "result": None,
            }
        },
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
                "current_node": "analysis",
            }
        },
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


# ─────────────────────────────────────────────────────────────────────────────
# TOOL ARGUMENT MODELS (from tools.py)
# ─────────────────────────────────────────────────────────────────────────────


class WebFetchArgs(BaseModel):
    """Arguments for web content fetching."""

    url: str = Field(..., description="URL to fetch content from")


class SearchArgs(BaseModel):
    """Arguments for document search."""

    query: str = Field(..., description="Search query")
    collection: str = Field(
        default="strategy_book", description="Collection to search in"
    )
    limit: int = Field(default=5, description="Maximum number of results to return")


class CodeGenArgs(BaseModel):
    """Arguments for code generation."""

    template: str = Field(..., description="Template name (e.g. 'fastapi', 'nextjs')")
    context: Dict[str, Any] = Field(
        ..., description="Generation context with requirements"
    )


class CodeValidationArgs(BaseModel):
    """Arguments for code validation."""

    code: str = Field(..., description="Code to validate")
    language: str = Field(..., description="Programming language")


class CodeAnalysisArgs(BaseModel):
    """Arguments for code analysis."""

    code: str = Field(..., description="Code to analyze")
    language: str = Field(..., description="Programming language")


class WebSearchArgs(BaseModel):
    """Arguments for web search."""

    query: str = Field(..., description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results")
    include_domains: Optional[List[str]] = Field(
        default=None, description="Domains to include"
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None, description="Domains to exclude"
    )


class REPLArgs(BaseModel):
    """Arguments for Python REPL execution."""

    code: str = Field(..., description="Python code to execute")
    timeout: int = Field(default=30, description="Execution timeout in seconds")
    capture_output: bool = Field(
        default=True, description="Whether to capture stdout/stderr"
    )


class FileOperationArgs(BaseModel):
    """Arguments for file operations."""

    file_path: str = Field(..., description="Path to the file")
    content: Optional[str] = Field(
        default=None, description="Content to write (for write operations)"
    )
    encoding: str = Field(default="utf-8", description="File encoding")


class DocumentManagementArgs(BaseModel):
    """Arguments for document management operations."""

    document_id: Optional[str] = Field(default=None, description="Document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    format: str = Field(
        default="markdown", description="Document format (markdown, html, text)"
    )
    tags: List[str] = Field(default_factory=list, description="Document tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN LOOP MODELS (from human_loop.py)
# ─────────────────────────────────────────────────────────────────────────────


class HumanFeedbackRequest(BaseModel):
    """Model for human feedback requests."""

    project_id: str
    task_id: str
    reason: str
    context: Dict[str, Any] = {}
    deadline: Optional[datetime] = None
    priority: str = "normal"


class HumanFeedbackResponse(BaseModel):
    """Model for human feedback responses."""

    request_id: str
    feedback: str
    approved: bool
    comments: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# NODE MODELS (from node.py)
# ─────────────────────────────────────────────────────────────────────────────


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

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolNode(BaseModel):
    """Node that executes a specific tool."""

    name: str
    node_type: NodeType = NodeType.TOOL
    tool_name: str


class AgentNode(BaseModel):
    """Node that executes an agent."""

    name: str
    node_type: NodeType = NodeType.AGENT
    agent_role: str


class HumanLoopNode(BaseModel):
    """Node that handles human-in-the-loop operations."""

    name: str
    node_type: NodeType = NodeType.HUMAN_LOOP
    review_type: str


# ─────────────────────────────────────────────────────────────────────────────
# SERVICE MODELS (from interfaces.py)
# ─────────────────────────────────────────────────────────────────────────────


class ServiceStatus(BaseModel):
    """Service health status model."""

    name: str
    status: Status
    details: dict = {}
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION MODELS (from config.py)
# ─────────────────────────────────────────────────────────────────────────────


class APIConfig(BaseModel):
    """API configuration."""

    model_config = ConfigDict(validate_assignment=True)

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = True
    workers: int = 1


class DatabaseConfig(BaseModel):
    """Database configuration."""

    model_config = ConfigDict(validate_assignment=True)

    astra_db_id: str = Field(default_factory=lambda: os.getenv("ASTRA_DB_ID", ""))
    astra_db_region: str = Field(
        default_factory=lambda: os.getenv("ASTRA_DB_REGION", "")
    )
    astra_db_token: str = Field(
        default_factory=lambda: os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
    )
    collections: Dict[str, Any] = Field(default_factory=dict)


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""

    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.7
    max_tokens: int = 4096


class DeepseekConfig(BaseModel):
    """Deepseek configuration."""

    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096


class GroqConfig(BaseModel):
    """Groq configuration."""

    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = "mixtral-8x7b-32768"
    temperature: float = 0.7
    max_tokens: int = 4096


class AstraDBCollection(BaseModel):
    """AstraDB collection configuration."""

    model_config = ConfigDict(validate_assignment=True)

    name: str
    keyspace: str = "default_keyspace"
    dimensions: int = 1536
    similarity_metric: str = "cosine"
    description: Optional[str] = None


class AstraConfig(BaseModel):
    """AstraDB configuration."""

    model_config = ConfigDict(validate_assignment=True)

    token: str = Field(
        default_factory=lambda: os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
    )
    api_endpoint: str = Field(
        default_factory=lambda: f"https://{os.getenv('ASTRA_DB_ID')}-{os.getenv('ASTRA_DB_REGION')}.apps.astra.datastax.com"
    )
    region: str = Field(default_factory=lambda: os.getenv("ASTRA_DB_REGION", ""))
    database_id: str = Field(default_factory=lambda: os.getenv("ASTRA_DB_ID", ""))
    keyspace: str = "default_keyspace"
    collections: Dict[str, AstraDBCollection] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """Base LLM configuration."""

    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096


class LLMRegistry(BaseModel):
    """Registry for all available LLM configurations."""

    model_config = ConfigDict(validate_assignment=True)

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    deepseek: Optional[DeepseekConfig] = Field(default=None)
    groq: Optional[GroqConfig] = Field(default=None)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(validate_assignment=True)

    level: str = Field(default_factory=lambda: os.getenv("APP_LOG_LEVEL", "INFO"))
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None


class AgentConfig(BaseModel):
    """Agent configuration."""

    model_config = ConfigDict(validate_assignment=True)

    name: str
    role: AgentRole
    system_prompt: str
    tools: List[str] = Field(default_factory=list)
    collections: List[str] = Field(default_factory=list)


class TeamConfig(BaseModel):
    """Team configuration."""

    model_config = ConfigDict(validate_assignment=True)

    name: str
    agents: List[AgentConfig]
    description: Optional[str] = None


class WorkflowConfig(BaseModel):
    """Workflow configuration."""

    model_config = ConfigDict(validate_assignment=True)

    task_timeout_seconds: int = 300  # 5 minutes
    max_retries: int = 3
    retry_delay_seconds: int = 60
    escalation_threshold: int = 2


class ConfigState(BaseState):
    """Configuration state."""

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "api": {"host": "0.0.0.0", "port": 8000},
                "database": {"astra_db_id": "example-id"},
                "llm_registry": {},
                "logging": {"level": "INFO"},
                "workflow": {},
                "environment": "development",
                "status": "idle",
            }
        },
    )

    api: APIConfig
    database: DatabaseConfig
    llm_registry: LLMRegistry = Field(default_factory=LLMRegistry)
    logging: LoggingConfig
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    environment: str = Field(
        default_factory=lambda: os.getenv("APP_ENV", "development")
    )
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    status: Status = Status.IDLE


# ─────────────────────────────────────────────────────────────────────────────
# TYPE ALIASES AND UNIONS
# ─────────────────────────────────────────────────────────────────────────────

# State type union for generic state handling
StateType = Union[
    BaseState,
    AgentState,
    ChatState,
    DocumentState,
    CodeState,
    TRDState,
    HumanLoopState,
    GraphState,
    ProjectState,
    ToolState,
    DatabaseState,
    TaskState,
    WorkflowState,
    ConfigState,
]

# Export all symbols
__all__ = [
    # Enums
    "Status",
    "AgentRole",
    "TaskPriority",
    "ReviewAction",
    "AgentType",
    "AgentStatus",
    "NodeType",
    # Base Models
    "BaseState",
    "BaseNode",
    # Core State Models
    "AgentState",
    "ChatState",
    "DocumentState",
    "CodeState",
    "TRDState",
    "HumanLoopState",
    "GraphState",
    "ProjectState",
    "ToolState",
    "DatabaseState",
    "TaskState",
    "WorkflowState",
    # Agent Models
    "AgentInfo",
    "AgentInfoExtended",
    "AgentPerformance",
    "SupervisorState",
    "NodeExecutionResult",
    "AgentResponse",
    # Request/Response Models
    "CodeGenerationRequest",
    "CodeGenerationResponse",
    "ResearchRequest",
    "ResearchResponse",
    "HumanFeedbackRequest",
    "HumanFeedbackResponse",
    # Tool Models
    "WebFetchArgs",
    "SearchArgs",
    "CodeGenArgs",
    "CodeValidationArgs",
    "CodeAnalysisArgs",
    "WebSearchArgs",
    "REPLArgs",
    "FileOperationArgs",
    "DocumentManagementArgs",
    # Node Models
    "ToolNode",
    "AgentNode",
    "HumanLoopNode",
    # Service Models
    "ServiceStatus",
    # Configuration Models
    "APIConfig",
    "DatabaseConfig",
    "OpenAIConfig",
    "DeepseekConfig",
    "GroqConfig",
    "AstraDBCollection",
    "AstraConfig",
    "LLMConfig",
    "LLMRegistry",
    "LoggingConfig",
    "AgentConfig",
    "TeamConfig",
    "WorkflowConfig",
    "ConfigState",
    # Utility Models
    "Task",
    "Workflow",
    "Message",
    # Type Aliases
    "StateType",
]
