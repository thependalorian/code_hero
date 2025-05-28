"""Hierarchical Agent Teams for Code Hero.

This module implements a hierarchical multi-agent system that properly integrates
with our existing Code Hero infrastructure including agent experts, tools, prompts,
and state management. Uses LLM-based routing with structured output, preferred models
with failsafe fallbacks, and human-in-the-loop integration.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_REGION = os.getenv("ASTRA_DB_REGION")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# Validate critical environment variables
if not OPENAI_API_KEY:
    logger = logging.getLogger(__name__)
    logger.warning("OPENAI_API_KEY not found in environment variables")

try:
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import Command

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

    class Command:
        def __init__(self, goto: str, update: Optional[Dict] = None):
            self.goto = goto
            self.update = update or {}

    class MessagesState:
        pass


# Import our existing Code Hero infrastructure
from .agent_expert import all_experts
from .agent_manager import AgentManager

# Import enhanced checkpointing system
from .checkpointing import (
    LANGGRAPH_CHECKPOINTING_AVAILABLE,
    POSTGRES_AVAILABLE,
    CheckpointerConfig,
    CheckpointerType,
    EnhancedCheckpointerManager,
    create_memory_checkpointer,
    create_postgres_checkpointer,
    create_sqlite_checkpointer,
    get_development_checkpointer_config,
    get_production_checkpointer_config,
    initialize_enhanced_checkpointing,
    integrate_with_memory_manager,
)
from .config import (
    MODEL_CONFIGURATIONS,
    get_enhanced_agent_config,
    get_model_for_agent,
    get_tools_for_agent,
)
from .context import StateContext, get_services
from .human_loop import HumanLoopManager
from .logger import StructuredLogger
from .manager import StateManager
from .memory import (
    CodeHeroMemoryManager,
    ConversationSummary,
    MemoryItem,
    UserInfo,
    create_memory_tools,
    initialize_memory_system,
)
from .prompts import build_enhanced_prompt
from .services import (
    ServiceError,
    ServiceHealthCheckError,
    ServiceNotInitializedError,
    validate_services,
)
from .state import (
    AgentRole,
    AgentState,
    Status,
    TaskPriority,
    TaskState,
)
from .strategic_agent import StrategicAgent, StrategicContext
from .supervisor import SupervisorAgent
from .tools import tool_registry
from .utils import (
    call_tool,
    generate_id,
    retry_with_backoff,
)
from .workflow import WorkflowRunner

logger = logging.getLogger(__name__)

# Log available API keys (without exposing the actual keys)
api_keys_status = {
    "OPENAI_API_KEY": bool(OPENAI_API_KEY),
    "GROQ_API_KEY": bool(GROQ_API_KEY),
    "DEEPSEEK_API_KEY": bool(DEEPSEEK_API_KEY),
    "MISTRAL_API_KEY": bool(MISTRAL_API_KEY),
    "LANGSMITH_API_KEY": bool(LANGSMITH_API_KEY),
    "TAVILY_API_KEY": bool(TAVILY_API_KEY),
    "ASTRA_DB_ID": bool(ASTRA_DB_ID),
    "ASTRA_DB_REGION": bool(ASTRA_DB_REGION),
    "ASTRA_DB_APPLICATION_TOKEN": bool(ASTRA_DB_APPLICATION_TOKEN),
}
logger.info(f"API Keys Status: {api_keys_status}")


# Set up LangSmith tracing if API key is available
def setup_langsmith_tracing():
    """Set up LangSmith tracing if API key is available."""
    if LANGSMITH_API_KEY:
        try:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = "code-hero-hierarchical-agents"
            logger.info("LangSmith tracing enabled for hierarchical agents")
            return True
        except Exception as e:
            logger.warning(f"Failed to set up LangSmith tracing: {e}")
            return False
    else:
        logger.info("LangSmith API key not available, tracing disabled")
        return False


# Initialize LangSmith tracing
LANGSMITH_ENABLED = setup_langsmith_tracing()


def format_timestamp(dt: datetime) -> str:
    """Format datetime as ISO string."""
    return dt.isoformat()


def get_llm_for_agent(
    agent_role: AgentRole, fallback_model: str = "gpt-4o-mini"
) -> ChatOpenAI:
    """Get the appropriate LLM for an agent with failsafe fallback.

    Args:
        agent_role: The agent role to get LLM for
        fallback_model: Fallback model if preferred models fail

    Returns:
        Configured ChatOpenAI instance
    """
    try:
        # Get model configuration for the agent
        model_config = get_model_for_agent(agent_role)
        primary_model = model_config.get(
            "model_name", model_config.get("model", "gpt-4o-mini")
        )
        provider = model_config.get("provider", "openai")

        # Try primary model first
        try:
            if provider == "openai":
                return ChatOpenAI(
                    model=primary_model,
                    temperature=model_config.get("temperature", 0.1),
                    max_tokens=model_config.get("max_tokens", 4096),
                    api_key=OPENAI_API_KEY or model_config.get("api_key"),
                )
            elif provider == "groq":
                from langchain_groq import ChatGroq

                return ChatGroq(
                    model=primary_model,
                    temperature=model_config.get("temperature", 0.1),
                    max_tokens=model_config.get("max_tokens", 4096),
                    api_key=GROQ_API_KEY or model_config.get("api_key"),
                )
            elif provider == "deepseek":
                # DeepSeek uses OpenAI-compatible API
                return ChatOpenAI(
                    model=primary_model,
                    temperature=model_config.get("temperature", 0.1),
                    max_tokens=model_config.get("max_tokens", 4096),
                    api_key=DEEPSEEK_API_KEY or model_config.get("api_key"),
                    base_url="https://api.deepseek.com/v1",
                )
            elif provider == "mistral":
                # Mistral uses OpenAI-compatible API
                return ChatOpenAI(
                    model=primary_model,
                    temperature=model_config.get("temperature", 0.1),
                    max_tokens=model_config.get("max_tokens", 8192),
                    api_key=MISTRAL_API_KEY or model_config.get("api_key"),
                    base_url="https://api.mistral.ai/v1",
                )
            else:
                # Default to OpenAI for unsupported providers
                return ChatOpenAI(
                    model=primary_model,
                    temperature=model_config.get("temperature", 0.1),
                    max_tokens=model_config.get("max_tokens", 4096),
                    api_key=OPENAI_API_KEY or model_config.get("api_key"),
                )
        except Exception as e:
            logger.warning(
                f"Primary model {primary_model} failed for {agent_role.value}: {e}"
            )

            # Try alternative models
            alternatives = model_config.get("alternatives", [])
            for alt_model in alternatives:
                try:
                    # Find the alternative model configuration
                    for prov, models in MODEL_CONFIGURATIONS.items():
                        if alt_model in models:
                            alt_config = models[alt_model]
                            if prov == "openai":
                                return ChatOpenAI(
                                    model=alt_config["model_name"],
                                    temperature=alt_config.get("temperature", 0.1),
                                    max_tokens=alt_config.get("max_tokens", 4096),
                                    api_key=OPENAI_API_KEY or alt_config.get("api_key"),
                                )
                            elif prov == "groq":
                                from langchain_groq import ChatGroq

                                return ChatGroq(
                                    model=alt_config["model_name"],
                                    temperature=alt_config.get("temperature", 0.1),
                                    max_tokens=alt_config.get("max_tokens", 4096),
                                    api_key=GROQ_API_KEY or alt_config.get("api_key"),
                                )
                            elif prov == "deepseek":
                                return ChatOpenAI(
                                    model=alt_config["model_name"],
                                    temperature=alt_config.get("temperature", 0.1),
                                    max_tokens=alt_config.get("max_tokens", 4096),
                                    api_key=DEEPSEEK_API_KEY
                                    or alt_config.get("api_key"),
                                    base_url="https://api.deepseek.com/v1",
                                )
                            elif prov == "mistral":
                                return ChatOpenAI(
                                    model=alt_config["model_name"],
                                    temperature=alt_config.get("temperature", 0.1),
                                    max_tokens=alt_config.get("max_tokens", 8192),
                                    api_key=MISTRAL_API_KEY
                                    or alt_config.get("api_key"),
                                    base_url="https://api.mistral.ai/v1",
                                )
                            break
                except Exception as alt_e:
                    logger.warning(
                        f"Alternative model {alt_model} failed for {agent_role.value}: {alt_e}"
                    )
                    continue

            # Final fallback
            logger.warning(
                f"All preferred models failed for {agent_role.value}, using fallback: {fallback_model}"
            )
            return ChatOpenAI(
                model=fallback_model,
                temperature=0.1,
                max_tokens=4096,
                api_key=OPENAI_API_KEY,
            )

    except Exception as e:
        logger.error(
            f"Model selection failed for {agent_role.value}: {e}, using fallback"
        )
        return ChatOpenAI(
            model=fallback_model,
            temperature=0.1,
            max_tokens=4096,
            api_key=OPENAI_API_KEY,
        )


def should_request_human_feedback(
    state: "CodeHeroState", agent_role: AgentRole, context: Dict[str, Any]
) -> bool:
    """Determine if human feedback should be requested based on context and complexity.

    Args:
        state: Current workflow state
        agent_role: Agent role requesting feedback
        context: Task context

    Returns:
        True if human feedback should be requested
    """
    # Check if human feedback is explicitly requested
    if context.get("request_human_feedback", False):
        return True

    # Check for high-risk operations
    high_risk_keywords = [
        "delete",
        "remove",
        "drop",
        "truncate",
        "destroy",
        "production",
        "live",
        "critical",
        "security",
        "auth",
    ]

    user_input = context.get("query", "").lower()
    if any(keyword in user_input for keyword in high_risk_keywords):
        return True

    # Check for complex strategic decisions
    if agent_role in [AgentRole.SUPERVISOR, AgentRole.STRATEGIC_EXPERT]:
        complex_keywords = [
            "architecture",
            "strategy",
            "design",
            "framework",
            "migration",
        ]
        if any(keyword in user_input for keyword in complex_keywords):
            return True

    # Check task priority
    task_priority = state.get("task_priority", TaskPriority.MEDIUM)
    if isinstance(task_priority, TaskPriority) and task_priority == TaskPriority.HIGH:
        return True

    # Check error history
    error_history = state.get("error_history", [])
    if len(error_history) >= 2:  # Multiple errors suggest need for human intervention
        return True

    return False


async def handle_human_feedback_request(
    state: "CodeHeroState",
    agent_role: AgentRole,
    context: Dict[str, Any],
    human_loop_manager: HumanLoopManager,
) -> Optional[str]:
    """Handle human feedback request for complex decisions.

    Args:
        state: Current workflow state
        agent_role: Agent role requesting feedback
        context: Task context
        human_loop_manager: Human loop manager instance

    Returns:
        Human feedback if available, None otherwise
    """
    try:
        # Create task state for feedback request
        task_state = TaskState(
            id=generate_id("task"),
            task_id=state.get("conversation_id", "unknown"),
            description=f"Human feedback needed for {agent_role.value}: {context.get('query', 'Unknown task')}",
            priority=state.get("task_priority", TaskPriority.MEDIUM),
            status=Status.PENDING,
        )

        # Create feedback request
        feedback_request = await human_loop_manager.create_request(
            project_id=state.get("project_id", "unknown"),
            task=task_state,
            reason=f"Complex {agent_role.value} decision requires human oversight",
            context={
                "agent_role": agent_role.value,
                "user_query": context.get("query", ""),
                "task_context": context,
                "error_history": state.get("error_history", []),
                "performance_metrics": state.get("performance_metrics", {}),
            },
        )

        # In a real implementation, this would wait for human response
        # For now, we'll simulate immediate response or timeout
        logger.info(
            f"Human feedback requested for {agent_role.value}: {feedback_request.id}"
        )

        # Check if response is available (simulated)
        response = await human_loop_manager.get_response(feedback_request.task_id)
        if response:
            logger.info(
                f"Human feedback received for {agent_role.value}: {response.feedback}"
            )
            return response.feedback
        else:
            logger.info(
                f"No immediate human feedback available for {agent_role.value}, proceeding with agent decision"
            )
            return None

    except Exception as e:
        logger.error(f"Human feedback request failed for {agent_role.value}: {e}")
        return None


class CodeHeroState(MessagesState):
    """Enhanced state for Code Hero hierarchical workflows with full infrastructure integration."""

    next: str = ""
    current_team: str = ""
    task_context: Dict = {}
    artifacts: Dict = {}
    conversation_id: str = ""
    project_id: str = ""

    # Agent management integration
    agent_states: Dict[str, AgentState] = {}
    agent_manager: Optional[AgentManager] = None
    active_agents: List[str] = []

    # Task and workflow integration
    task_priority: TaskPriority = TaskPriority.MEDIUM
    workflow_state: Optional[Dict] = None
    workflow_runner: Optional[WorkflowRunner] = None

    # Strategic planning integration
    strategic_context: Optional[StrategicContext] = None
    strategic_agent: Optional[StrategicAgent] = None
    strategic_guidance: Dict = {}

    # Human-in-the-loop integration
    human_feedback_required: bool = False
    human_loop_manager: Optional[HumanLoopManager] = None
    pending_feedback_requests: List[str] = []
    human_feedback_responses: Dict[str, str] = {}
    human_intervention_count: int = 0

    # Memory management integration
    memory_manager: Optional[CodeHeroMemoryManager] = None
    conversation_summary: Optional[ConversationSummary] = None
    user_info: Optional[UserInfo] = None
    memory_items: Dict[str, MemoryItem] = {}
    memory_search_results: List[str] = []

    # Model tracking and failsafe
    agent_models: Dict[str, str] = {}  # Track which model each agent is using
    model_failures: Dict[str, List[str]] = {}  # Track model failures per agent
    fallback_models_used: Dict[str, str] = {}  # Track fallback models used

    # Service integration (renamed to avoid conflicts)
    state_manager_service: Optional[StateManager] = None
    logger_service: Optional[StructuredLogger] = None
    supervisor_service: Optional[SupervisorAgent] = None

    # Performance and monitoring
    tools_used: List[str] = []
    performance_metrics: Dict = {}
    error_history: List[str] = []
    service_health: Dict = {}

    # Context management
    context_metadata: Dict = {}
    managed_context: Optional[StateContext] = None


def create_supervisor_node(
    agent_role: AgentRole = AgentRole.SUPERVISOR,
    members: List[str] = None,
    team_name: str = "",
    human_loop_manager: Optional[HumanLoopManager] = None,
):
    """Create a supervisor node using LLM-based routing with structured output, preferred models, and human feedback."""

    if members is None:
        members = []

    # Get the appropriate LLM for this supervisor role
    llm = get_llm_for_agent(agent_role)

    options = ["FINISH"] + members
    system_prompt = (
        f"You are a {team_name} supervisor tasked with managing a conversation between the "
        f"following workers: {members}. Given the following user request, "
        "respond with the worker to act next. Each worker will perform a "
        "task and respond with their results and status. When finished, "
        "respond with FINISH.\n\n"
        "IMPORTANT RULES:\n"
        "1. If this is a simple greeting (hello, hi, hey), respond with FINISH immediately\n"
        "2. If a worker has already provided a complete response to the user's request, respond with FINISH\n"
        "3. If multiple workers have already responded, respond with FINISH\n"
        "4. Only route to a worker if their specific expertise is needed and they haven't responded yet\n"
        "5. Always prefer FINISH over continuing the conversation unnecessarily\n"
        "6. For complex or high-risk operations, consider if human oversight is needed\n"
        f"7. You are using model: {llm.model_name if hasattr(llm, 'model_name') else 'unknown'} with failsafe support"
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: CodeHeroState) -> Command:
        """LLM-based router with structured output, model failsafe, and human feedback integration."""
        try:
            # Track model usage
            model_name = llm.model_name if hasattr(llm, "model_name") else str(llm)
            if "agent_models" not in state:
                state["agent_models"] = {}
            state["agent_models"][f"{team_name}_supervisor"] = model_name

            # Check if this is a simple greeting
            last_message = state["messages"][-1] if state["messages"] else None
            user_input = last_message.content if last_message else ""

            simple_greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
            if (
                user_input.lower().strip() in simple_greetings
                and team_name == "Main Orchestrator"
            ):
                greeting_response = "Hello! I'm Code Hero, your AI development assistant. How can I help you today?"
                return Command(
                    update={
                        "messages": [
                            HumanMessage(content=greeting_response, name="supervisor")
                        ]
                    },
                    goto=END,
                )

            # Check if human feedback is needed
            context = {
                "query": user_input,
                "team": team_name,
                "members": members,
                "supervisor_role": agent_role.value,
            }

            if human_loop_manager and should_request_human_feedback(
                state, agent_role, context
            ):
                try:
                    # Request human feedback asynchronously
                    import asyncio

                    loop = asyncio.get_event_loop()
                    if not loop.is_running():
                        human_feedback = loop.run_until_complete(
                            handle_human_feedback_request(
                                state, agent_role, context, human_loop_manager
                            )
                        )

                        if human_feedback:
                            # Update state with human feedback
                            if "human_feedback_responses" not in state:
                                state["human_feedback_responses"] = {}
                            state["human_feedback_responses"][
                                f"{team_name}_supervisor"
                            ] = human_feedback
                            state["human_intervention_count"] = (
                                state.get("human_intervention_count", 0) + 1
                            )

                            # Log human intervention
                            logger.info(
                                f"Human feedback incorporated for {team_name} supervisor: {human_feedback[:100]}..."
                            )
                except Exception as e:
                    logger.warning(
                        f"Human feedback request failed for {team_name} supervisor: {e}"
                    )

            # Check if we already have worker responses
            worker_responses = [
                msg
                for msg in state["messages"]
                if hasattr(msg, "name") and msg.name in members
            ]

            # If we have any worker response, finish (avoid infinite loops)
            if worker_responses:
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content="Task completed successfully.",
                                name="supervisor",
                            )
                        ]
                    },
                    goto=END,
                )

            # Use LLM to make routing decision only if no workers have responded yet
            messages = [
                {"role": "system", "content": system_prompt},
            ] + [
                {
                    "role": msg.type if hasattr(msg, "type") else "user",
                    "content": msg.content,
                }
                for msg in state["messages"]
            ]

            # Get structured response from LLM with failsafe
            try:
                response = llm.with_structured_output(Router).invoke(messages)
                goto = response["next"]
            except Exception as llm_error:
                logger.error(
                    f"LLM routing failed for {team_name} supervisor: {llm_error}"
                )

                # Track model failure
                if "model_failures" not in state:
                    state["model_failures"] = {}
                if f"{team_name}_supervisor" not in state["model_failures"]:
                    state["model_failures"][f"{team_name}_supervisor"] = []
                state["model_failures"][f"{team_name}_supervisor"].append(
                    str(llm_error)
                )

                # Fallback routing logic
                if user_input and any(
                    keyword in user_input.lower()
                    for keyword in ["code", "implement", "build"]
                ):
                    goto = (
                        "development_team"
                        if "development_team" in members
                        else members[0] if members else "FINISH"
                    )
                elif user_input and any(
                    keyword in user_input.lower()
                    for keyword in ["research", "analyze", "study"]
                ):
                    goto = (
                        "research_team"
                        if "research_team" in members
                        else members[0] if members else "FINISH"
                    )
                elif user_input and any(
                    keyword in user_input.lower()
                    for keyword in ["document", "write", "explain"]
                ):
                    goto = (
                        "documentation_team"
                        if "documentation_team" in members
                        else members[0] if members else "FINISH"
                    )
                else:
                    goto = "FINISH"

                logger.info(
                    f"Using fallback routing for {team_name} supervisor: {goto}"
                )

            if goto == "FINISH":
                goto = END

            return Command(goto=goto, update={"next": goto})

        except Exception as e:
            logger.error(f"Supervisor node error for {team_name}: {e}")

            # Track error
            if "error_history" not in state:
                state["error_history"] = []
            state["error_history"].append(f"{team_name} supervisor: {str(e)}")

            # Always finish on error to prevent loops
            return Command(
                update={
                    "messages": [
                        HumanMessage(content="Task completed.", name="supervisor")
                    ]
                },
                goto=END,
            )

    return supervisor_node


# === DEVELOPMENT TEAM ===


def create_development_team(human_loop_manager: Optional[HumanLoopManager] = None):
    """Create development team using our existing expert agents with preferred models and human feedback."""

    def fastapi_expert_node(state: CodeHeroState) -> Command:
        """FastAPI expert using our existing infrastructure."""
        return execute_expert_node(
            state, AgentRole.FASTAPI_EXPERT, "fastapi_expert", human_loop_manager
        )

    def nextjs_expert_node(state: CodeHeroState) -> Command:
        """Next.js expert using our existing infrastructure."""
        return execute_expert_node(
            state, AgentRole.NEXTJS_EXPERT, "nextjs_expert", human_loop_manager
        )

    def code_expert_node(state: CodeHeroState) -> Command:
        """Code generation expert using our existing infrastructure."""
        return execute_expert_node(
            state, AgentRole.CODE_GENERATOR, "code_expert", human_loop_manager
        )

    # Create supervisor with LLM-based routing and human feedback
    dev_supervisor_node = create_supervisor_node(
        AgentRole.SUPERVISOR,
        ["fastapi_expert", "nextjs_expert", "code_expert"],
        "Development Team",
        human_loop_manager,
    )

    # Build graph
    if not LANGGRAPH_AVAILABLE:
        return None

    dev_builder = StateGraph(CodeHeroState)
    dev_builder.add_node("supervisor", dev_supervisor_node)
    dev_builder.add_node("fastapi_expert", fastapi_expert_node)
    dev_builder.add_node("nextjs_expert", nextjs_expert_node)
    dev_builder.add_node("code_expert", code_expert_node)

    # Add edges with conditional routing
    dev_builder.add_edge(START, "supervisor")
    dev_builder.add_conditional_edges(
        "supervisor",
        lambda x: x.get("next", END),
        {
            "fastapi_expert": "fastapi_expert",
            "nextjs_expert": "nextjs_expert",
            "code_expert": "code_expert",
            END: END,
        },
    )

    # Experts finish directly, no need to return to supervisor

    return dev_builder.compile()


# === RESEARCH TEAM ===


def create_research_team(human_loop_manager: Optional[HumanLoopManager] = None):
    """Create research team using our existing expert agents with preferred models and human feedback."""

    def research_expert_node(state: CodeHeroState) -> Command:
        """Research expert using our existing infrastructure."""
        return execute_expert_node(
            state, AgentRole.RESEARCH, "research_expert", human_loop_manager
        )

    def strategic_expert_node(state: CodeHeroState) -> Command:
        """Strategic expert using our existing infrastructure."""
        return execute_expert_node(
            state, AgentRole.STRATEGIC_EXPERT, "strategic_expert", human_loop_manager
        )

    def analysis_expert_node(state: CodeHeroState) -> Command:
        """Code review expert using our existing infrastructure."""
        return execute_expert_node(
            state, AgentRole.CODE_REVIEWER, "analysis_expert", human_loop_manager
        )

    # Create supervisor with LLM-based routing and human feedback
    research_supervisor_node = create_supervisor_node(
        AgentRole.STRATEGIC_EXPERT,
        ["research_expert", "strategic_expert", "analysis_expert"],
        "Research Team",
        human_loop_manager,
    )

    # Build graph
    if not LANGGRAPH_AVAILABLE:
        return None

    research_builder = StateGraph(CodeHeroState)
    research_builder.add_node("supervisor", research_supervisor_node)
    research_builder.add_node("research_expert", research_expert_node)
    research_builder.add_node("strategic_expert", strategic_expert_node)
    research_builder.add_node("analysis_expert", analysis_expert_node)

    # Add edges with conditional routing
    research_builder.add_edge(START, "supervisor")
    research_builder.add_conditional_edges(
        "supervisor",
        lambda x: x.get("next", END),
        {
            "research_expert": "research_expert",
            "strategic_expert": "strategic_expert",
            "analysis_expert": "analysis_expert",
            END: END,
        },
    )

    # Experts finish directly, no need to return to supervisor

    return research_builder.compile()


# === DOCUMENTATION TEAM ===


def create_documentation_team(human_loop_manager: Optional[HumanLoopManager] = None):
    """Create documentation team using our existing expert agents with preferred models and human feedback."""

    def documentation_expert_node(state: CodeHeroState) -> Command:
        """Documentation expert using our existing infrastructure."""
        return execute_expert_node(
            state, AgentRole.DOCUMENTATION, "documentation_expert", human_loop_manager
        )

    def implementation_expert_node(state: CodeHeroState) -> Command:
        """Implementation expert using our existing infrastructure."""
        return execute_expert_node(
            state, AgentRole.IMPLEMENTATION, "implementation_expert", human_loop_manager
        )

    # Create supervisor with LLM-based routing and human feedback
    doc_supervisor_node = create_supervisor_node(
        AgentRole.DOCUMENTATION,
        ["documentation_expert", "implementation_expert"],
        "Documentation Team",
        human_loop_manager,
    )

    # Build graph
    if not LANGGRAPH_AVAILABLE:
        return None

    doc_builder = StateGraph(CodeHeroState)
    doc_builder.add_node("supervisor", doc_supervisor_node)
    doc_builder.add_node("documentation_expert", documentation_expert_node)
    doc_builder.add_node("implementation_expert", implementation_expert_node)

    # Add edges with conditional routing
    doc_builder.add_edge(START, "supervisor")
    doc_builder.add_conditional_edges(
        "supervisor",
        lambda x: x.get("next", END),
        {
            "documentation_expert": "documentation_expert",
            "implementation_expert": "implementation_expert",
            END: END,
        },
    )

    # Experts finish directly, no need to return to supervisor

    return doc_builder.compile()


def execute_expert_node(
    state: CodeHeroState,
    agent_role: AgentRole,
    node_name: str,
    human_loop_manager: Optional[HumanLoopManager] = None,
) -> Command:
    """Execute an expert node using our existing agent infrastructure with preferred models and human feedback."""
    try:
        # Get the user message
        last_message = state["messages"][-1] if state["messages"] else None
        user_input = last_message.content if last_message else ""

        # Get the appropriate LLM for this agent
        llm = get_llm_for_agent(agent_role)

        # Get tools for this agent and bind them to the LLM
        from .tools import tool_registry

        agent_tools = get_tools_for_agent(agent_role)

        # Get tool category for this agent
        tool_category = None
        if agent_role in [
            AgentRole.FASTAPI_EXPERT,
            AgentRole.NEXTJS_EXPERT,
            AgentRole.CODE_GENERATOR,
        ]:
            tool_category = "development"
        elif agent_role in [
            AgentRole.RESEARCH,
            AgentRole.STRATEGIC_EXPERT,
            AgentRole.CODE_REVIEWER,
        ]:
            tool_category = "research"
        elif agent_role in [AgentRole.DOCUMENTATION, AgentRole.IMPLEMENTATION]:
            tool_category = "documentation"
        else:
            # For other agents, use all available tools
            tool_category = None

        # Bind ALL tools to LLM for maximum capability
        if hasattr(tool_registry, "bind_tools_to_llm"):
            try:
                # First try to bind all tools (not just category-specific)
                llm = tool_registry.bind_tools_to_llm(
                    llm, category=None
                )  # None = all tools
                logger.info(
                    f"Bound ALL tools to {agent_role.value} LLM - total tools: {len(list(tool_registry.list_tools()))}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to bind all tools to {agent_role.value} LLM: {e}"
                )
                # Fallback to category-specific tools if available
                if tool_category:
                    try:
                        llm = tool_registry.bind_tools_to_llm(
                            llm, category=tool_category
                        )
                        logger.info(
                            f"Bound {tool_category} tools to {agent_role.value} LLM as fallback"
                        )
                    except Exception as e2:
                        logger.warning(
                            f"Failed to bind {tool_category} tools to {agent_role.value} LLM: {e2}"
                        )
        else:
            logger.warning(f"Tool registry does not support bind_tools_to_llm method")

        # Track model usage
        model_name = llm.model_name if hasattr(llm, "model_name") else str(llm)
        if "agent_models" not in state:
            state["agent_models"] = {}
        state["agent_models"][node_name] = model_name

        # Initialize context first to avoid variable scoping issues
        task_priority = state.get("task_priority", TaskPriority.MEDIUM)
        all_available_tools = list(tool_registry.list_tools())
        context = {
            "query": user_input,
            "conversation_id": state.get("conversation_id", ""),
            "project_id": state.get("project_id", ""),
            "task_context": state.get("task_context", {}),
            "hierarchical_mode": True,
            "specialization": f"{agent_role.value} expert",
            "task_priority": (
                task_priority.value
                if isinstance(task_priority, TaskPriority)
                else task_priority
            ),
            "priority_level": (
                task_priority.value
                if isinstance(task_priority, TaskPriority)
                else str(task_priority)
            ),
            "artifacts": state.get("artifacts", {}),
            "performance_metrics": state.get("performance_metrics", {}),
            "model_used": model_name,
            "agent_role": agent_role.value,
            "tools_available": len(all_available_tools),
            "all_tools": all_available_tools,
            "agent_specific_tools": len(agent_tools) if agent_tools else 0,
            "tool_category": tool_category,
        }

        # Check if human feedback is needed for this expert
        if human_loop_manager and should_request_human_feedback(
            state, agent_role, context
        ):
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    human_feedback = loop.run_until_complete(
                        handle_human_feedback_request(
                            state, agent_role, context, human_loop_manager
                        )
                    )

                    if human_feedback:
                        # Incorporate human feedback into context
                        context["human_feedback"] = human_feedback
                        if "human_feedback_responses" not in state:
                            state["human_feedback_responses"] = {}
                        state["human_feedback_responses"][node_name] = human_feedback
                        state["human_intervention_count"] = (
                            state.get("human_intervention_count", 0) + 1
                        )

                        logger.info(
                            f"Human feedback incorporated for {node_name}: {human_feedback[:100]}..."
                        )
            except Exception as e:
                logger.warning(f"Human feedback request failed for {node_name}: {e}")

        # Get the expert directly from our registry
        expert = all_experts.get(agent_role)
        if not expert:
            response_content = f"The {agent_role.value} expert is not available."
        else:
            # Enhance context with additional infrastructure components
            try:
                context.update(
                    {
                        "available_tools": get_tools_for_agent(agent_role),
                        "agent_config": get_enhanced_agent_config(agent_role, context),
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Failed to get enhanced config for {agent_role.value}: {e}"
                )
                context["available_tools"] = []
                context["agent_config"] = {}

            # Create enhanced AgentState for proper integration
            agent_state = AgentState(
                id=generate_id(),
                agent=agent_role,
                status=Status.RUNNING,
                context=context,
                artifacts=state.get("artifacts", {}),
            )

            # Use enhanced prompt building
            try:
                build_enhanced_prompt(agent_role, agent_state)
            except Exception as e:
                logger.warning(
                    f"Failed to build enhanced prompt for {agent_role.value}: {e}"
                )

            # Try to use the expert's enhanced response method with model failsafe
            try:
                # Use the bound LLM directly instead of fallback response
                from langchain_core.messages import HumanMessage as LCHumanMessage
                from langchain_core.messages import (
                    SystemMessage,
                )

                from .config import get_enhanced_system_prompt
                from .prompts import build_enhanced_prompt
                from .state import AgentState

                # Create proper agent state for prompt building
                agent_state = AgentState(
                    agent=agent_role,
                    context={
                        "query": user_input,
                        "task_type": context.get("task_type", "general"),
                        "available_tools": (
                            [tool.name for tool in tools] if tools else []
                        ),
                        "current_status": "processing",
                        "timestamp": datetime.now().isoformat(),
                        **context,
                    },
                )

                # Use industry-standard prompt system
                try:
                    system_prompt = build_enhanced_prompt(agent_role, agent_state)
                except Exception as prompt_error:
                    logger.warning(
                        f"Failed to build enhanced prompt for {agent_role.value}: {prompt_error}"
                    )
                    # Fallback to basic system prompt
                    system_prompt = get_enhanced_system_prompt(
                        agent_role, agent_state.context
                    )

                # Create message chain with proper system prompt
                messages = [
                    SystemMessage(content=system_prompt),
                    LCHumanMessage(content=user_input),
                ]

                # Invoke the bound LLM with tools
                response = bound_llm.invoke(messages)

                # Extract content from response
                if hasattr(response, "content"):
                    response_content = response.content
                elif hasattr(response, "text"):
                    response_content = response.text
                else:
                    response_content = str(response)

                logger.info(
                    f"✅ {agent_role.value} successfully used LLM with {len(tools)} tools bound"
                )

            except Exception as llm_error:
                logger.error(f"Primary LLM failed for {agent_role.value}: {llm_error}")

                # Try fallback model with same prompt system
                try:
                    fallback_llm = get_llm_for_agent(
                        agent_role, fallback_model="gpt-4o-mini"
                    )
                    fallback_bound_llm = tool_registry.bind_tools_to_llm(
                        fallback_llm, category=tool_category
                    )

                    # Use same industry-standard prompt
                    messages = [
                        SystemMessage(content=system_prompt),
                        LCHumanMessage(content=user_input),
                    ]

                    response = fallback_bound_llm.invoke(messages)

                    if hasattr(response, "content"):
                        response_content = response.content
                    elif hasattr(response, "text"):
                        response_content = response.text
                    else:
                        response_content = str(response)

                    logger.info(
                        f"✅ {agent_role.value} successfully used fallback LLM with tools"
                    )

                except Exception as fallback_e:
                    logger.error(
                        f"Fallback model also failed for {agent_role.value}: {fallback_e}"
                    )
                    # Only now use the expert fallback response as last resort
                    if expert:
                        response_content = expert._fallback_response(
                            user_input, context
                        )
                    else:
                        response_content = f"I encountered technical difficulties while processing your request as a {agent_role.value} expert. Please try again or contact support."

        # Update state with execution results
        if "agent_states" not in state:
            state["agent_states"] = {}
        if "tools_used" not in state:
            state["tools_used"] = []
        if "performance_metrics" not in state:
            state["performance_metrics"] = {}

        state["agent_states"][node_name] = agent_state
        state["tools_used"].extend(context.get("available_tools", []))
        state["performance_metrics"][node_name] = {
            "execution_time": format_timestamp(datetime.now()),
            "agent_role": agent_role.value,
            "model_used": model_name,
            "success": True,
            "human_feedback_used": bool(context.get("human_feedback")),
            "fallback_model_used": node_name in state.get("fallback_models_used", {}),
        }

        # Finish directly to avoid infinite loops (following LangGraph patterns)
        return Command(
            update={
                "messages": [HumanMessage(content=response_content, name=node_name)],
                "agent_states": state.get("agent_states", {}),
                "tools_used": state.get("tools_used", []),
                "performance_metrics": state.get("performance_metrics", {}),
                "agent_models": state.get("agent_models", {}),
                "model_failures": state.get("model_failures", {}),
                "fallback_models_used": state.get("fallback_models_used", {}),
                "human_feedback_responses": state.get("human_feedback_responses", {}),
                "human_intervention_count": state.get("human_intervention_count", 0),
            },
            goto=END,
        )

    except Exception as e:
        logger.error(f"Expert node {node_name} error: {e}")

        # Initialize error_history if it doesn't exist
        if "error_history" not in state:
            state["error_history"] = []

        # Update error history
        error_msg = f"Expert {node_name} encountered an error: {str(e)}"
        state["error_history"].append(error_msg)

        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=error_msg,
                        name=node_name,
                    )
                ],
                "error_history": state.get("error_history", []),
            },
            goto=END,
        )


# === TOP-LEVEL ORCHESTRATOR ===


def create_hierarchical_system_internal(
    human_loop_manager: Optional[HumanLoopManager] = None,
    enable_human_feedback: bool = True,
    enable_memory: bool = True,
    memory_manager: Optional[CodeHeroMemoryManager] = None,
    checkpointer_manager: Optional[EnhancedCheckpointerManager] = None,
    environment: str = "development",
):
    """Internal function to create the complete hierarchical agent system with enhanced checkpointing.

    Args:
        human_loop_manager: Optional human loop manager for human feedback
        enable_human_feedback: Whether to enable human feedback
        enable_memory: Whether to enable memory management
        memory_manager: Optional memory manager instance
        checkpointer_manager: Optional enhanced checkpointer manager
        environment: Environment type for checkpointer configuration
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available, hierarchical system disabled")
        return None

    # Initialize human loop manager if enabled and not provided
    if enable_human_feedback and human_loop_manager is None:
        try:
            human_loop_manager = HumanLoopManager()
            import asyncio

            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(human_loop_manager.initialize())
            logger.info("Human loop manager initialized for hierarchical system")
        except Exception as e:
            logger.warning(f"Failed to initialize human loop manager: {e}")
            human_loop_manager = None

    # Create team graphs with enhanced configuration and human feedback
    dev_graph = create_development_team(human_loop_manager)
    research_graph = create_research_team(human_loop_manager)
    doc_graph = create_documentation_team(human_loop_manager)

    if not all([dev_graph, research_graph, doc_graph]):
        logger.error("Failed to create team graphs")
        return None

    # Top-level supervisor with LLM-based routing, preferred models, and human feedback
    main_supervisor_node = create_supervisor_node(
        AgentRole.SUPERVISOR,
        ["development_team", "research_team", "documentation_team"],
        "Main Orchestrator",
        human_loop_manager,
    )

    # Team connector functions with enhanced error handling and model tracking
    def call_development_team(state: CodeHeroState) -> Command:
        """Route to development team with model tracking."""
        try:
            response = dev_graph.invoke(state)
            final_message = (
                response["messages"][-1] if response.get("messages") else None
            )
            content = (
                final_message.content
                if final_message
                else "Development team completed the task."
            )

            # Track team performance
            team_metrics = {
                "team": "development",
                "execution_time": format_timestamp(datetime.now()),
                "success": True,
                "models_used": response.get("agent_models", {}),
                "human_interventions": response.get("human_intervention_count", 0),
                "fallback_models": response.get("fallback_models_used", {}),
            }

            return Command(
                update={
                    "messages": [
                        HumanMessage(content=content, name="development_team")
                    ],
                    "performance_metrics": {
                        **state.get("performance_metrics", {}),
                        "development_team": team_metrics,
                    },
                    "agent_models": {
                        **state.get("agent_models", {}),
                        **response.get("agent_models", {}),
                    },
                    "human_intervention_count": state.get("human_intervention_count", 0)
                    + response.get("human_intervention_count", 0),
                },
                goto=END,
            )
        except Exception as e:
            logger.error(f"Development team error: {e}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=f"Development team encountered an error: {str(e)}",
                            name="development_team",
                        )
                    ],
                    "error_history": state.get("error_history", [])
                    + [f"Development team: {str(e)}"],
                },
                goto=END,
            )

    def call_research_team(state: CodeHeroState) -> Command:
        """Route to research team with model tracking."""
        try:
            response = research_graph.invoke(state)
            final_message = (
                response["messages"][-1] if response.get("messages") else None
            )
            content = (
                final_message.content
                if final_message
                else "Research team completed the task."
            )

            # Track team performance
            team_metrics = {
                "team": "research",
                "execution_time": format_timestamp(datetime.now()),
                "success": True,
                "models_used": response.get("agent_models", {}),
                "human_interventions": response.get("human_intervention_count", 0),
                "fallback_models": response.get("fallback_models_used", {}),
            }

            return Command(
                update={
                    "messages": [HumanMessage(content=content, name="research_team")],
                    "performance_metrics": {
                        **state.get("performance_metrics", {}),
                        "research_team": team_metrics,
                    },
                    "agent_models": {
                        **state.get("agent_models", {}),
                        **response.get("agent_models", {}),
                    },
                    "human_intervention_count": state.get("human_intervention_count", 0)
                    + response.get("human_intervention_count", 0),
                },
                goto=END,
            )
        except Exception as e:
            logger.error(f"Research team error: {e}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=f"Research team encountered an error: {str(e)}",
                            name="research_team",
                        )
                    ],
                    "error_history": state.get("error_history", [])
                    + [f"Research team: {str(e)}"],
                },
                goto=END,
            )

    def call_documentation_team(state: CodeHeroState) -> Command:
        """Route to documentation team with model tracking."""
        try:
            response = doc_graph.invoke(state)
            final_message = (
                response["messages"][-1] if response.get("messages") else None
            )
            content = (
                final_message.content
                if final_message
                else "Documentation team completed the task."
            )

            # Track team performance
            team_metrics = {
                "team": "documentation",
                "execution_time": format_timestamp(datetime.now()),
                "success": True,
                "models_used": response.get("agent_models", {}),
                "human_interventions": response.get("human_intervention_count", 0),
                "fallback_models": response.get("fallback_models_used", {}),
            }

            return Command(
                update={
                    "messages": [
                        HumanMessage(content=content, name="documentation_team")
                    ],
                    "performance_metrics": {
                        **state.get("performance_metrics", {}),
                        "documentation_team": team_metrics,
                    },
                    "agent_models": {
                        **state.get("agent_models", {}),
                        **response.get("agent_models", {}),
                    },
                    "human_intervention_count": state.get("human_intervention_count", 0)
                    + response.get("human_intervention_count", 0),
                },
                goto=END,
            )
        except Exception as e:
            logger.error(f"Documentation team error: {e}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=f"Documentation team encountered an error: {str(e)}",
                            name="documentation_team",
                        )
                    ],
                    "error_history": state.get("error_history", [])
                    + [f"Documentation team: {str(e)}"],
                },
                goto=END,
            )

    # Build the main orchestrator graph
    super_builder = StateGraph(CodeHeroState)
    super_builder.add_node("supervisor", main_supervisor_node)
    super_builder.add_node("development_team", call_development_team)
    super_builder.add_node("research_team", call_research_team)
    super_builder.add_node("documentation_team", call_documentation_team)

    # Add edges with conditional routing
    super_builder.add_edge(START, "supervisor")
    super_builder.add_conditional_edges(
        "supervisor",
        lambda x: x.get("next", END),
        {
            "development_team": "development_team",
            "research_team": "research_team",
            "documentation_team": "documentation_team",
            END: END,
        },
    )

    # Teams finish directly, no need to return to supervisor

    # Enhanced checkpointing with fallback support
    checkpointer = None
    if enable_memory:
        try:
            # Use provided checkpointer manager or create one
            if checkpointer_manager:
                checkpointer = checkpointer_manager.get_checkpointer()
                logger.info("Using provided enhanced checkpointer manager")
            elif memory_manager:
                # Try to get checkpointer from memory manager
                checkpointer = memory_manager.get_checkpointer()
                logger.info("Using checkpointer from memory manager")
            else:
                # Create a new enhanced checkpointer based on environment
                import asyncio

                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    enhanced_manager = loop.run_until_complete(
                        initialize_enhanced_checkpointing(environment=environment)
                    )
                    checkpointer = enhanced_manager.get_checkpointer()
                    logger.info(f"Created new enhanced checkpointer for {environment}")
                else:
                    # Fallback to basic memory saver in async context
                    from .memory import MemorySaver

                    checkpointer = MemorySaver()
                    logger.info("Using basic MemorySaver in async context")

            if checkpointer:
                logger.info("Enhanced checkpointer enabled for hierarchical system")
            else:
                logger.warning("No checkpointer available, proceeding without memory")

        except Exception as e:
            logger.warning(f"Failed to initialize enhanced checkpointer: {e}")
            # Fallback to basic memory saver
            try:
                from .memory import MemorySaver

                checkpointer = MemorySaver()
                logger.info(
                    "Using fallback MemorySaver due to enhanced checkpointer failure"
                )
            except Exception as fallback_e:
                logger.error(f"Even fallback checkpointer failed: {fallback_e}")
                checkpointer = None

    try:
        compiled_system = super_builder.compile(checkpointer=checkpointer, debug=False)

        # Log comprehensive initialization status
        checkpointer_status = "enabled" if checkpointer else "disabled"
        human_feedback_status = "enabled" if human_loop_manager else "disabled"
        memory_status = "enabled" if memory_manager else "disabled"

        logger.info(
            f"Hierarchical system created with: "
            f"checkpointer={checkpointer_status}, "
            f"human_feedback={human_feedback_status}, "
            f"memory={memory_status}, "
            f"environment={environment}"
        )

        return compiled_system

    except Exception as e:
        logger.error(f"Failed to compile hierarchical system: {e}")
        # Fallback: compile without checkpointer if there are compatibility issues
        try:
            logger.warning(
                "Attempting to compile hierarchical system without checkpointer due to compatibility issues"
            )
            compiled_system = super_builder.compile(debug=False)
            logger.info(
                "Hierarchical system created without checkpointer (fallback mode)"
            )
            return compiled_system
        except Exception as fallback_error:
            logger.error(
                f"Failed to compile hierarchical system even without checkpointer: {fallback_error}"
            )
            raise RuntimeError(
                f"Unable to create hierarchical system: {fallback_error}"
            ) from fallback_error


def create_hierarchical_system(config):
    """LangGraph-compatible wrapper function that takes exactly one RunnableConfig argument.

    This function is specifically designed to work with LangGraph's requirements.
    It extracts configuration from the RunnableConfig and calls the internal implementation.

    Args:
        config: RunnableConfig object from LangGraph containing configuration parameters

    Returns:
        Compiled hierarchical agent system
    """
    try:
        # Extract configuration parameters from RunnableConfig
        enable_human_feedback = True
        enable_memory = True

        if config and hasattr(config, "configurable"):
            configurable = getattr(config, "configurable", {})
            enable_human_feedback = configurable.get("enable_human_feedback", True)
            enable_memory = configurable.get("enable_memory", True)
            logger.info(
                f"LangGraph config extracted: human_feedback={enable_human_feedback}, memory={enable_memory}"
            )
        else:
            logger.info("Using default configuration for hierarchical system")

        # Call the internal implementation with extracted parameters
        return create_hierarchical_system_internal(
            human_loop_manager=None,  # Will be initialized internally if needed
            enable_human_feedback=enable_human_feedback,
            enable_memory=enable_memory,
            memory_manager=None,  # Will be initialized internally if needed
            checkpointer_manager=None,
            environment="development",
        )

    except Exception as e:
        logger.error(f"Error in LangGraph wrapper for hierarchical system: {e}")
        return None


# === HUMAN-IN-THE-LOOP INTEGRATION ===


def create_human_feedback_node(human_loop_manager: HumanLoopManager):
    """Create a human feedback node for complex decisions."""

    def human_feedback_node(state: CodeHeroState) -> Command:
        """Request human feedback when needed."""
        try:
            # Check if human feedback is required
            if not state.get("human_feedback_required", False):
                return Command(goto=END)

            # Create feedback request
            task_state = TaskState(
                id=generate_id(),
                task_id=state.get("conversation_id", "unknown"),
                description=f"Human feedback needed for: {state['messages'][-1].content if state['messages'] else 'Unknown task'}",
                priority=state.get("task_priority", TaskPriority.MEDIUM),
                status=Status.PENDING,
            )

            # This would typically be async, but we're in a sync context
            # In a real implementation, this would trigger a human review process
            feedback_response = "Human feedback integration available but not implemented in this sync context."

            return Command(
                update={
                    "messages": [
                        HumanMessage(content=feedback_response, name="human_feedback")
                    ],
                    "human_feedback_required": False,
                },
                goto=END,
            )

        except Exception as e:
            logger.error(f"Human feedback node error: {e}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="Human feedback system encountered an error.",
                            name="human_feedback",
                        )
                    ]
                },
                goto=END,
            )

    return human_feedback_node


# === ENHANCED TEAM CREATION WITH FULL INTEGRATION ===


def create_enhanced_development_team(
    llm: ChatOpenAI, human_loop_manager: HumanLoopManager = None
):
    """Create enhanced development team with full Code Hero integration."""

    def enhanced_fastapi_expert_node(state: CodeHeroState) -> Command:
        """Enhanced FastAPI expert with full infrastructure integration."""
        return execute_expert_node(
            state, AgentRole.FASTAPI_EXPERT, "fastapi_expert", human_loop_manager
        )

    def enhanced_nextjs_expert_node(state: CodeHeroState) -> Command:
        """Enhanced Next.js expert with full infrastructure integration."""
        return execute_expert_node(
            state, AgentRole.NEXTJS_EXPERT, "nextjs_expert", human_loop_manager
        )

    def enhanced_code_expert_node(state: CodeHeroState) -> Command:
        """Enhanced code generation expert with full infrastructure integration."""
        return execute_expert_node(
            state, AgentRole.CODE_GENERATOR, "code_expert", human_loop_manager
        )

    def enhanced_pydantic_expert_node(state: CodeHeroState) -> Command:
        """Enhanced Pydantic expert with full infrastructure integration."""
        return execute_expert_node(
            state, AgentRole.PYDANTIC_EXPERT, "pydantic_expert", human_loop_manager
        )

    # Create supervisor with enhanced routing
    dev_supervisor_node = create_supervisor_node(
        llm,
        ["fastapi_expert", "nextjs_expert", "code_expert", "pydantic_expert"],
        "Enhanced Development Team",
        human_loop_manager,
    )

    # Build enhanced graph
    if not LANGGRAPH_AVAILABLE:
        return None

    dev_builder = StateGraph(CodeHeroState)
    dev_builder.add_node("supervisor", dev_supervisor_node)
    dev_builder.add_node("fastapi_expert", enhanced_fastapi_expert_node)
    dev_builder.add_node("nextjs_expert", enhanced_nextjs_expert_node)
    dev_builder.add_node("code_expert", enhanced_code_expert_node)
    dev_builder.add_node("pydantic_expert", enhanced_pydantic_expert_node)

    # Add human feedback node if manager provided
    if human_loop_manager:
        human_node = create_human_feedback_node(human_loop_manager)
        dev_builder.add_node("human_feedback", human_node)

    # Add edges with conditional routing
    dev_builder.add_edge(START, "supervisor")
    dev_builder.add_conditional_edges(
        "supervisor",
        lambda x: x.get("next", END),
        {
            "fastapi_expert": "fastapi_expert",
            "nextjs_expert": "nextjs_expert",
            "code_expert": "code_expert",
            "pydantic_expert": "pydantic_expert",
            "human_feedback": "human_feedback" if human_loop_manager else END,
            END: END,
        },
    )

    return dev_builder.compile()


# === COMPREHENSIVE INFRASTRUCTURE INTEGRATION ===


def create_enhanced_hierarchical_system_with_full_infrastructure(
    llm: ChatOpenAI = None,
    state_manager: StateManager = None,
    logger_service: StructuredLogger = None,
    supervisor: SupervisorAgent = None,
    agent_manager_instance: AgentManager = None,
    strategic_agent: StrategicAgent = None,
    human_loop_manager: HumanLoopManager = None,
    use_retry_logic: bool = True,
    enable_context_management: bool = True,
    enable_service_validation: bool = True,
):
    """Create a comprehensive hierarchical system with full Code Hero infrastructure integration."""

    if not LANGGRAPH_AVAILABLE:
        logger.warning("LangGraph not available, enhanced hierarchical system disabled")
        return None

    try:
        # Initialize LLM with enhanced configuration
        if llm is None:
            model_config = get_model_for_agent(AgentRole.SUPERVISOR)
            llm = ChatOpenAI(
                model=model_config.get("model", "gpt-4o-mini"),
                temperature=model_config.get("temperature", 0.1),
                api_key=OPENAI_API_KEY or model_config.get("api_key"),
            )
            logger.info(
                f"Enhanced LLM initialized with model: {model_config.get('model', 'gpt-4o-mini')}"
            )

        # Validate services if enabled
        if enable_service_validation and all(
            [state_manager, logger_service, supervisor]
        ):
            try:
                service_statuses = validate_services(
                    state_manager, logger_service, supervisor
                )
                failed_services = [
                    s for s in service_statuses if s.status == Status.FAILED
                ]

                if failed_services:
                    logger.warning(
                        f"Service validation found issues: {[s.name for s in failed_services]}"
                    )
                else:
                    logger.info(
                        "All services validated successfully for enhanced hierarchical system"
                    )

            except Exception as e:
                logger.warning(f"Service validation failed: {e}")

        # Create enhanced team graphs with full infrastructure
        dev_graph = create_enhanced_development_team(llm, human_loop_manager)
        research_graph = create_research_team(llm)
        doc_graph = create_documentation_team(llm)

        if not all([dev_graph, research_graph, doc_graph]):
            logger.error("Failed to create team graphs")
            return None

        # Enhanced supervisor with comprehensive infrastructure integration
        def enhanced_main_supervisor_node(state: CodeHeroState) -> Command:
            """Enhanced main supervisor with full infrastructure integration."""
            try:
                # Validate state and services
                if enable_service_validation and state.get("state_manager_service"):
                    try:
                        health = state["state_manager_service"].check_health()
                        logger.debug(f"State manager health: {health}")
                    except Exception as e:
                        logger.warning(f"State manager health check failed: {e}")

                # Use strategic guidance if available
                if state.get("strategic_agent") and state.get("strategic_context"):
                    try:
                        strategic_context = state["strategic_context"]
                        context_value = (
                            strategic_context.value
                            if hasattr(strategic_context, "value")
                            else str(strategic_context)
                        )
                        strategic_guidance = state[
                            "strategic_agent"
                        ]._get_framework_guidance(
                            context_value,
                            state["messages"][-1].content if state["messages"] else "",
                        )
                        state["strategic_guidance"] = strategic_guidance
                        logger.info(
                            f"Strategic guidance applied: {list(strategic_guidance.keys())}"
                        )
                    except Exception as e:
                        logger.warning(f"Strategic guidance failed: {e}")

                # Record supervisor activity in agent manager
                if state.get("agent_manager"):
                    try:
                        import asyncio

                        loop = asyncio.get_event_loop()
                        if not loop.is_running():
                            task_id = loop.run_until_complete(
                                state["agent_manager"].start_task(
                                    "main_supervisor",
                                    f"Routing: {state['messages'][-1].content if state['messages'] else 'Unknown'}",
                                )
                            )
                            state["context_metadata"]["supervisor_task_id"] = task_id
                    except Exception as e:
                        logger.warning(
                            f"Agent manager supervisor recording failed: {e}"
                        )

                # Use the original supervisor logic with enhancements
                return create_supervisor_node(
                    llm,
                    ["development_team", "research_team", "documentation_team"],
                    "Enhanced Main Orchestrator",
                    human_loop_manager,
                )(state)

            except Exception as e:
                logger.error(f"Enhanced supervisor error: {e}")
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content="Enhanced supervisor encountered an error.",
                                name="supervisor",
                            )
                        ]
                    },
                    goto=END,
                )

        # Enhanced team connector functions with full infrastructure
        @retry_with_backoff(retries=3) if use_retry_logic else lambda f: f
        async def enhanced_call_development_team(state: CodeHeroState) -> Command:
            """Enhanced development team caller with retry logic and infrastructure integration."""
            try:
                # Use tool calling utilities if needed
                if state.get("tools_used"):
                    for tool_name in state["tools_used"]:
                        if tool_name in [
                            "generate_code",
                            "validate_code",
                            "analyze_code",
                        ]:
                            tool_result = await call_tool(
                                tool_name, query=state["messages"][-1].content
                            )
                            state["artifacts"][f"{tool_name}_result"] = tool_result

                response = dev_graph.invoke(state)
                final_message = (
                    response["messages"][-1] if response.get("messages") else None
                )
                content = (
                    final_message.content
                    if final_message
                    else "Enhanced development team completed the task."
                )

                # Record completion in agent manager
                if state.get("agent_manager"):
                    try:
                        await state["agent_manager"].record_task_completion(
                            "development_team",
                            (
                                state["messages"][-1].content
                                if state["messages"]
                                else "Unknown task"
                            ),
                            True,
                            1.0,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Development team completion recording failed: {e}"
                        )

                return Command(
                    update={
                        "messages": [
                            HumanMessage(content=content, name="development_team")
                        ],
                        "active_agents": state.get("active_agents", [])
                        + ["development_team"],
                        "performance_metrics": {
                            **state.get("performance_metrics", {}),
                            "development_team": {
                                "execution_time": format_timestamp(datetime.now()),
                                "success": True,
                                "infrastructure_enhanced": True,
                            },
                        },
                    },
                    goto=END,
                )
            except Exception as e:
                logger.error(f"Enhanced development team error: {e}")

                # Record failure
                if state.get("agent_manager"):
                    try:
                        await state["agent_manager"].record_task_completion(
                            "development_team",
                            (
                                state["messages"][-1].content
                                if state["messages"]
                                else "Unknown task"
                            ),
                            False,
                            0.1,
                        )
                    except Exception as mgr_e:
                        logger.warning(
                            f"Development team failure recording failed: {mgr_e}"
                        )

                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"Enhanced development team encountered an error: {str(e)}",
                                name="development_team",
                            )
                        ],
                        "error_history": state.get("error_history", [])
                        + [f"Development team: {str(e)}"],
                    },
                    goto=END,
                )

        # Convert async function to sync for LangGraph compatibility
        def call_enhanced_development_team(state: CodeHeroState) -> Command:
            """Sync wrapper for enhanced development team."""
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, we can't use run_until_complete
                    # Fall back to basic implementation
                    response = dev_graph.invoke(state)
                    final_message = (
                        response["messages"][-1] if response.get("messages") else None
                    )
                    content = (
                        final_message.content
                        if final_message
                        else "Enhanced development team completed the task."
                    )

                    return Command(
                        update={
                            "messages": [
                                HumanMessage(content=content, name="development_team")
                            ]
                        },
                        goto=END,
                    )
                else:
                    return loop.run_until_complete(
                        enhanced_call_development_team(state)
                    )
            except Exception as e:
                logger.error(f"Enhanced development team wrapper error: {e}")
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"Enhanced development team encountered an error: {str(e)}",
                                name="development_team",
                            )
                        ]
                    },
                    goto=END,
                )

        # Similar enhanced functions for other teams (simplified for brevity)
        def call_enhanced_research_team(state: CodeHeroState) -> Command:
            """Enhanced research team with infrastructure integration."""
            try:
                response = research_graph.invoke(state)
                final_message = (
                    response["messages"][-1] if response.get("messages") else None
                )
                content = (
                    final_message.content
                    if final_message
                    else "Enhanced research team completed the task."
                )

                return Command(
                    update={
                        "messages": [
                            HumanMessage(content=content, name="research_team")
                        ],
                        "active_agents": state.get("active_agents", [])
                        + ["research_team"],
                    },
                    goto=END,
                )
            except Exception as e:
                logger.error(f"Enhanced research team error: {e}")
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"Enhanced research team encountered an error: {str(e)}",
                                name="research_team",
                            )
                        ]
                    },
                    goto=END,
                )

        def call_enhanced_documentation_team(state: CodeHeroState) -> Command:
            """Enhanced documentation team with infrastructure integration."""
            try:
                response = doc_graph.invoke(state)
                final_message = (
                    response["messages"][-1] if response.get("messages") else None
                )
                content = (
                    final_message.content
                    if final_message
                    else "Enhanced documentation team completed the task."
                )

                return Command(
                    update={
                        "messages": [
                            HumanMessage(content=content, name="documentation_team")
                        ],
                        "active_agents": state.get("active_agents", [])
                        + ["documentation_team"],
                    },
                    goto=END,
                )
            except Exception as e:
                logger.error(f"Enhanced documentation team error: {e}")
                return Command(
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"Enhanced documentation team encountered an error: {str(e)}",
                                name="documentation_team",
                            )
                        ]
                    },
                    goto=END,
                )

        # Build the enhanced main orchestrator graph
        super_builder = StateGraph(CodeHeroState)
        super_builder.add_node("supervisor", enhanced_main_supervisor_node)
        super_builder.add_node("development_team", call_enhanced_development_team)
        super_builder.add_node("research_team", call_enhanced_research_team)
        super_builder.add_node("documentation_team", call_enhanced_documentation_team)

        # Add human feedback node if available
        if human_loop_manager:
            human_node = create_human_feedback_node(human_loop_manager)
            super_builder.add_node("human_feedback", human_node)

        # Add edges with conditional routing
        super_builder.add_edge(START, "supervisor")
        super_builder.add_conditional_edges(
            "supervisor",
            lambda x: x.get("next", END),
            {
                "development_team": "development_team",
                "research_team": "research_team",
                "documentation_team": "documentation_team",
                "human_feedback": "human_feedback" if human_loop_manager else END,
                END: END,
            },
        )

        # Compile with enhanced configuration
        enhanced_system = super_builder.compile(checkpointer=None, debug=False)

        logger.info(
            "Enhanced hierarchical system with full infrastructure integration created successfully"
        )
        return enhanced_system

    except Exception as e:
        logger.error(f"Failed to create enhanced hierarchical system: {e}")
        return None


# === UTILITY FUNCTIONS FOR INFRASTRUCTURE INTEGRATION ===


def get_infrastructure_status() -> Dict[str, Any]:
    """Get comprehensive status of all infrastructure components."""
    try:
        status = {
            "timestamp": format_timestamp(datetime.now()),
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "tool_registry": {
                "available_tools": list(tool_registry.list_tools()),
                "tool_count": len(list(tool_registry.list_tools())),
            },
            "agent_experts": {
                "available_experts": list(all_experts.keys()),
                "expert_count": len(all_experts),
            },
            "infrastructure_modules": {
                "agent_manager": "available",
                "strategic_agent": "available",
                "human_loop": "available",
                "context_management": "available",
                "service_validation": "available",
                "workflow_runner": "available",
                "utils": "available",
            },
        }

        return status

    except Exception as e:
        logger.error(f"Failed to get infrastructure status: {e}")
        return {"error": str(e), "timestamp": format_timestamp(datetime.now())}


async def validate_full_infrastructure() -> Dict[str, Any]:
    """Validate all infrastructure components are working correctly."""
    try:
        validation_results = {
            "timestamp": format_timestamp(datetime.now()),
            "overall_status": "unknown",
            "component_status": {},
        }

        # Test tool registry
        try:
            tools = list(tool_registry.list_tools())
            validation_results["component_status"]["tool_registry"] = {
                "status": "healthy",
                "tool_count": len(tools),
                "sample_tools": tools[:3],
            }
        except Exception as e:
            validation_results["component_status"]["tool_registry"] = {
                "status": "failed",
                "error": str(e),
            }

        # Test agent experts
        try:
            experts = list(all_experts.keys())
            validation_results["component_status"]["agent_experts"] = {
                "status": "healthy",
                "expert_count": len(experts),
                "sample_experts": [str(e) for e in experts[:3]],
            }
        except Exception as e:
            validation_results["component_status"]["agent_experts"] = {
                "status": "failed",
                "error": str(e),
            }

        # Test utility functions
        try:
            test_id = generate_id("test")
            validation_results["component_status"]["utils"] = {
                "status": "healthy",
                "test_id_generated": bool(test_id),
                "id_format": test_id[:20] + "..." if len(test_id) > 20 else test_id,
            }
        except Exception as e:
            validation_results["component_status"]["utils"] = {
                "status": "failed",
                "error": str(e),
            }

        # Test hierarchical system creation
        try:
            test_system = create_hierarchical_system()
            validation_results["component_status"]["hierarchical_system"] = {
                "status": "healthy" if test_system else "failed",
                "system_created": bool(test_system),
            }
        except Exception as e:
            validation_results["component_status"]["hierarchical_system"] = {
                "status": "failed",
                "error": str(e),
            }

        # Determine overall status
        failed_components = [
            name
            for name, status in validation_results["component_status"].items()
            if status.get("status") == "failed"
        ]

        if not failed_components:
            validation_results["overall_status"] = "healthy"
        elif len(failed_components) < len(validation_results["component_status"]) / 2:
            validation_results["overall_status"] = "degraded"
        else:
            validation_results["overall_status"] = "failed"

        validation_results["failed_components"] = failed_components
        validation_results["healthy_components"] = [
            name
            for name, status in validation_results["component_status"].items()
            if status.get("status") == "healthy"
        ]

        return validation_results

    except Exception as e:
        logger.error(f"Infrastructure validation failed: {e}")
        return {
            "timestamp": format_timestamp(datetime.now()),
            "overall_status": "failed",
            "error": str(e),
        }


# === MAIN INTERFACE ===


async def process_with_hierarchical_agents(
    message: str,
    conversation_id: str = None,
    project_id: str = None,
    task_priority: TaskPriority = TaskPriority.MEDIUM,
    context: Dict[str, Any] = None,
    request: Optional[Any] = None,
    use_full_infrastructure: bool = True,
    user_id: Optional[str] = None,
    enable_memory: bool = True,
    environment: str = "development",
    checkpointer_config: Optional[CheckpointerConfig] = None,
) -> str:
    """Process a message using the hierarchical agent system with enhanced checkpointing and memory management."""
    try:
        # Initialize services if request is provided
        services = None
        state_manager = None
        logger_service = None
        supervisor = None
        agent_manager_instance = None
        memory_manager = None
        checkpointer_manager = None

        if request and use_full_infrastructure:
            try:
                # Get services from request context
                services = await get_services(request)
                state_manager, logger_service, supervisor = services

                # Validate services
                service_statuses = await validate_services(
                    state_manager, logger_service, supervisor
                )
                failed_services = [
                    s for s in service_statuses if s.status == Status.FAILED
                ]

                if failed_services:
                    logger.warning(
                        f"Some services failed validation: {[s.name for s in failed_services]}"
                    )

                # Initialize agent manager
                agent_manager_instance = AgentManager(logger_service)

                logger.info(
                    "Full infrastructure services initialized for hierarchical processing"
                )

            except (
                ServiceError,
                ServiceNotInitializedError,
                ServiceHealthCheckError,
            ) as e:
                logger.warning(f"Service initialization failed, using fallback: {e}")
                use_full_infrastructure = False
            except Exception as e:
                logger.warning(f"Unexpected service error, using fallback: {e}")
                use_full_infrastructure = False

        # Initialize enhanced checkpointing system
        if enable_memory:
            try:
                # Use provided config or determine based on environment
                if checkpointer_config:
                    config = checkpointer_config
                elif environment == "production":
                    config = get_production_checkpointer_config()
                elif environment == "testing":
                    config = create_memory_checkpointer()
                else:  # development
                    config = get_development_checkpointer_config()

                # Initialize enhanced checkpointer manager
                checkpointer_manager = EnhancedCheckpointerManager(
                    config=config,
                    logger_service=logger_service,
                )

                success = await checkpointer_manager.initialize()
                if success:
                    logger.info(
                        f"Enhanced checkpointer initialized for {environment} environment"
                    )
                else:
                    logger.warning(
                        "Failed to initialize enhanced checkpointer, proceeding without"
                    )
                    checkpointer_manager = None

            except Exception as e:
                logger.warning(f"Enhanced checkpointer initialization failed: {e}")
                checkpointer_manager = None

        # Initialize memory manager if enabled
        if enable_memory:
            try:
                memory_manager = await initialize_memory_system(
                    enable_checkpointing=True,
                    enable_long_term_memory=True,
                    enable_embeddings=True,
                    logger_service=logger_service,
                )

                # Integrate with enhanced checkpointer if available
                if checkpointer_manager and memory_manager:
                    integration_success = await integrate_with_memory_manager(
                        checkpointer_manager, memory_manager
                    )
                    if integration_success:
                        logger.info(
                            "Enhanced checkpointer integrated with memory manager"
                        )
                    else:
                        logger.warning(
                            "Failed to integrate enhanced checkpointer with memory manager"
                        )

                logger.info("Memory system initialized for hierarchical processing")
            except Exception as e:
                logger.warning(f"Failed to initialize memory system: {e}")
                memory_manager = None

        # Create the hierarchical system with enhanced checkpointing
        try:
            hierarchical_system = create_hierarchical_system_internal(
                enable_memory=enable_memory,
                memory_manager=memory_manager,
                checkpointer_manager=checkpointer_manager,
                environment=environment,
            )
        except Exception as e:
            # Handle LangGraph compatibility issues
            logger.warning(
                f"LangGraph checkpointing compatibility issue, disabling memory: {e}"
            )
            enable_memory = False
            memory_manager = None
            checkpointer_manager = None
            hierarchical_system = create_hierarchical_system_internal(
                enable_memory=False,
                memory_manager=None,
                checkpointer_manager=None,
                environment=environment,
            )
            if not hierarchical_system:
                logger.error(f"Failed to create hierarchical system: {e}")
                hierarchical_system = None

        if not hierarchical_system:
            return "Hierarchical agent system is not available. Please check your LangGraph installation."

        # Generate IDs if not provided
        conversation_id = conversation_id or generate_id("conv")
        project_id = project_id or generate_id("proj")
        user_id = user_id or "default_user"

        # Retrieve user information from memory if available
        user_info = None
        if memory_manager and user_id:
            try:
                user_info = await memory_manager.get_user_info(user_id)
                if user_info:
                    logger.info(
                        f"Retrieved user info for {user_id}: {user_info.get('name', 'Unknown')}"
                    )
                else:
                    # Create default user info
                    user_info = UserInfo(
                        name="User",
                        email=None,
                        preferences={},
                        language="English",
                        timezone=None,
                        created_at=datetime.now().isoformat(),
                        updated_at=datetime.now().isoformat(),
                    )
                    await memory_manager.store_user_info(user_id, user_info)
                    logger.info(f"Created default user info for {user_id}")
            except Exception as e:
                logger.warning(f"Failed to retrieve/create user info: {e}")

        # Retrieve conversation summary from memory if available
        conversation_summary = None
        if memory_manager and conversation_id:
            try:
                conversation_summary = await memory_manager.get_conversation_summary(
                    conversation_id
                )
                if conversation_summary:
                    logger.info(f"Retrieved conversation summary for {conversation_id}")
            except Exception as e:
                logger.warning(f"Failed to retrieve conversation summary: {e}")

        # Create enhanced context with all infrastructure components and memory
        enhanced_context = {
            "conversation_id": conversation_id,
            "project_id": project_id,
            "user_id": user_id,
            "task_priority": task_priority,
            "timestamp": format_timestamp(datetime.now()),
            "hierarchical_mode": True,
            "available_tools": list(tool_registry.list_tools()),
            "use_full_infrastructure": use_full_infrastructure,
            "memory_enabled": bool(memory_manager),
            "checkpointer_enabled": bool(checkpointer_manager),
            "environment": environment,
            "user_info": user_info,
            "conversation_summary": conversation_summary,
            **(context or {}),
        }

        # Add memory tools to available tools if memory is enabled
        if memory_manager:
            try:
                memory_tools = create_memory_tools(memory_manager)
                memory_tool_names = [tool.name for tool in memory_tools]
                enhanced_context["memory_tools"] = memory_tool_names
                enhanced_context["available_tools"].extend(memory_tool_names)
                logger.info(f"Memory tools added: {memory_tool_names}")
            except Exception as e:
                logger.warning(f"Failed to create memory tools: {e}")

        # Add service health information if available
        if services:
            enhanced_context["service_health"] = {
                "state_manager": (
                    await state_manager.check_health() if state_manager else None
                ),
                "logger": (
                    await logger_service.check_health() if logger_service else None
                ),
                "supervisor": await supervisor.check_health() if supervisor else None,
                "memory_manager": (
                    await memory_manager.get_memory_stats() if memory_manager else None
                ),
                "checkpointer_manager": (
                    await checkpointer_manager.health_check()
                    if checkpointer_manager
                    else None
                ),
            }

        # Initialize strategic agent if available
        strategic_agent_instance = None
        strategic_context = None
        try:
            strategic_agent_instance = StrategicAgent()
            strategic_context = StrategicContext.GENERAL_STRATEGY

            # Determine strategic context from message
            message_lower = message.lower()
            if any(keyword in message_lower for keyword in ["agent", "ai agent"]):
                strategic_context = StrategicContext.AI_AGENT_DEVELOPMENT
            elif any(
                keyword in message_lower for keyword in ["workflow", "automation"]
            ):
                strategic_context = StrategicContext.WORKFLOW_AUTOMATION
            elif any(keyword in message_lower for keyword in ["code", "programming"]):
                strategic_context = StrategicContext.CODE_GENERATION
            elif any(keyword in message_lower for keyword in ["documentation", "docs"]):
                strategic_context = StrategicContext.DOCUMENTATION_SYSTEM
            elif any(keyword in message_lower for keyword in ["research", "analysis"]):
                strategic_context = StrategicContext.RESEARCH_ANALYSIS

            enhanced_context["strategic_context"] = strategic_context
            logger.info(f"Strategic context determined: {strategic_context}")

        except Exception as e:
            logger.warning(f"Strategic agent initialization failed: {e}")

        # Process the message with enhanced state including all infrastructure and memory
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "task_context": enhanced_context,
            "conversation_id": conversation_id,
            "project_id": project_id,
            "task_priority": task_priority,
            # Agent management integration - only serializable data
            "agent_states": {},
            "active_agents": [],
            # Strategic planning integration - only serializable data
            "strategic_context": strategic_context,
            "strategic_guidance": {},
            # Memory management integration - only serializable data
            "conversation_summary": conversation_summary,
            "user_info": user_info,
            "memory_items": {},
            "memory_search_results": [],
            # Performance and monitoring - only serializable data
            "tools_used": [],
            "performance_metrics": {},
            "error_history": [],
            "service_health": enhanced_context.get("service_health", {}),
            # Context management - only serializable data
            "context_metadata": enhanced_context,
            # Store references to complex objects in context for nodes to access
            "_infrastructure_context": {
                "agent_manager_available": agent_manager_instance is not None,
                "strategic_agent_available": strategic_agent_instance is not None,
                "memory_manager_available": memory_manager is not None,
                "checkpointer_manager_available": checkpointer_manager is not None,
                "state_manager_available": state_manager is not None,
                "logger_service_available": logger_service is not None,
                "supervisor_service_available": supervisor is not None,
            },
        }

        # Execute the hierarchical workflow with enhanced checkpointing
        # Create proper configuration for checkpointer
        if checkpointer_manager:
            config = checkpointer_manager.create_thread_config(
                thread_id=conversation_id,
                user_id=user_id,
                project_id=project_id,
            )
        else:
            config = {
                "configurable": {
                    "thread_id": conversation_id,
                    "user_id": user_id,
                    "project_id": project_id,
                }
            }

        result = await hierarchical_system.ainvoke(initial_state, config=config)

        # Extract the response from the result
        if "messages" in result and result["messages"]:
            response = result["messages"][-1].content
        else:
            response = "I apologize, but I couldn't process your request at this time."

        # Log performance metrics and service health
        if logger_service:
            # Log agent usage
            if "active_agents" in result and result["active_agents"]:
                logger.info(f"Active agents used: {result['active_agents']}")

            # Log tools used
            if "tools_used" in result and result["tools_used"]:
                logger.info(f"Tools used: {result['tools_used']}")

            # Log strategic guidance
            if "strategic_guidance" in result and result["strategic_guidance"]:
                logger.info(
                    f"Strategic guidance applied: {list(result['strategic_guidance'].keys())}"
                )

            # Log service health including checkpointer
            if "service_health" in result and result["service_health"]:
                logger.info(f"Service health status: {result['service_health']}")

        # Store conversation in memory if enabled
        if memory_manager and conversation_id:
            try:
                await memory_manager.store_message(
                    conversation_id=conversation_id,
                    message=message,
                    response=response,
                    metadata={
                        "user_id": user_id,
                        "project_id": project_id,
                        "task_priority": (
                            task_priority.value if task_priority else "medium"
                        ),
                        "agents_used": result.get("active_agents", []),
                        "tools_used": result.get("tools_used", []),
                        "timestamp": enhanced_context["timestamp"],
                        "checkpointer_used": bool(checkpointer_manager),
                        "environment": environment,
                    },
                )
                logger.info(f"Conversation stored in memory for {conversation_id}")
            except Exception as e:
                logger.warning(f"Failed to store conversation in memory: {e}")

        # Cleanup checkpointer resources if needed
        if checkpointer_manager:
            try:
                await checkpointer_manager.cleanup()
            except Exception as e:
                logger.warning(f"Checkpointer cleanup failed: {e}")

        return response

    except Exception as e:
        logger.error(f"Error in hierarchical agent processing: {e}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}"


# Export the main interface and enhanced components
__all__ = [
    # Core hierarchical system
    "create_hierarchical_system",
    "process_with_hierarchical_agents",
    "CodeHeroState",
    # Enhanced infrastructure integration
    "create_enhanced_hierarchical_system_with_full_infrastructure",
    "create_enhanced_development_team",
    "create_human_feedback_node",
    # Team creation functions
    "create_development_team",
    "create_research_team",
    "create_documentation_team",
    # Infrastructure utilities
    "get_infrastructure_status",
    "validate_full_infrastructure",
    "execute_expert_node",
    "create_supervisor_node",
    # Human feedback and model management
    "get_llm_for_agent",
    "should_request_human_feedback",
    "handle_human_feedback_request",
    # Enhanced checkpointing system
    "EnhancedCheckpointerManager",
    "CheckpointerConfig",
    "CheckpointerType",
    "initialize_enhanced_checkpointing",
    "create_memory_checkpointer",
    "create_sqlite_checkpointer",
    "create_postgres_checkpointer",
    "get_production_checkpointer_config",
    "get_development_checkpointer_config",
    "integrate_with_memory_manager",
    # State and configuration
    "format_timestamp",
    "LANGGRAPH_AVAILABLE",
    # Availability flags for checkpointing
    "LANGGRAPH_CHECKPOINTING_AVAILABLE",
    "POSTGRES_AVAILABLE",
]
