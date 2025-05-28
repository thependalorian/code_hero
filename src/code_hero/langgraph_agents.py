"""Enhanced LangGraph agents for Code Hero with Command-based routing and industry-level coordination."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.types import Command

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

    # Fallback Command implementation
    class Command:
        def __init__(self, goto: str, update: Optional[Dict[str, Any]] = None):
            self.goto = goto
            self.update = update or {}


from .state import (
    AgentResponse,
    AgentRole,
    CodeGenerationRequest,
    CodeGenerationResponse,
    ResearchRequest,
    ResearchResponse,
)


# Enhanced state for agent workflows with Command support
class CodeHeroWorkflowState(MessagesState):
    """Enhanced state for Code Hero workflows with Command-based routing."""

    # Core workflow state
    current_agent: str = "supervisor"
    task_type: str = "general"
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None

    # Agent coordination
    agent_history: List[str] = []
    handoff_context: Dict[str, Any] = {}
    shared_memory: Dict[str, Any] = {}

    # Tool and context management
    tools_available: List[str] = []
    context: Dict[str, Any] = {}
    status: str = "pending"

    # Command routing
    next_agent: Optional[str] = None
    routing_reason: Optional[str] = None
    requires_human_review: bool = False


# Agent routing types for type safety
AgentName = Literal[
    "supervisor",
    "code_expert",
    "research_expert",
    "fastapi_expert",
    "nextjs_expert",
    "documentation_expert",
    "review_expert",
    "human_review",
]


def create_agent_response(
    agent_role: AgentRole, user_input: str, context: Dict[str, Any]
) -> str:
    """Create enhanced agent response using industry-level prompts."""
    try:
        # Import required modules for LLM integration
        from .agent_expert import experts
        from .state import AgentState, Status

        # Get the appropriate expert for this role
        expert = experts.get(agent_role)
        if not expert:
            return f"I'm sorry, the {agent_role.value.replace('_', ' ').title()} expert is not available right now."

        # Create a temporary agent state for the expert
        temp_state = AgentState(
            id=f"temp_{agent_role.value}_{datetime.utcnow().timestamp()}",
            agent=agent_role,
            status=Status.PENDING,
            context=context,
        )

        # Use the expert's LLM integration to generate response
        if hasattr(expert, "generate_response_with_llm"):
            # Use the expert's LLM method if available
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we need to handle this differently
                # For now, use the fallback response method
                response = expert._fallback_response(user_input, context)
            else:
                response = loop.run_until_complete(
                    expert.generate_response_with_llm(user_input, context)
                )
        else:
            # Use fallback response method
            response = expert._fallback_response(user_input, context)

        return response

    except Exception as e:
        # Enhanced error handling with more informative messages
        role_name = agent_role.value.replace("_", " ").title()
        return f"I'm your {role_name} specialist. I encountered a technical issue while processing your request: {str(e)}. Please try again or rephrase your question."


# === SUPERVISOR AGENT WITH COMMAND ROUTING ===


def supervisor_router(state: CodeHeroWorkflowState) -> Command:
    """Enhanced supervisor with Command-based routing following LangGraph best practices."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message:
        return Command(goto="supervisor_response")

    user_input = last_message.content.lower()

    # Advanced routing logic with context preservation
    routing_context = {
        "user_input": user_input,
        "agent_history": state.get("agent_history", []),
        "shared_memory": state.get("shared_memory", {}),
        "task_complexity": "medium",  # Could be determined by analysis
    }

    # Code-related requests
    if any(
        keyword in user_input
        for keyword in [
            "code",
            "implement",
            "build",
            "create",
            "develop",
            "function",
            "class",
            "api",
            "endpoint",
            "component",
        ]
    ):
        # Determine specific code expert needed
        if any(
            keyword in user_input for keyword in ["fastapi", "backend", "api", "server"]
        ):
            next_agent = "fastapi_expert"
            routing_reason = "FastAPI backend development required"
        elif any(
            keyword in user_input
            for keyword in ["nextjs", "react", "frontend", "component", "ui"]
        ):
            next_agent = "nextjs_expert"
            routing_reason = "Next.js frontend development required"
        else:
            next_agent = "code_expert"
            routing_reason = "General code development required"

        return Command(
            goto=next_agent,
            update={
                "task_type": "code_development",
                "routing_reason": routing_reason,
                "handoff_context": routing_context,
                "agent_history": state.get("agent_history", []) + ["supervisor"],
            },
        )

    # Research and analysis requests
    elif any(
        keyword in user_input
        for keyword in [
            "search",
            "find",
            "research",
            "analyze",
            "investigate",
            "documentation",
            "explain",
            "what is",
            "how does",
        ]
    ):
        return Command(
            goto="research_expert",
            update={
                "task_type": "research_analysis",
                "routing_reason": "Research and analysis required",
                "handoff_context": routing_context,
                "agent_history": state.get("agent_history", []) + ["supervisor"],
            },
        )

    # Documentation requests
    elif any(
        keyword in user_input
        for keyword in ["document", "docs", "readme", "guide", "tutorial", "explain"]
    ):
        return Command(
            goto="documentation_expert",
            update={
                "task_type": "documentation",
                "routing_reason": "Documentation creation required",
                "handoff_context": routing_context,
                "agent_history": state.get("agent_history", []) + ["supervisor"],
            },
        )

    # Review and validation requests
    elif any(
        keyword in user_input
        for keyword in ["review", "check", "validate", "test", "quality", "improve"]
    ):
        return Command(
            goto="review_expert",
            update={
                "task_type": "review_validation",
                "routing_reason": "Code review and validation required",
                "handoff_context": routing_context,
                "agent_history": state.get("agent_history", []) + ["supervisor"],
            },
        )

    # Default to supervisor response for general queries
    else:
        return Command(
            goto="supervisor_response",
            update={
                "task_type": "general_assistance",
                "routing_reason": "General assistance provided by supervisor",
                "handoff_context": routing_context,
            },
        )


def supervisor_response(state: CodeHeroWorkflowState) -> CodeHeroWorkflowState:
    """Enhanced supervisor response with comprehensive assistance."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message:
        content = "I'm your AI development supervisor. How can I assist you today?"
    else:
        user_input = last_message.content

        # Create comprehensive context for supervisor
        context = {
            "query": user_input,
            "available_experts": [
                "FastAPI Expert - Backend development and APIs",
                "Next.js Expert - Frontend development and React components",
                "Research Expert - Information gathering and analysis",
                "Documentation Expert - Technical writing and guides",
                "Code Review Expert - Quality assurance and validation",
            ],
            "capabilities": [
                "Multi-agent task coordination",
                "Code generation and review",
                "Research and analysis",
                "Documentation creation",
                "Technical consultation",
            ],
            "routing_history": state.get("agent_history", []),
            **state.get("context", {}),
        }

        # Use enhanced prompt system
        content = create_agent_response(AgentRole.SUPERVISOR, user_input, context)

    ai_message = AIMessage(content=content)
    state["messages"] = state["messages"] + [ai_message]
    state["status"] = "completed"
    state["current_agent"] = "supervisor"

    return state


# === SPECIALIZED EXPERT AGENTS WITH COMMAND SUPPORT ===


def fastapi_expert_response(state: CodeHeroWorkflowState) -> CodeHeroWorkflowState:
    """FastAPI expert with enhanced capabilities and potential handoffs."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message:
        return state

    user_input = last_message.content
    handoff_context = state.get("handoff_context", {})

    # Enhanced context for FastAPI development
    context = {
        "specialization": "FastAPI backend development",
        "expertise": [
            "REST API design and implementation",
            "Database integration with SQLAlchemy",
            "Authentication and authorization",
            "API documentation with OpenAPI",
            "Performance optimization",
            "Testing and validation",
        ],
        "handoff_context": handoff_context,
        "previous_agents": state.get("agent_history", []),
        **state.get("context", {}),
    }

    # Generate enhanced response
    response_content = create_agent_response(
        AgentRole.FASTAPI_EXPERT, user_input, context
    )

    # Check if handoff to review is needed
    if any(
        keyword in user_input.lower() for keyword in ["review", "check", "validate"]
    ):
        response_content += "\n\n*Recommending code review after implementation.*"
        state["next_agent"] = "review_expert"
        state["requires_human_review"] = True

    ai_message = AIMessage(content=response_content)
    state["messages"] = state["messages"] + [ai_message]
    state["current_agent"] = "fastapi_expert"
    state["agent_history"] = state.get("agent_history", []) + ["fastapi_expert"]

    return state


def nextjs_expert_response(state: CodeHeroWorkflowState) -> CodeHeroWorkflowState:
    """Next.js expert with enhanced capabilities and component focus."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message:
        return state

    user_input = last_message.content
    handoff_context = state.get("handoff_context", {})

    # Enhanced context for Next.js development
    context = {
        "specialization": "Next.js frontend development",
        "expertise": [
            "React component development",
            "Next.js App Router and SSR",
            "Tailwind CSS and DaisyUI styling",
            "State management and hooks",
            "Performance optimization",
            "Responsive design patterns",
        ],
        "handoff_context": handoff_context,
        "previous_agents": state.get("agent_history", []),
        **state.get("context", {}),
    }

    # Generate enhanced response
    response_content = create_agent_response(
        AgentRole.NEXTJS_EXPERT, user_input, context
    )

    ai_message = AIMessage(content=response_content)
    state["messages"] = state["messages"] + [ai_message]
    state["current_agent"] = "nextjs_expert"
    state["agent_history"] = state.get("agent_history", []) + ["nextjs_expert"]

    return state


def research_expert_response(state: CodeHeroWorkflowState) -> CodeHeroWorkflowState:
    """Research expert with comprehensive analysis capabilities."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message:
        return state

    user_input = last_message.content
    handoff_context = state.get("handoff_context", {})

    # Enhanced context for research
    context = {
        "specialization": "Research and analysis",
        "capabilities": [
            "Document search and analysis",
            "Web research and fact-checking",
            "Technical documentation review",
            "Best practices identification",
            "Comparative analysis",
            "Trend identification",
        ],
        "handoff_context": handoff_context,
        "previous_agents": state.get("agent_history", []),
        **state.get("context", {}),
    }

    # Generate enhanced response
    response_content = create_agent_response(AgentRole.RESEARCH, user_input, context)

    ai_message = AIMessage(content=response_content)
    state["messages"] = state["messages"] + [ai_message]
    state["current_agent"] = "research_expert"
    state["agent_history"] = state.get("agent_history", []) + ["research_expert"]

    return state


def documentation_expert_response(
    state: CodeHeroWorkflowState,
) -> CodeHeroWorkflowState:
    """Documentation expert with comprehensive writing capabilities."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message:
        return state

    user_input = last_message.content
    handoff_context = state.get("handoff_context", {})

    # Enhanced context for documentation
    context = {
        "specialization": "Technical documentation",
        "expertise": [
            "API documentation",
            "User guides and tutorials",
            "Code documentation",
            "README files",
            "Architecture documentation",
            "Best practices guides",
        ],
        "handoff_context": handoff_context,
        "previous_agents": state.get("agent_history", []),
        **state.get("context", {}),
    }

    # Generate enhanced response
    response_content = create_agent_response(
        AgentRole.DOCUMENTATION, user_input, context
    )

    ai_message = AIMessage(content=response_content)
    state["messages"] = state["messages"] + [ai_message]
    state["current_agent"] = "documentation_expert"
    state["agent_history"] = state.get("agent_history", []) + ["documentation_expert"]

    return state


def review_expert_response(state: CodeHeroWorkflowState) -> CodeHeroWorkflowState:
    """Code review expert with quality assurance focus."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message:
        return state

    user_input = last_message.content
    handoff_context = state.get("handoff_context", {})

    # Enhanced context for code review
    context = {
        "specialization": "Code review and quality assurance",
        "focus_areas": [
            "Code quality and standards",
            "Security best practices",
            "Performance optimization",
            "Testing coverage",
            "Documentation completeness",
            "Maintainability assessment",
        ],
        "handoff_context": handoff_context,
        "previous_agents": state.get("agent_history", []),
        **state.get("context", {}),
    }

    # Generate enhanced response
    response_content = create_agent_response(
        AgentRole.CODE_REVIEWER, user_input, context
    )

    ai_message = AIMessage(content=response_content)
    state["messages"] = state["messages"] + [ai_message]
    state["current_agent"] = "review_expert"
    state["agent_history"] = state.get("agent_history", []) + ["review_expert"]

    return state


def code_expert_response(state: CodeHeroWorkflowState) -> CodeHeroWorkflowState:
    """General code expert for multi-language development."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message:
        return state

    user_input = last_message.content
    handoff_context = state.get("handoff_context", {})

    # Enhanced context for general coding
    context = {
        "specialization": "General code development",
        "languages": ["Python", "JavaScript", "TypeScript", "SQL"],
        "frameworks": ["FastAPI", "Next.js", "React", "LangChain"],
        "capabilities": [
            "Algorithm implementation",
            "Data structure design",
            "Code optimization",
            "Debugging assistance",
            "Architecture design",
        ],
        "handoff_context": handoff_context,
        "previous_agents": state.get("agent_history", []),
        **state.get("context", {}),
    }

    # Generate enhanced response
    response_content = create_agent_response(
        AgentRole.CODE_GENERATOR, user_input, context
    )

    ai_message = AIMessage(content=response_content)
    state["messages"] = state["messages"] + [ai_message]
    state["current_agent"] = "code_expert"
    state["agent_history"] = state.get("agent_history", []) + ["code_expert"]

    return state


# === ENHANCED WORKFLOW CREATION ===


def create_langgraph_workflow():
    """Create the enhanced LangGraph workflow with Command-based routing and industry-level coordination."""
    if not LANGGRAPH_AVAILABLE:
        return None

    # Create the main workflow state graph
    workflow = StateGraph(CodeHeroWorkflowState)

    # Add all agent nodes
    workflow.add_node("supervisor_response", supervisor_response)
    workflow.add_node("fastapi_expert", fastapi_expert_response)
    workflow.add_node("nextjs_expert", nextjs_expert_response)
    workflow.add_node("code_expert", code_expert_response)
    workflow.add_node("research_expert", research_expert_response)
    workflow.add_node("documentation_expert", documentation_expert_response)
    workflow.add_node("review_expert", review_expert_response)

    # Enhanced routing with Command support
    workflow.add_conditional_edges(
        START,
        supervisor_router,
        {
            "supervisor_response": "supervisor_response",
            "fastapi_expert": "fastapi_expert",
            "nextjs_expert": "nextjs_expert",
            "code_expert": "code_expert",
            "research_expert": "research_expert",
            "documentation_expert": "documentation_expert",
            "review_expert": "review_expert",
        },
    )

    # All expert paths lead to END for now
    # In advanced scenarios, we could add conditional handoffs
    workflow.add_edge("supervisor_response", END)
    workflow.add_edge("fastapi_expert", END)
    workflow.add_edge("nextjs_expert", END)
    workflow.add_edge("code_expert", END)
    workflow.add_edge("research_expert", END)
    workflow.add_edge("documentation_expert", END)
    workflow.add_edge("review_expert", END)

    # Compile with memory for state persistence
    memory = MemorySaver()
    compiled_workflow = workflow.compile(checkpointer=memory)

    return compiled_workflow


# === LEGACY WORKFLOW SUPPORT ===

# Keep existing workflow functions for backward compatibility
code_expert_workflow = None
research_expert_workflow = None
supervisor_workflow = None


def create_code_expert_workflow():
    """Legacy function - redirects to main workflow."""
    return create_langgraph_workflow()


def create_research_expert_workflow():
    """Legacy function - redirects to main workflow."""
    return create_langgraph_workflow()


def create_supervisor_workflow():
    """Legacy function - redirects to main workflow."""
    return create_langgraph_workflow()


# Export enhanced workflows
__all__ = [
    "CodeHeroWorkflowState",
    "AgentName",
    "supervisor_router",
    "create_agent_response",
    "create_langgraph_workflow",
    # Legacy exports
    "AgentResponse",
    "CodeGenerationRequest",
    "CodeGenerationResponse",
    "ResearchRequest",
    "ResearchResponse",
    "code_expert_workflow",
    "research_expert_workflow",
    "supervisor_workflow",
]
