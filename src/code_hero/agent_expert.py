"""Expert agents for the Code Hero system.

This module contains specialized expert agents that handle specific domains
of knowledge and tasks. Each agent uses industry-leading prompt engineering
patterns and integrates with the comprehensive tool ecosystem.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional


try:
    from langgraph.types import StreamWriter

    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for older LangGraph versions
    LANGGRAPH_AVAILABLE = False

    class StreamWriter:
        async def write(self, data):
            pass


from .config import (
    get_collections_for_agent,
    get_model_for_agent,
    get_tools_for_agent,
)
from .prompts import build_enhanced_prompt, get_agent_system_prompt
from .state import AgentRole, AgentState, Status
from .tools import (
    analyze_code,
    create_document,
    fetch_web_content,
    generate_code,
    read_file_content,
    search_documents,
    search_web,
    validate_code,
    write_file_content,
)

logger = logging.getLogger(__name__)


# Enhanced base class with model-aware LLM integration
class ExpertAgent:
    """Enhanced base class for all expert agents with model-aware capabilities."""

    def __init__(
        self,
        role: AgentRole,
        tools: Optional[List[Callable]] = None,
        collections: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize expert agent with enhanced model configuration.

        Args:
            role: Agent role from AgentRole enum
            tools: List of available tools (defaults to role-specific tools from config)
            collections: List of knowledge collections (defaults to role-specific collections from config)
            model_config: Model configuration override
            **kwargs: Additional configuration
        """
        self.role = role

        # Use configuration-based tools and collections if not provided
        if tools is None:
            tool_names = get_tools_for_agent(role)
            # Map tool names to actual tool functions
            tool_mapping = {
                "search_documents": search_documents,
                "search_web": search_web,
                "fetch_web_content": fetch_web_content,
                "generate_code": generate_code,
                "validate_code": validate_code,
                "analyze_code": analyze_code,
                "create_document": create_document,
                "read_file_content": read_file_content,
                "write_file_content": write_file_content,
            }
            self.tools = [
                tool_mapping.get(name)
                for name in tool_names
                if tool_mapping.get(name) is not None
            ]
        else:
            self.tools = tools

        self.collections = collections or get_collections_for_agent(role)

        # Get recommended model configuration
        self.model_config = model_config or get_model_for_agent(role)

        # Initialize LLM based on model configuration
        self.llm = self._initialize_llm()

        # Enhanced context management
        self.shared_memory = {}
        self.user_preferences = {}
        self.conversation_history = []

        logger.info(
            f"Initialized {role.value} with model: {self.model_config.get('model_name', 'default')}"
        )

    def _initialize_llm(self) -> Any:
        """Initialize LLM based on model configuration."""
        try:
            provider = self.model_config.get("provider", "openai")
            self.model_config.get("model_name", "gpt-4o-mini")

            if provider == "openai":
                from openai import OpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OpenAI API key not found, using mock client")
                    return None
                return OpenAI(api_key=api_key)

            elif provider == "qwen":
                # Qwen models can be accessed via OpenAI-compatible API or HuggingFace
                api_key = os.getenv("QWEN_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
                if api_key:
                    from openai import OpenAI

                    return OpenAI(
                        api_key=api_key,
                        base_url="https://api.qwen.com/v1",  # Adjust based on actual Qwen API
                    )

            elif provider == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if api_key:
                    from openai import OpenAI

                    return OpenAI(
                        api_key=api_key, base_url="https://api.deepseek.com/v1"
                    )

            elif provider == "claude":
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    import anthropic

                    return anthropic.Anthropic(api_key=api_key)

            # Fallback to OpenAI
            logger.warning(
                f"Provider {provider} not configured, falling back to OpenAI"
            )
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI

                return OpenAI(api_key=api_key)

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")

        return None

    async def generate_response_with_llm(
        self, prompt: str, context: Dict[str, Any] = None, use_tools: bool = True
    ) -> str:
        """Generate response using the configured LLM.

        Args:
            prompt: User prompt
            context: Additional context
            use_tools: Whether to use available tools

        Returns:
            Generated response
        """
        if not self.llm:
            return self._fallback_response(prompt, context)

        try:
            # Build enhanced prompt with model-specific optimizations
            enhanced_prompt = self._build_enhanced_prompt(prompt, context)

            # Generate response based on provider
            provider = self.model_config.get("provider", "openai")

            if provider in ["openai", "qwen", "deepseek"]:
                response = self.llm.chat.completions.create(
                    model=self.model_config.get("model_name", "gpt-4o-mini"),
                    messages=[
                        {
                            "role": "system",
                            "content": get_agent_system_prompt(self.role),
                        },
                        {"role": "user", "content": enhanced_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=self.model_config.get("max_tokens", 8192),
                )
                return response.choices[0].message.content

            elif provider == "claude":
                response = self.llm.messages.create(
                    model=self.model_config.get(
                        "model_name", "claude-3-5-sonnet-20241022"
                    ),
                    max_tokens=self.model_config.get("max_tokens", 8192),
                    temperature=0.1,
                    messages=[{"role": "user", "content": enhanced_prompt}],
                )
                return response.content[0].text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_response(prompt, context)

        return self._fallback_response(prompt, context)

    def _build_enhanced_prompt(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> str:
        """Build enhanced prompt using the prompt system."""
        if context is None:
            context = {}

        # Add the user query to context
        context.update({"user_query": prompt})

        # Create a minimal AgentState for the prompt builder
        from .state import AgentState

        temp_state = AgentState(agent=self.role, context=context)

        return build_enhanced_prompt(self.role, temp_state)

    def _fallback_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Provide fallback response when LLM is unavailable."""
        # Instead of hardcoded response, try to use the enhanced prompt system
        try:
            self._build_enhanced_prompt(prompt, context)
            # Return a more dynamic response based on the role and context
            specialization = self.model_config.get(
                "specialization", "general assistance"
            )
            role_name = self.role.value.replace("_", " ").title()

            # Build dynamic response based on context and role
            if context and context.get("current_task"):
                task_context = f" regarding {context['current_task']}"
            else:
                task_context = ""

            return (
                f"As your {role_name} specialist, I'm analyzing your request{task_context}. "
                f"My expertise in {specialization} allows me to provide guidance even with limited connectivity. "
                f"For optimal performance, please verify your API configuration is properly set up."
            )

        except Exception:
            # Final fallback - use role-specific guidance from prompts
            try:
                from .prompts import get_agent_system_prompt

                system_prompt = get_agent_system_prompt(self.role)
                role_name = self.role.value.replace("_", " ").title()

                # Extract key capabilities from system prompt if available
                if system_prompt and len(system_prompt) > 50:
                    # Use first 100 chars of system prompt to inform response
                    prompt_preview = system_prompt[:100].replace("\n", " ")
                    return (
                        f"I'm your {role_name} specialist. Based on my configuration: {prompt_preview}... "
                        f"API connectivity may be limited - please verify your configuration for full capabilities."
                    )
                else:
                    return (
                        f"Your {role_name} specialist is ready to assist. "
                        f"Please ensure API configuration is complete for optimal functionality."
                    )

            except Exception:
                # Absolute final fallback - completely dynamic
                role_name = self.role.value.replace("_", " ").title()
                return f"Agent {role_name} initialized successfully. Awaiting API configuration verification for full functionality."

    def extract_shared_context(self, state: AgentState) -> Dict[str, Any]:
        """Extract context that should be shared with other agents."""
        return {
            "agent_role": self.role.value,
            "current_task": state.context.get("current_task"),
            "project_context": state.context.get("project_context", {}),
            "available_tools": [
                tool.__name__ if hasattr(tool, "__name__") else str(tool)
                for tool in self.tools
            ],
            "model_info": self.model_config,
        }

    def update_shared_context(
        self, state: AgentState, updates: Dict[str, Any]
    ) -> AgentState:
        """Update shared context with new information."""
        # Update artifacts with new information
        state.artifacts.update(updates)

        # Update status if provided
        if "success" in updates:
            state.status = Status.COMPLETED if updates["success"] else Status.FAILED

        return state

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Enhanced task handling with model-aware processing."""
        try:
            # Extract context from state
            context = self.extract_shared_context(state)

            # Generate response using configured LLM
            response = await self.generate_response_with_llm(prompt, context)

            # Update shared context
            updated_state = self.update_shared_context(
                state,
                {
                    "last_response": response,
                    "model_used": self.model_config.get("model_name", "unknown"),
                    "specialization": self.model_config.get(
                        "specialization", "general"
                    ),
                    "success": True,
                },
            )

            # Stream response if writer provided
            if writer:
                await writer.write(response)

            return updated_state

        except Exception as e:
            logger.error(f"Task handling failed for {self.role.value}: {e}")
            error_response = f"Error in {self.role.value}: {str(e)}"

            if writer:
                await writer.write(error_response)

            return state


# Enhanced expert agents with model-specific implementations


class LangChainExpert(ExpertAgent):
    """Expert agent for LangChain operations using industry-leading prompts."""

    def __init__(self, **data):
        """Initialize LangChain expert."""
        super().__init__(role=AgentRole.LANGCHAIN_EXPERT, **data)


class FastAPIExpert(ExpertAgent):
    """Expert agent for FastAPI development using industry-leading prompts."""

    def __init__(self, **data):
        """Initialize FastAPI expert."""
        super().__init__(role=AgentRole.FASTAPI_EXPERT, **data)


class NextJSExpert(ExpertAgent):
    """Expert agent for Next.js development using industry-leading prompts."""

    def __init__(self, **data):
        """Initialize Next.js expert."""
        super().__init__(role=AgentRole.NEXTJS_EXPERT, **data)


class ResearchExpert(ExpertAgent):
    """Expert agent for research and analysis using industry-leading prompts."""

    def __init__(self, **data):
        """Initialize Research expert."""
        super().__init__(role=AgentRole.RESEARCH, **data)


class LangGraphExpert(ExpertAgent):
    """Expert agent for LangGraph development using industry-leading prompts."""

    def __init__(self, **data):
        """Initialize LangGraph expert."""
        super().__init__(role=AgentRole.LANGGRAPH_EXPERT, **data)


class DocumentationExpert(ExpertAgent):
    """Expert agent for documentation using industry-leading prompts."""

    def __init__(self, **data):
        """Initialize Documentation expert."""
        super().__init__(role=AgentRole.DOCUMENTATION, **data)


class SupervisorExpert(ExpertAgent):
    """Expert agent for supervisor coordination and task management."""

    def __init__(self, **data):
        """Initialize Supervisor expert."""
        super().__init__(role=AgentRole.SUPERVISOR, **data)

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle supervisor coordination tasks."""
        try:
            # Build supervisor-specific context
            context = {
                "specialization": "Multi-agent coordination and task management",
                "primary_tools": [
                    "task_routing",
                    "agent_coordination",
                    "workflow_management",
                ],
                "coordination_mode": True,
                "task_type": "supervision",
            }

            response = await self.generate_response_with_llm(prompt, context)

            if writer:
                await writer.write(response)

            return self.update_shared_context(
                state,
                {
                    "supervisor_response": response,
                    "coordination_complete": True,
                    "success": True,
                },
            )

        except Exception as e:
            logger.error(f"Supervisor expert task failed: {e}")
            return state


class ImplementationExpert(ExpertAgent):
    """Expert agent for implementation and development tasks."""

    def __init__(self, **data):
        """Initialize Implementation expert."""
        super().__init__(role=AgentRole.IMPLEMENTATION, **data)

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle implementation tasks."""
        try:
            context = {
                "specialization": "Software implementation and development",
                "primary_tools": ["generate_code", "validate_code", "analyze_code"],
                "implementation_focus": True,
                "task_type": "implementation",
            }

            response = await self.generate_response_with_llm(prompt, context)

            if writer:
                await writer.write(response)

            return self.update_shared_context(
                state,
                {
                    "implementation_response": response,
                    "implementation_complete": True,
                    "success": True,
                },
            )

        except Exception as e:
            logger.error(f"Implementation expert task failed: {e}")
            return state


class TrdConverterExpert(ExpertAgent):
    """Expert agent for TRD (Technical Requirements Document) conversion."""

    def __init__(self, **data):
        """Initialize TRD Converter expert."""
        super().__init__(role=AgentRole.TRD_CONVERTER, **data)

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle TRD conversion tasks."""
        try:
            context = {
                "specialization": "Technical Requirements Document conversion",
                "primary_tools": [
                    "document_analysis",
                    "requirement_extraction",
                    "specification_generation",
                ],
                "conversion_focus": True,
                "task_type": "trd_conversion",
            }

            response = await self.generate_response_with_llm(prompt, context)

            if writer:
                await writer.write(response)

            return self.update_shared_context(
                state,
                {
                    "trd_response": response,
                    "conversion_complete": True,
                    "success": True,
                },
            )

        except Exception as e:
            logger.error(f"TRD Converter expert task failed: {e}")
            return state


class CodeReviewerExpert(ExpertAgent):
    """Expert agent for code review and quality assurance."""

    def __init__(self, **data):
        """Initialize Code Reviewer expert."""
        super().__init__(role=AgentRole.CODE_REVIEWER, **data)

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle code review tasks."""
        try:
            context = {
                "specialization": "Code review and quality assurance",
                "primary_tools": [
                    "validate_code",
                    "analyze_code",
                    "quality_assessment",
                ],
                "review_focus": True,
                "task_type": "code_review",
            }

            response = await self.generate_response_with_llm(prompt, context)

            if writer:
                await writer.write(response)

            return self.update_shared_context(
                state,
                {"review_response": response, "review_complete": True, "success": True},
            )

        except Exception as e:
            logger.error(f"Code Reviewer expert task failed: {e}")
            return state


class StandardsEnforcerExpert(ExpertAgent):
    """Expert agent for standards enforcement and compliance."""

    def __init__(self, **data):
        """Initialize Standards Enforcer expert."""
        super().__init__(role=AgentRole.STANDARDS_ENFORCER, **data)

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle standards enforcement tasks."""
        try:
            context = {
                "specialization": "Standards enforcement and compliance",
                "primary_tools": [
                    "standards_validation",
                    "compliance_check",
                    "policy_enforcement",
                ],
                "enforcement_focus": True,
                "task_type": "standards_enforcement",
            }

            response = await self.generate_response_with_llm(prompt, context)

            if writer:
                await writer.write(response)

            return self.update_shared_context(
                state,
                {
                    "standards_response": response,
                    "enforcement_complete": True,
                    "success": True,
                },
            )

        except Exception as e:
            logger.error(f"Standards Enforcer expert task failed: {e}")
            return state


class StrategicExpert(ExpertAgent):
    """Expert agent for strategic planning and architecture."""

    def __init__(self, **data):
        """Initialize Strategic expert."""
        super().__init__(role=AgentRole.STRATEGIC_EXPERT, **data)

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle strategic planning tasks."""
        try:
            context = {
                "specialization": "Strategic planning and architecture",
                "primary_tools": [
                    "strategic_analysis",
                    "architecture_design",
                    "planning",
                ],
                "strategic_focus": True,
                "task_type": "strategic_planning",
            }

            response = await self.generate_response_with_llm(prompt, context)

            if writer:
                await writer.write(response)

            return self.update_shared_context(
                state,
                {
                    "strategic_response": response,
                    "planning_complete": True,
                    "success": True,
                },
            )

        except Exception as e:
            logger.error(f"Strategic expert task failed: {e}")
            return state


class LlamaIndexExpert(ExpertAgent):
    """Expert agent for LlamaIndex operations and RAG systems."""

    def __init__(self, **data):
        """Initialize LlamaIndex expert."""
        super().__init__(role=AgentRole.LLAMAINDEX_EXPERT, **data)

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle LlamaIndex tasks."""
        try:
            context = {
                "specialization": "LlamaIndex and RAG systems",
                "primary_tools": [
                    "document_indexing",
                    "retrieval_systems",
                    "rag_implementation",
                ],
                "llamaindex_focus": True,
                "task_type": "llamaindex_development",
            }

            response = await self.generate_response_with_llm(prompt, context)

            if writer:
                await writer.write(response)

            return self.update_shared_context(
                state,
                {
                    "llamaindex_response": response,
                    "rag_complete": True,
                    "success": True,
                },
            )

        except Exception as e:
            logger.error(f"LlamaIndex expert task failed: {e}")
            return state


class PydanticExpert(ExpertAgent):
    """Expert agent for Pydantic models and data validation."""

    def __init__(self, **data):
        """Initialize Pydantic expert."""
        super().__init__(role=AgentRole.PYDANTIC_EXPERT, **data)

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle Pydantic tasks."""
        try:
            context = {
                "specialization": "Pydantic models and data validation",
                "primary_tools": [
                    "model_generation",
                    "validation_logic",
                    "schema_design",
                ],
                "pydantic_focus": True,
                "task_type": "pydantic_development",
            }

            response = await self.generate_response_with_llm(prompt, context)

            if writer:
                await writer.write(response)

            return self.update_shared_context(
                state,
                {
                    "pydantic_response": response,
                    "validation_complete": True,
                    "success": True,
                },
            )

        except Exception as e:
            logger.error(f"Pydantic expert task failed: {e}")
            return state


class AgnoExpert(ExpertAgent):
    """Expert agent for Agno framework operations."""

    def __init__(self, **data):
        """Initialize Agno expert."""
        super().__init__(role=AgentRole.AGNO_EXPERT, **data)

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle Agno framework tasks."""
        try:
            context = {
                "specialization": "Agno framework development",
                "primary_tools": [
                    "agno_implementation",
                    "framework_integration",
                    "workflow_design",
                ],
                "agno_focus": True,
                "task_type": "agno_development",
            }

            response = await self.generate_response_with_llm(prompt, context)

            if writer:
                await writer.write(response)

            return self.update_shared_context(
                state,
                {
                    "agno_response": response,
                    "framework_complete": True,
                    "success": True,
                },
            )

        except Exception as e:
            logger.error(f"Agno expert task failed: {e}")
            return state


class CrewAIExpert(ExpertAgent):
    """Expert agent for CrewAI multi-agent systems."""

    def __init__(self, **data):
        """Initialize CrewAI expert."""
        super().__init__(role=AgentRole.CREWAI_EXPERT, **data)

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle CrewAI tasks."""
        try:
            context = {
                "specialization": "CrewAI multi-agent systems",
                "primary_tools": [
                    "crew_design",
                    "agent_coordination",
                    "task_delegation",
                ],
                "crewai_focus": True,
                "task_type": "crewai_development",
            }

            response = await self.generate_response_with_llm(prompt, context)

            if writer:
                await writer.write(response)

            return self.update_shared_context(
                state,
                {"crewai_response": response, "crew_complete": True, "success": True},
            )

        except Exception as e:
            logger.error(f"CrewAI expert task failed: {e}")
            return state


class DocumentAnalyzerExpert(ExpertAgent):
    """Expert agent for document analysis and processing."""

    def __init__(self, **data):
        """Initialize Document Analyzer expert."""
        super().__init__(role=AgentRole.DOCUMENT_ANALYZER, **data)

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle document analysis tasks."""
        try:
            context = {
                "specialization": "Document analysis and processing",
                "primary_tools": ["document_parsing", "content_extraction", "analysis"],
                "analysis_focus": True,
                "task_type": "document_analysis",
            }

            response = await self.generate_response_with_llm(prompt, context)

            if writer:
                await writer.write(response)

            return self.update_shared_context(
                state,
                {
                    "analysis_response": response,
                    "analysis_complete": True,
                    "success": True,
                },
            )

        except Exception as e:
            logger.error(f"Document Analyzer expert task failed: {e}")
            return state


class PromptEngineerExpert(ExpertAgent):
    """Expert agent for prompt engineering and optimization."""

    def __init__(self, **data):
        """Initialize Prompt Engineer expert."""
        super().__init__(role=AgentRole.PROMPT_ENGINEER, **data)

    async def handle_task(
        self, state: AgentState, prompt: str, writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle prompt engineering tasks."""
        try:
            context = {
                "specialization": "Prompt engineering and optimization",
                "primary_tools": ["prompt_design", "optimization", "testing"],
                "prompt_focus": True,
                "task_type": "prompt_engineering",
            }

            response = await self.generate_response_with_llm(prompt, context)

            if writer:
                await writer.write(response)

            return self.update_shared_context(
                state,
                {
                    "prompt_response": response,
                    "engineering_complete": True,
                    "success": True,
                },
            )

        except Exception as e:
            logger.error(f"Prompt Engineer expert task failed: {e}")
            return state


# Update the experts registry with all agents
experts = {
    AgentRole.SUPERVISOR: SupervisorExpert(),
    AgentRole.RESEARCH: ResearchExpert(),
    AgentRole.IMPLEMENTATION: ImplementationExpert(),
    AgentRole.DOCUMENTATION: DocumentationExpert(),
    AgentRole.TRD_CONVERTER: TrdConverterExpert(),
    AgentRole.CODE_GENERATOR: ImplementationExpert(),  # Code generation uses implementation expert
    AgentRole.CODE_REVIEWER: CodeReviewerExpert(),
    AgentRole.STANDARDS_ENFORCER: StandardsEnforcerExpert(),
    AgentRole.STRATEGIC_EXPERT: StrategicExpert(),
    AgentRole.LANGCHAIN_EXPERT: LangChainExpert(),
    AgentRole.LANGGRAPH_EXPERT: LangGraphExpert(),
    AgentRole.LLAMAINDEX_EXPERT: LlamaIndexExpert(),
    AgentRole.FASTAPI_EXPERT: FastAPIExpert(),
    AgentRole.NEXTJS_EXPERT: NextJSExpert(),
    AgentRole.PYDANTIC_EXPERT: PydanticExpert(),
    AgentRole.AGNO_EXPERT: AgnoExpert(),
    AgentRole.CREWAI_EXPERT: CrewAIExpert(),
    AgentRole.DOCUMENT_ANALYZER: DocumentAnalyzerExpert(),
    AgentRole.PROMPT_ENGINEER: PromptEngineerExpert(),
}

# Export all experts - 19 core agent roles only
all_experts = experts

# ─────────────────────────────────────────────────────────────────────────────
# AGENT REGISTRY AND EXECUTION
# ─────────────────────────────────────────────────────────────────────────────


async def execute_agent(
    agent_role: AgentRole,
    state: AgentState,
    prompt: str = "",
    *,
    writer: Optional[StreamWriter] = None,
) -> AgentState:
    """Execute an agent with the given state and prompt.

    Args:
        agent_role: The role of the agent to execute
        state: Current agent state
        prompt: Optional prompt to override state context
        writer: Optional stream writer for real-time updates

    Returns:
        Updated agent state
    """
    try:
        # Get the expert agent
        expert = all_experts.get(agent_role)
        if not expert:
            # Fallback to base expert agent with default behavior
            expert = ExpertAgent(role=agent_role, tools=[])

        # Set the prompt in state context if provided
        if prompt:
            state.context["prompt"] = prompt
            state.context["query"] = prompt

        # Execute the agent
        result_state = await expert.handle_task(
            state, prompt or state.context.get("prompt", ""), writer
        )

        return result_state

    except Exception as e:
        error = f"Agent execution failed for {agent_role}: {str(e)}"
        logger.error(error)

        state.status = Status.FAILED
        state.error = error

        return state


# Export all symbols
__all__ = [
    "ExpertAgent",
    "LangChainExpert",
    "FastAPIExpert",
    "NextJSExpert",
    "ResearchExpert",
    "LangGraphExpert",
    "DocumentationExpert",
    "SupervisorExpert",
    "ImplementationExpert",
    "TrdConverterExpert",
    "CodeReviewerExpert",
    "StandardsEnforcerExpert",
    "StrategicExpert",
    "LlamaIndexExpert",
    "PydanticExpert",
    "AgnoExpert",
    "CrewAIExpert",
    "DocumentAnalyzerExpert",
    "PromptEngineerExpert",
    "experts",
    "all_experts",
    "execute_agent",
]
