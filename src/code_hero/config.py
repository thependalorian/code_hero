"""Configuration management for the backend system.

This module handles all configuration-related functionality including
environment variables, API keys, database settings, and agent configurations.
All configuration models are now consolidated in state.py.
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# Import prompt system for enhanced agent configurations
from .prompts import CORE_SYSTEM_INSTRUCTIONS, build_enhanced_prompt

# Import all configuration models from consolidated state
from .state import (
    AgentConfig,
    AgentRole,
    APIConfig,
    AstraConfig,
    AstraDBCollection,
    ConfigState,
    DatabaseConfig,
    DeepseekConfig,
    GroqConfig,
    LLMConfig,
    LLMRegistry,
    LoggingConfig,
    OpenAIConfig,
    TeamConfig,
    WorkflowConfig,
)

# Enhanced model configurations for available LLMs
MODEL_CONFIGURATIONS = {
    # OpenAI Models - Industry standard
    "openai": {
        "gpt-4o": {
            "model_name": "gpt-4o",
            "specialization": "Multimodal reasoning and coding",
            "context_length": 128000,
            "strengths": ["reasoning", "coding", "multimodal", "reliability"],
            "use_cases": [
                "enterprise_development",
                "complex_tasks",
                "production_systems",
            ],
            "temperature": 0.1,
            "max_tokens": 8192,
        },
        "gpt-4o-mini": {
            "model_name": "gpt-4o-mini",
            "specialization": "Efficient version of GPT-4o",
            "context_length": 128000,
            "strengths": ["cost_efficiency", "speed", "coding", "general_purpose"],
            "use_cases": [
                "development_assistance",
                "rapid_prototyping",
                "cost_sensitive_tasks",
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        "gpt-4-turbo": {
            "model_name": "gpt-4-turbo-preview",
            "specialization": "Advanced reasoning and complex problem solving",
            "context_length": 128000,
            "strengths": ["reasoning", "analysis", "complex_tasks", "reliability"],
            "use_cases": ["strategic_planning", "research", "complex_analysis"],
            "temperature": 0.1,
            "max_tokens": 4096,
        },
    },
    # DeepSeek Models - Strong coding performance
    "deepseek": {
        "deepseek-chat": {
            "model_name": "deepseek-chat",
            "specialization": "General purpose with strong coding capabilities",
            "context_length": 32000,
            "strengths": ["coding", "reasoning", "cost_effective", "general_purpose"],
            "use_cases": ["code_generation", "implementation", "general_development"],
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        "deepseek-coder": {
            "model_name": "deepseek-coder",
            "specialization": "Specialized coding model",
            "context_length": 16000,
            "strengths": [
                "code_generation",
                "debugging",
                "code_review",
                "optimization",
            ],
            "use_cases": ["code_generation", "code_review", "implementation"],
            "temperature": 0.1,
            "max_tokens": 4096,
        },
    },
    # Groq Models - Fast inference
    "groq": {
        "mixtral-8x7b": {
            "model_name": "mixtral-8x7b-32768",
            "specialization": "Fast inference with good reasoning",
            "context_length": 32768,
            "strengths": ["speed", "reasoning", "multilingual", "cost_effective"],
            "use_cases": ["rapid_prototyping", "real_time_assistance", "general_tasks"],
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        "llama3-70b": {
            "model_name": "llama3-70b-8192",
            "specialization": "Large model with strong capabilities",
            "context_length": 8192,
            "strengths": ["reasoning", "coding", "analysis", "general_purpose"],
            "use_cases": ["complex_reasoning", "analysis", "strategic_planning"],
            "temperature": 0.1,
            "max_tokens": 4096,
        },
    },
    # Mistral Models - Including Devstral for coding agents
    "mistral": {
        "devstral-small": {
            "model_name": "devstral-small-2505",
            "specialization": "Agentic LLM for software engineering tasks",
            "context_length": 128000,
            "parameters": "24B",
            "strengths": [
                "agentic_coding",
                "multi_file_editing",
                "codebase_exploration",
                "real_world_software_engineering",
                "swe_bench_optimized",
                "python_expert",
                "local_deployment",
            ],
            "use_cases": [
                "software_engineering_agents",
                "multi_file_code_editing",
                "github_issue_resolution",
                "codebase_navigation",
                "python_development",
                "local_coding_assistant",
            ],
            "temperature": 0.1,
            "max_tokens": 8192,
            "api_endpoint": "https://api.mistral.ai/v1",
            "pricing": {
                "input_tokens": 0.1,  # $0.1/M tokens
                "output_tokens": 0.3,  # $0.3/M tokens
            },
            "benchmark_scores": {
                "swe_bench_verified": 46.8,
                "outperforms": ["deepseek-v3-0324", "qwen3-232b", "gpt-4.1-mini"],
            },
            "deployment_options": [
                "mistral_api",
                "huggingface",
                "ollama",
                "local_deployment",
            ],
            "hardware_requirements": {
                "minimum": "RTX 4090 or Mac 32GB RAM",
                "recommended": "High-end consumer hardware",
            },
        },
        "codestral": {
            "model_name": "codestral-22b",
            "specialization": "Code generation and completion",
            "context_length": 32000,
            "parameters": "22B",
            "strengths": [
                "code_completion",
                "multi_language_support",
                "ide_integration",
                "fast_inference",
            ],
            "use_cases": [
                "code_completion",
                "ide_plugins",
                "rapid_development",
                "multi_language_coding",
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
            "supported_languages": 80,
        },
    },
}

# Agent-specific model recommendations based on available models
AGENT_MODEL_RECOMMENDATIONS = {
    AgentRole.SUPERVISOR: {
        "primary": "gpt-4o",
        "alternatives": ["gpt-4-turbo", "llama3-70b"],
        "reasoning": "Needs strong reasoning for coordination and decision making",
    },
    AgentRole.RESEARCH: {
        "primary": "gpt-4-turbo",
        "alternatives": ["gpt-4o", "llama3-70b"],
        "reasoning": "Requires deep analysis and comprehensive reasoning capabilities",
    },
    AgentRole.IMPLEMENTATION: {
        "primary": "deepseek-chat",
        "alternatives": ["devstral-small", "gpt-4o", "deepseek-coder"],
        "reasoning": "Strong coding capabilities with cost efficiency, Devstral excellent for agentic software engineering",
    },
    AgentRole.CODE_GENERATOR: {
        "primary": "deepseek-coder",
        "alternatives": ["devstral-small", "deepseek-chat", "gpt-4o"],
        "reasoning": "Specialized for code generation and optimization, Devstral ideal for software engineering agents",
    },
    AgentRole.CODE_REVIEWER: {
        "primary": "gpt-4o",
        "alternatives": ["devstral-small", "deepseek-coder", "gpt-4-turbo"],
        "reasoning": "Comprehensive analysis for quality assessment, Devstral excellent for agentic code review",
    },
    AgentRole.FASTAPI_EXPERT: {
        "primary": "gpt-4o",
        "alternatives": ["devstral-small", "deepseek-coder", "deepseek-chat"],
        "reasoning": "Strong Python and API development capabilities, Devstral excels at Python frameworks",
    },
    AgentRole.NEXTJS_EXPERT: {
        "primary": "gpt-4o",
        "alternatives": ["deepseek-chat", "devstral-small", "gpt-4-turbo"],
        "reasoning": "Complex frontend frameworks require strong reasoning, Devstral good for implementation",
    },
    AgentRole.LANGCHAIN_EXPERT: {
        "primary": "gpt-4o",
        "alternatives": ["gpt-4-turbo", "devstral-small", "deepseek-chat"],
        "reasoning": "Complex framework integration requires advanced reasoning, Devstral excellent for implementation",
    },
    AgentRole.LANGGRAPH_EXPERT: {
        "primary": "gpt-4-turbo",
        "alternatives": ["gpt-4o", "devstral-small", "llama3-70b"],
        "reasoning": "Graph-based reasoning requires advanced analytical capabilities, Devstral for implementation",
    },
    AgentRole.STRATEGIC_EXPERT: {
        "primary": "gpt-4-turbo",
        "alternatives": ["gpt-4o", "llama3-70b"],
        "reasoning": "Strategic planning requires deep analytical thinking",
    },
    AgentRole.DOCUMENTATION: {
        "primary": "gpt-4o",
        "alternatives": ["gpt-4-turbo", "mixtral-8x7b"],
        "reasoning": "Clear communication and comprehensive coverage needed",
    },
    AgentRole.STANDARDS_ENFORCER: {
        "primary": "gpt-4-turbo",
        "alternatives": ["devstral-small", "gpt-4o", "llama3-70b"],
        "reasoning": "Thorough analysis and attention to detail, Devstral ideal for code standards enforcement",
    },
    AgentRole.TRD_CONVERTER: {
        "primary": "gpt-4o",
        "alternatives": ["gpt-4-turbo", "devstral-small", "deepseek-chat"],
        "reasoning": "Document analysis and conversion requires strong comprehension, Devstral for implementation",
    },
    AgentRole.LLAMAINDEX_EXPERT: {
        "primary": "gpt-4o",
        "alternatives": ["devstral-small", "deepseek-chat", "gpt-4-turbo"],
        "reasoning": "RAG systems require understanding of complex architectures, Devstral for Python implementation",
    },
    AgentRole.PYDANTIC_EXPERT: {
        "primary": "deepseek-chat",
        "alternatives": ["devstral-small", "deepseek-coder", "gpt-4o"],
        "reasoning": "Python-specific expertise with strong typing knowledge, Devstral excellent for Python validation",
    },
    AgentRole.AGNO_EXPERT: {
        "primary": "gpt-4o",
        "alternatives": ["devstral-small", "gpt-4-turbo", "deepseek-chat"],
        "reasoning": "Framework expertise requires comprehensive understanding, Devstral perfect for agentic implementation",
    },
    AgentRole.CREWAI_EXPERT: {
        "primary": "gpt-4o",
        "alternatives": ["gpt-4-turbo", "devstral-small", "llama3-70b"],
        "reasoning": "Multi-agent coordination requires advanced reasoning, Devstral excellent for implementation",
    },
    AgentRole.DOCUMENT_ANALYZER: {
        "primary": "gpt-4-turbo",
        "alternatives": ["gpt-4o", "llama3-70b"],
        "reasoning": "Document analysis requires deep comprehension and reasoning",
    },
    AgentRole.PROMPT_ENGINEER: {
        "primary": "gpt-4o",
        "alternatives": ["gpt-4-turbo", "mixtral-8x7b"],
        "reasoning": "Prompt engineering requires understanding of LLM behavior",
    },
}

# Load environment variables
load_dotenv()


def get_enhanced_system_prompt(role: AgentRole, context: Dict[str, Any] = None) -> str:
    """Get enhanced system prompt for an agent role.

    Args:
        role: Agent role
        context: Additional context for prompt building

    Returns:
        Enhanced system prompt using industry-leading patterns
    """
    from .state import AgentState

    # Create minimal agent state for prompt building
    agent_state = AgentState(agent=role, context=context or {})

    try:
        return build_enhanced_prompt(role, agent_state)
    except Exception:
        # Fallback to core instructions if prompt building fails
        return CORE_SYSTEM_INSTRUCTIONS


def get_research_team_config() -> TeamConfig:
    """Get research team configuration."""
    return TeamConfig(
        name="Research Team",
        description="Team focused on research and analysis tasks",
        agents=[
            AgentConfig(
                name="Research Agent",
                role=AgentRole.RESEARCH,
                system_prompt=get_enhanced_system_prompt(
                    AgentRole.RESEARCH,
                    {
                        "specialization": "research and analysis",
                        "primary_tools": [
                            "search_documents",
                            "search_web",
                            "fetch_web_content",
                        ],
                    },
                ),
                tools=["search_documents", "search_web", "fetch_web_content"],
                collections=["strategy_book", "langchain_docs", "langgraph_docs"],
            ),
            AgentConfig(
                name="Document Analyzer",
                role=AgentRole.DOCUMENT_ANALYZER,
                system_prompt=get_enhanced_system_prompt(
                    AgentRole.DOCUMENT_ANALYZER,
                    {
                        "specialization": "document analysis and insight extraction",
                        "primary_tools": ["search_documents", "analyze_code"],
                    },
                ),
                tools=["search_documents", "analyze_code"],
                collections=["strategy_book", "langchain_docs"],
            ),
        ],
    )


def get_implementation_team_config() -> TeamConfig:
    """Get implementation team configuration."""
    return TeamConfig(
        name="Implementation Team",
        description="Team focused on code generation and implementation",
        agents=[
            AgentConfig(
                name="Code Generator",
                role=AgentRole.CODE_GENERATOR,
                system_prompt=get_enhanced_system_prompt(
                    AgentRole.CODE_GENERATOR,
                    {
                        "specialization": "high-quality code generation",
                        "primary_tools": [
                            "generate_code",
                            "validate_code",
                            "analyze_code",
                        ],
                    },
                ),
                tools=["generate_code", "validate_code", "analyze_code"],
                collections=["langchain_docs", "fastapi_docs", "nextjs_docs"],
            ),
            AgentConfig(
                name="FastAPI Expert",
                role=AgentRole.FASTAPI_EXPERT,
                system_prompt=get_enhanced_system_prompt(
                    AgentRole.FASTAPI_EXPERT,
                    {
                        "specialization": "FastAPI backend development",
                        "primary_tools": ["generate_code", "validate_code"],
                    },
                ),
                tools=["generate_code", "validate_code"],
                collections=["fastapi_docs", "pydantic_docs"],
            ),
            AgentConfig(
                name="Next.js Expert",
                role=AgentRole.NEXTJS_EXPERT,
                system_prompt=get_enhanced_system_prompt(
                    AgentRole.NEXTJS_EXPERT,
                    {
                        "specialization": "Next.js frontend development",
                        "primary_tools": ["generate_code", "validate_code"],
                    },
                ),
                tools=["generate_code", "validate_code"],
                collections=["nextjs_docs"],
            ),
        ],
    )


def get_documentation_team_config() -> TeamConfig:
    """Get documentation team configuration."""
    return TeamConfig(
        name="Documentation Team",
        description="Team focused on documentation and standards",
        agents=[
            AgentConfig(
                name="Documentation Agent",
                role=AgentRole.DOCUMENTATION,
                system_prompt=get_enhanced_system_prompt(
                    AgentRole.DOCUMENTATION,
                    {
                        "specialization": "comprehensive documentation creation",
                        "primary_tools": ["create_document", "search_documents"],
                    },
                ),
                tools=["create_document", "search_documents"],
                collections=["strategy_book"],
            ),
            AgentConfig(
                name="Standards Enforcer",
                role=AgentRole.STANDARDS_ENFORCER,
                system_prompt=get_enhanced_system_prompt(
                    AgentRole.STANDARDS_ENFORCER,
                    {
                        "specialization": "code quality and compliance enforcement",
                        "primary_tools": ["validate_code", "analyze_code"],
                    },
                ),
                tools=["validate_code", "analyze_code"],
                collections=["strategy_book"],
            ),
        ],
    )


def get_all_team_configs() -> List[TeamConfig]:
    """Get all team configurations."""
    return [
        get_research_team_config(),
        get_implementation_team_config(),
        get_documentation_team_config(),
    ]


# Enhanced LLM configurations for various models
class QwenConfig(BaseModel):
    """Qwen model configuration."""

    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("QWEN_API_KEY", ""))
    model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 8192
    context_length: int = 128000
    supports_languages: int = 92
    specialization: str = "Advanced coding and reasoning"


class DeepSeekConfig(BaseModel):
    """DeepSeek model configuration."""

    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    model: str = "deepseek-ai/DeepSeek-Coder-V2-Instruct"
    temperature: float = 0.1
    max_tokens: int = 8192
    context_length: int = 128000
    supports_languages: int = 338
    specialization: str = "Open-source coding excellence"


class ClaudeConfig(BaseModel):
    """Claude model configuration."""

    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.1
    max_tokens: int = 8192
    context_length: int = 200000
    specialization: str = "Advanced reasoning and safety"


class GeminiConfig(BaseModel):
    """Gemini model configuration."""

    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model: str = "gemini-2.5-pro"
    temperature: float = 0.1
    max_tokens: int = 8192
    context_length: int = 1000000
    specialization: str = "Multimodal reasoning and long context"


class MistralConfig(BaseModel):
    """Mistral model configuration including Devstral."""

    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", ""))
    model: str = "devstral-small-2505"
    temperature: float = 0.1
    max_tokens: int = 8192
    context_length: int = 128000
    parameters: str = "24B"
    specialization: str = "Agentic LLM for software engineering tasks"

    # Devstral-specific configuration
    devstral_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "swe_bench_score": 46.8,
            "agentic_capabilities": True,
            "multi_file_editing": True,
            "codebase_exploration": True,
            "local_deployment": True,
            "hardware_requirements": "RTX 4090 or Mac 32GB RAM",
            "supported_frameworks": ["OpenHands", "SWE-Agent"],
            "license": "Apache 2.0",
            "deployment_options": ["API", "HuggingFace", "Ollama", "Local"],
        }
    )

    # Alternative Mistral models
    codestral_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "model_name": "codestral-22b",
            "parameters": "22B",
            "context_length": 32000,
            "supported_languages": 80,
            "specialization": "Code generation and completion",
        }
    )


# Enhanced LLM Registry with multiple model providers
class EnhancedLLMRegistry(BaseModel):
    """Enhanced registry for all available LLM configurations."""

    model_config = ConfigDict(validate_assignment=True)

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    deepseek: Optional[DeepSeekConfig] = Field(default_factory=DeepSeekConfig)
    groq: Optional[GroqConfig] = Field(default=None)
    qwen: Optional[QwenConfig] = Field(default_factory=QwenConfig)
    claude: Optional[ClaudeConfig] = Field(default_factory=ClaudeConfig)
    gemini: Optional[GeminiConfig] = Field(default_factory=GeminiConfig)
    mistral: Optional[MistralConfig] = Field(default_factory=MistralConfig)


def get_model_for_agent(agent_role: AgentRole) -> Dict[str, Any]:
    """Get the recommended model configuration for a specific agent role.

    Args:
        agent_role: The role of the agent

    Returns:
        Dictionary containing model configuration and metadata
    """
    recommendations = AGENT_MODEL_RECOMMENDATIONS.get(agent_role)
    if not recommendations:
        # Default to OpenAI for unknown roles
        return {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "reasoning": "Default fallback model",
        }

    primary_model = recommendations["primary"]

    # Find the model configuration
    for provider, models in MODEL_CONFIGURATIONS.items():
        if primary_model in models:
            config = models[primary_model].copy()
            config["provider"] = provider
            config["alternatives"] = recommendations["alternatives"]
            config["reasoning"] = recommendations["reasoning"]
            return config

    # Fallback if model not found
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "reasoning": "Fallback due to configuration error",
    }


def get_enhanced_agent_config(
    role: AgentRole, context: Dict[str, Any] = None
) -> AgentConfig:
    """Get enhanced agent configuration with model recommendations.

    Args:
        role: Agent role
        context: Additional context for configuration

    Returns:
        Enhanced agent configuration with model recommendations
    """
    model_config = get_model_for_agent(role)

    return AgentConfig(
        name=f"{role.value.replace('_', ' ').title()} Agent",
        role=role,
        system_prompt=get_enhanced_system_prompt(role, context),
        tools=get_tools_for_agent(role),
        collections=get_collections_for_agent(role),
        model_config=model_config,
    )


def get_tools_for_agent(role: AgentRole) -> List[str]:
    """Get recommended tools for an agent role."""
    tool_mapping = {
        # Core agents
        AgentRole.SUPERVISOR: [
            "search_documents",
            "generate_code",
            "validate_code",
            "search_web",
        ],
        AgentRole.RESEARCH: ["search_documents", "search_web", "fetch_web_content"],
        AgentRole.IMPLEMENTATION: ["generate_code", "validate_code", "analyze_code"],
        AgentRole.DOCUMENTATION: [
            "search_documents",
            "create_document",
            "read_file_content",
            "write_file_content",
        ],
        AgentRole.TRD_CONVERTER: [
            "search_documents",
            "generate_code",
            "create_document",
        ],
        AgentRole.CODE_GENERATOR: ["generate_code", "validate_code", "analyze_code"],
        AgentRole.CODE_REVIEWER: ["validate_code", "analyze_code"],
        AgentRole.STANDARDS_ENFORCER: ["validate_code", "analyze_code"],
        AgentRole.STRATEGIC_EXPERT: ["search_documents", "search_web"],
        # Framework experts
        AgentRole.LANGCHAIN_EXPERT: ["search_documents", "generate_code", "search_web"],
        AgentRole.LANGGRAPH_EXPERT: [
            "generate_code",
            "validate_code",
            "search_documents",
        ],
        AgentRole.LLAMAINDEX_EXPERT: ["search_documents"],
        AgentRole.FASTAPI_EXPERT: ["generate_code", "validate_code", "analyze_code"],
        AgentRole.NEXTJS_EXPERT: ["generate_code", "validate_code", "search_documents"],
        AgentRole.PYDANTIC_EXPERT: ["generate_code", "validate_code"],
        AgentRole.AGNO_EXPERT: ["generate_code", "search_documents"],
        AgentRole.CREWAI_EXPERT: ["generate_code", "search_documents"],
        # Analysis agents
        AgentRole.DOCUMENT_ANALYZER: ["search_documents", "fetch_web_content"],
        AgentRole.PROMPT_ENGINEER: ["search_documents"],
    }
    return tool_mapping.get(role, ["search_documents"])


def get_collections_for_agent(role: AgentRole) -> List[str]:
    """Get recommended collections for an agent role."""
    collection_mapping = {
        # Core agents
        AgentRole.SUPERVISOR: ["strategy_book", "langchain_docs", "langgraph_docs"],
        AgentRole.RESEARCH: ["strategy_book", "research_papers"],
        AgentRole.IMPLEMENTATION: ["programming_docs", "best_practices"],
        AgentRole.DOCUMENTATION: ["documentation_templates", "style_guides"],
        AgentRole.TRD_CONVERTER: ["strategy_book", "requirements_docs"],
        AgentRole.CODE_GENERATOR: ["programming_docs", "best_practices"],
        AgentRole.CODE_REVIEWER: ["code_standards", "best_practices"],
        AgentRole.STANDARDS_ENFORCER: ["code_standards", "compliance_docs"],
        AgentRole.STRATEGIC_EXPERT: ["strategy_book", "business_docs"],
        # Framework experts
        AgentRole.LANGCHAIN_EXPERT: ["langchain_docs", "llm_docs"],
        AgentRole.LANGGRAPH_EXPERT: ["langgraph_docs", "workflow_docs"],
        AgentRole.LLAMAINDEX_EXPERT: ["llamaindex_docs", "rag_docs"],
        AgentRole.FASTAPI_EXPERT: ["fastapi_docs", "python_docs", "pydantic_docs"],
        AgentRole.NEXTJS_EXPERT: ["nextjs_docs", "react_docs", "typescript_docs"],
        AgentRole.PYDANTIC_EXPERT: ["pydantic_docs", "python_docs"],
        AgentRole.AGNO_EXPERT: ["agno_docs", "framework_docs"],
        AgentRole.CREWAI_EXPERT: ["crewai_docs", "multiagent_docs"],
        # Analysis agents
        AgentRole.DOCUMENT_ANALYZER: ["strategy_book", "analysis_docs"],
        AgentRole.PROMPT_ENGINEER: ["prompt_docs", "llm_docs"],
    }
    return collection_mapping.get(role, ["strategy_book"])


# Enhanced team configurations with model-aware agents
def get_enhanced_research_team_config() -> TeamConfig:
    """Get enhanced research team configuration with model recommendations."""
    return TeamConfig(
        name="Enhanced Research Team",
        description="AI-powered research team with optimized model selection",
        agents=[
            get_enhanced_agent_config(
                AgentRole.RESEARCH,
                {
                    "specialization": "comprehensive research and analysis",
                    "preferred_models": ["gemini-2.5-pro", "claude-3.7-sonnet"],
                },
            ),
            get_enhanced_agent_config(
                AgentRole.DOCUMENT_ANALYZER,
                {
                    "specialization": "document analysis and insight extraction",
                    "preferred_models": ["qwen3-30b-a3b", "claude-3.5-sonnet"],
                },
            ),
        ],
    )


def get_enhanced_implementation_team_config() -> TeamConfig:
    """Get enhanced implementation team configuration with model recommendations."""
    return TeamConfig(
        name="Enhanced Implementation Team",
        description="AI-powered development team with specialized coding models",
        agents=[
            get_enhanced_agent_config(
                AgentRole.CODE_GENERATOR,
                {
                    "specialization": "multi-language code generation",
                    "preferred_models": ["qwen2.5-coder-32b", "deepseek-coder-v2"],
                },
            ),
            get_enhanced_agent_config(
                AgentRole.FASTAPI_EXPERT,
                {
                    "specialization": "Python backend development",
                    "preferred_models": ["qwen2.5-coder-32b", "deepseek-coder-v2"],
                },
            ),
            get_enhanced_agent_config(
                AgentRole.NEXTJS_EXPERT,
                {
                    "specialization": "React/TypeScript frontend development",
                    "preferred_models": ["qwen2.5-coder-32b", "claude-3.5-sonnet"],
                },
            ),
        ],
    )


def get_enhanced_all_team_configs() -> List[TeamConfig]:
    """Get all enhanced team configurations."""
    return [
        get_enhanced_research_team_config(),
        get_enhanced_implementation_team_config(),
        get_documentation_team_config(),
    ]


# Export all symbols
__all__ = [
    # Configuration Models (imported from state.py)
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
    # Prompt Integration
    "get_enhanced_system_prompt",
    # Utility Functions
    "get_research_team_config",
    "get_implementation_team_config",
    "get_documentation_team_config",
    "get_all_team_configs",
    # Enhanced LLM configurations
    "QwenConfig",
    "DeepSeekConfig",
    "ClaudeConfig",
    "GeminiConfig",
    "MistralConfig",
    "EnhancedLLMRegistry",
    # Enhanced team configurations
    "get_enhanced_research_team_config",
    "get_enhanced_implementation_team_config",
    "get_enhanced_all_team_configs",
]
