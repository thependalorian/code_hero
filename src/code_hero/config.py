"""Configuration management for the backend.

This module provides a centralized configuration system for the entire application,
including LLM providers, database, API, and workflow settings. It includes robust
validation, environment variable handling, and task-specific routing.
"""

import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dotenv import load_dotenv

from .state import BaseState, Status, AgentRole

# Load environment variables
load_dotenv()

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
    astra_db_region: str = Field(default_factory=lambda: os.getenv("ASTRA_DB_REGION", ""))
    astra_db_token: str = Field(default_factory=lambda: os.getenv("ASTRA_DB_APPLICATION_TOKEN", ""))
    collections: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("astra_db_token")
    @classmethod
    def validate_token(cls, v: str) -> str:
        """Validate Astra DB token."""
        if not v:
            raise ValueError("Astra DB token is required")
        return v


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""
    
    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.7
    max_tokens: int = 4096

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key."""
        if not v:
            raise ValueError("OpenAI API key is required")
        return v

    @field_validator("model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name."""
        valid_models = ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"]
        if v not in valid_models:
            raise ValueError(f"Invalid model name. Must be one of: {valid_models}")
        return v


class DeepseekConfig(BaseModel):
    """Deepseek configuration."""
    
    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key."""
        if not v:
            raise ValueError("Deepseek API key is required")
        return v

    @field_validator("model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name."""
        valid_models = ["deepseek-chat", "deepseek-coder"]
        if v not in valid_models:
            raise ValueError(f"Invalid model name. Must be one of: {valid_models}")
        return v


class GroqConfig(BaseModel):
    """Groq configuration."""
    
    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = "mixtral-8x7b-32768"
    temperature: float = 0.7
    max_tokens: int = 4096

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key."""
        if not v:
            raise ValueError("Groq API key is required")
        return v

    @field_validator("model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name."""
        valid_models = ["mixtral-8x7b-32768", "llama2-70b-4096"]
        if v not in valid_models:
            raise ValueError(f"Invalid model name. Must be one of: {valid_models}")
        return v


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

    token: str = Field(default_factory=lambda: os.getenv("ASTRA_DB_APPLICATION_TOKEN", ""))
    api_endpoint: str = Field(
        default_factory=lambda: f"https://{os.getenv('ASTRA_DB_ID')}-{os.getenv('ASTRA_DB_REGION')}.apps.astra.datastax.com"
    )
    region: str = Field(default_factory=lambda: os.getenv("ASTRA_DB_REGION", ""))
    database_id: str = Field(default_factory=lambda: os.getenv("ASTRA_DB_ID", ""))
    keyspace: str = "default_keyspace"
    collections: Dict[str, AstraDBCollection] = Field(
        default_factory=lambda: {
            "strategy_book": AstraDBCollection(
                name="strategy_book", description="Strategic framework documentation"
            ),
            "langchain_docs": AstraDBCollection(
                name="langchain_docs", description="LangChain documentation"
            ),
            "langgraph_docs": AstraDBCollection(
                name="langgraph_docs", description="LangGraph documentation"
            ),
            "llamaindex_docs": AstraDBCollection(
                name="llamaindex_docs", description="LlamaIndex documentation"
            ),
            "nextjs_docs": AstraDBCollection(
                name="nextjs_docs", description="Next.js documentation"
            ),
            "crewai_docs": AstraDBCollection(
                name="crewai_docs", description="CrewAI documentation"
            ),
            "fastapi_docs": AstraDBCollection(
                name="fastapi_docs", description="FastAPI documentation"
            ),
            "langsmith_docs": AstraDBCollection(
                name="langsmith_docs", description="LangSmith documentation"
            ),
            "pydantic_docs": AstraDBCollection(
                name="pydantic_docs", description="Pydantic documentation"
            ),
            "agno_phidata_docs": AstraDBCollection(
                name="agno_phidata_docs", description="Agno and Phidata documentation"
            ),
        }
    )


class LLMConfig(BaseModel):
    """Base LLM configuration."""
    
    model_config = ConfigDict(validate_assignment=True)

    api_key: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key."""
        if not v:
            raise ValueError("API key is required")
        return v


class LLMRegistry(BaseModel):
    """Registry for all available LLM configurations."""
    
    model_config = ConfigDict(validate_assignment=True)

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    deepseek: Optional[DeepseekConfig] = Field(default=None)
    groq: Optional[GroqConfig] = Field(default=None)

    def get_best_model_for_task(self, task_type: str) -> Union[OpenAIConfig, DeepseekConfig, GroqConfig]:
        """Get the best available model for a specific task type.

        Args:
            task_type: Type of task ('coding', 'reasoning', or 'trd_processing')

        Returns:
            The most suitable LLM configuration for the task
        """
        if task_type == "coding":
            # Prioritize DeepSeek for coding tasks
            if self.deepseek:
                return self.deepseek
            elif self.groq:
                return self.groq
        elif task_type == "trd_processing":
            # Prioritize GPT-4 for TRD processing
            if self.openai:
                return self.openai
            elif self.groq:
                return self.groq
        else:  # reasoning tasks
            # Prioritize OpenAI for reasoning tasks
            if self.openai:
                return self.openai
            elif self.deepseek:
                return self.deepseek
            elif self.groq:
                return self.groq

        raise ValueError(f"No suitable model configuration found for task type: {task_type}")


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
                "status": "idle"
            }
        }
    )

    api: APIConfig
    database: DatabaseConfig
    llm_registry: LLMRegistry = Field(default_factory=LLMRegistry)
    logging: LoggingConfig
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    environment: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    status: Status = Status.IDLE


def get_research_team_config() -> TeamConfig:
    """Get research team configuration."""
    return TeamConfig(
        name="research",
        description="Team responsible for research and analysis",
        agents=[
            AgentConfig(
                name="researcher",
                role=AgentRole.RESEARCH,
                system_prompt="You are an expert researcher...",
                tools=["web_search", "document_analysis"],
            ),
            AgentConfig(
                name="analyst",
                role=AgentRole.RESEARCH,
                system_prompt="You are a data analyst...",
                tools=["data_analysis", "visualization"],
            ),
        ],
    )


def get_implementation_team_config() -> TeamConfig:
    """Get implementation team configuration."""
    return TeamConfig(
        name="implementation",
        description="Team responsible for code implementation",
        agents=[
            AgentConfig(
                name="developer",
                role=AgentRole.IMPLEMENTATION,
                system_prompt="You are an expert developer...",
                tools=["code_generation", "testing"],
            ),
            AgentConfig(
                name="reviewer",
                role=AgentRole.CODE_REVIEWER,
                system_prompt="You are a code reviewer...",
                tools=["code_review", "linting"],
            ),
        ],
    )


def get_documentation_team_config() -> TeamConfig:
    """Get documentation team configuration."""
    return TeamConfig(
        name="documentation",
        description="Team responsible for documentation",
        agents=[
            AgentConfig(
                name="writer",
                role=AgentRole.DOCUMENTATION,
                system_prompt="You are a technical writer...",
                tools=["doc_generation", "markdown"],
            ),
            AgentConfig(
                name="reviewer",
                role=AgentRole.DOCUMENTATION,
                system_prompt="You are a documentation reviewer...",
                tools=["doc_review", "spell_check"],
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


# Export configurations
__all__ = [
    "APIConfig",
    "DatabaseConfig",
    "OpenAIConfig",
    "DeepseekConfig",
    "GroqConfig",
    "AstraConfig",
    "LLMRegistry",
    "LoggingConfig",
    "AgentConfig",
    "TeamConfig",
    "WorkflowConfig",
    "ConfigState",
    "get_research_team_config",
    "get_implementation_team_config",
    "get_documentation_team_config",
    "get_all_team_configs",
]
