"""Main FastAPI application and CLI interface.

This module provides both a FastAPI application for web server operation
and a CLI interface for command-line usage.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

import typer
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from . import agents_api, astra_db, chat, documents_api, graph
from .config import (
    APIConfig,
    ConfigState,
    DatabaseConfig,
    LLMRegistry,
    LoggingConfig,
    WorkflowConfig,
)
from .context import get_services
from .logger import StructuredLogger
from .manager import StateManager
from .state import BaseState, Status
from .supervisor import SupervisorAgent

# Create Typer app
cli = typer.Typer(
    name="code-hero",
    help="Code Hero - A powerful AI coding assistant",
    add_completion=False,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI app."""
    # Startup
    logger.info("Starting Code Hero API")

    try:
        # Step 1: Environment validation
        logger.info("Step 1/6: Environment validation")
        validate_environment()

        # Step 2: Configuration initialization
        logger.info("Step 2/6: Configuration initialization")
        config = initialize_config()
        initialize_services(config)

        # Step 3: Core services initialization
        logger.info("Step 3/6: Core services initialization")
        await app.state.logger.initialize()
        await app.state.state_manager.initialize()
        await app.state.supervisor.initialize()

        # Step 4: API connections initialization
        logger.info("Step 4/6: API connections initialization")
        await initialize_apis()

        # Step 5: Tools initialization
        logger.info("Step 5/6: Tools initialization")
        await initialize_tools()

        # Step 6: Agents and workflows initialization
        logger.info("Step 6/6: Agents and workflows initialization")
        await initialize_agents()
        await initialize_workflows()

        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Code Hero API")

    try:
        # Clean up in reverse order
        await app.state.supervisor.cleanup()
        await app.state.state_manager.cleanup()
        await app.state.logger.cleanup()

        # Clean up tools and APIs
        if hasattr(app.state, "astra_client"):
            # Clean up AstraDB connection
            pass

        if hasattr(app.state, "openai_client"):
            # Clean up OpenAI connection
            await app.state.openai_client.close()

        logger.info("Shutdown completed successfully")
    except Exception as e:
        logger.error(f"Failed to shut down cleanly: {str(e)}")
        raise


# Create FastAPI app
app = FastAPI(
    title="Code Hero API",
    description="API for the Code Hero AI coding assistant",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Configure logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("code_hero")

# Required environment variables
REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "ASTRA_DB_ID",
    "ASTRA_DB_REGION",
    "ASTRA_DB_APPLICATION_TOKEN",
]


def validate_environment():
    """Validate required environment variables."""
    logger.info("Validating environment variables...")

    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log environment variable status
    for var in REQUIRED_ENV_VARS:
        logger.info(f"{var} present: {bool(os.getenv(var))}")

    # Log optional variables
    logger.info(f"DEEPSEEK_API_KEY present: {bool(os.getenv('DEEPSEEK_API_KEY'))}")
    logger.info(f"GROQ_API_KEY present: {bool(os.getenv('GROQ_API_KEY'))}")
    logger.info(f"TAVILY_API_KEY present: {bool(os.getenv('TAVILY_API_KEY'))}")
    logger.info(f"LANGCHAIN_API_KEY present: {bool(os.getenv('LANGCHAIN_API_KEY'))}")
    logger.info(
        f"LANGCHAIN_TRACING_V2 present: {bool(os.getenv('LANGCHAIN_TRACING_V2'))}"
    )
    logger.info(
        f"LANGCHAIN_TRACING_ENDPOINT present: {bool(os.getenv('LANGCHAIN_TRACING_ENDPOINT'))}"
    )
    logger.info(f"LANGCHAIN_PROJECT present: {bool(os.getenv('LANGCHAIN_PROJECT'))}")


def initialize_config() -> ConfigState:
    """Initialize application configuration."""
    logger.info("Initializing configuration...")

    try:
        config = ConfigState(
            id="main_config",
            api=APIConfig(
                host="0.0.0.0", port=8000, debug=False, reload=True, workers=1
            ),
            database=DatabaseConfig(
                astra_db_id=os.getenv("ASTRA_DB_ID", ""),
                astra_db_region=os.getenv("ASTRA_DB_REGION", ""),
                astra_db_token=os.getenv("ASTRA_DB_APPLICATION_TOKEN", ""),
            ),
            llm_registry=LLMRegistry(),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                format="%(message)s",
                file="code_hero.log",
            ),
            workflow=WorkflowConfig(
                name="default",
                description="Default workflow configuration",
                nodes=[],
                max_retries=3,
                timeout_seconds=300,
            ),
        )
        logger.info("Configuration initialized successfully")
        return config
    except Exception as e:
        error_msg = f"Failed to initialize configuration: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


async def initialize_apis() -> None:
    """Initialize external API connections."""
    logger.info("Initializing API connections...")

    try:
        # Initialize OpenAI client
        from openai import AsyncOpenAI

        app.state.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("OpenAI client initialized")

        # Initialize AstraDB client
        from astrapy import DataAPIClient

        app.state.astra_client = DataAPIClient(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))
        app.state.astra_db = app.state.astra_client.get_database_by_api_endpoint(
            f"https://{os.getenv('ASTRA_DB_ID')}-{os.getenv('ASTRA_DB_REGION')}.apps.astra.datastax.com"
        )
        logger.info("AstraDB client initialized")

        # Initialize other API clients as needed
        if os.getenv("DEEPSEEK_API_KEY"):
            # Initialize Deepseek
            logger.info("Deepseek client initialized")

        if os.getenv("GROQ_API_KEY"):
            # Initialize Groq
            logger.info("Groq client initialized")

        logger.info("All API connections initialized successfully")
    except Exception as e:
        error_msg = f"Failed to initialize API connections: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


async def initialize_tools() -> None:
    """Initialize tool registry and tools."""
    logger.info("Initializing tools...")

    try:
        from .tools import (
            analyze_code,
            astra_retriever,
            generate_code,
            search_documents,
            tool_registry,
            validate_code,
        )

        # Initialize AstraDB tools
        await astra_retriever.initialize()

        # Register core tools
        tool_registry.register_tool("search_documents", search_documents)
        tool_registry.register_tool("generate_code", generate_code)
        tool_registry.register_tool("validate_code", validate_code)
        tool_registry.register_tool("analyze_code", analyze_code)

        logger.info("Tool registry initialized successfully")
    except Exception as e:
        error_msg = f"Failed to initialize tools: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


async def initialize_agents() -> None:
    """Initialize agent system."""
    logger.info("Initializing agent system...")

    try:
        from .agent_expert import experts

        # Initialize expert agents
        for role, expert in experts.items():
            if hasattr(expert, "initialize"):
                await expert.initialize()
            logger.info(f"Initialized {role} expert")

        logger.info("Agent system initialized successfully")
    except Exception as e:
        error_msg = f"Failed to initialize agents: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


async def initialize_workflows() -> None:
    """Initialize workflow system."""
    logger.info("Initializing workflow system...")

    try:
        from .chat import initialize_chat_workflows
        from .workflow import create_workflow_graph

        # Create and compile default workflow graph
        app.state.workflow_graph = create_workflow_graph()
        logger.info("Workflow graph initialized")

        # Initialize chat workflows (LangGraph integration)
        initialize_chat_workflows()
        logger.info("Chat workflows initialized")

        logger.info("Workflow system initialized successfully")
    except Exception as e:
        error_msg = f"Failed to initialize workflows: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def initialize_services(config: ConfigState):
    """Initialize application services."""
    logger.info("Initializing services...")

    try:
        # Initialize services in order
        app.state.config = config
        app.state.logger = StructuredLogger(config.logging)
        app.state.state_manager = StateManager(config)
        app.state.supervisor = SupervisorAgent(
            state_manager=app.state.state_manager, logger=app.state.logger
        )
        logger.info("Services initialized successfully")
    except Exception as e:
        error_msg = f"Failed to initialize services: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers with proper prefixes and tags
app.include_router(
    chat.router,
    prefix="/api/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    graph.router,
    prefix="/api/graph",
    tags=["graph"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    astra_db.router,
    prefix="/api/astra",
    tags=["astra"],
    responses={404: {"description": "Not found"}},
)

app.include_router(agents_api.router, prefix="/api/agents", tags=["agents"])

app.include_router(
    documents_api.router,
    prefix="/api/documents",
    tags=["documents"],
    responses={404: {"description": "Not found"}},
)


@app.get("/health")
async def health_check() -> Dict:
    """Health check endpoint."""
    try:
        # Check all services
        services_status = {
            "logger": await app.state.logger.check_health(),
            "state_manager": await app.state.state_manager.check_health(),
            "supervisor": await app.state.supervisor.check_health(),
        }

        # Check configuration
        config_status = {
            "api": app.state.config.api is not None,
            "database": app.state.config.database is not None,
            "llm_registry": app.state.config.llm_registry is not None,
            "logging": app.state.config.logging is not None,
        }

        return {
            "status": "healthy",
            "services": services_status,
            "config": config_status,
            "environment": app.state.config.environment,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"status": "unhealthy", "error": str(e)}
        )


@app.get("/")
async def root() -> Dict:
    """Root endpoint with API information."""
    return {
        "name": "Code Hero API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs",
        "redoc": "/api/redoc",
        "openapi": "/api/openapi.json",
        "endpoints": {"multi_agent": "/multi-agent/coordinate", "health": "/health"},
    }


@app.post("/multi-agent/coordinate")
async def coordinate_multi_agent_task(request: Dict[str, Any]) -> Dict[str, Any]:
    """Coordinate a task across multiple agents."""
    logger.info(
        f"Coordinating multi-agent task: {request.get('task_description', 'No description')}"
    )

    try:
        # Get services
        services = get_services()
        state_manager = services.state_manager
        supervisor = services.supervisor

        # Extract task details
        task_description = request.get("task_description", "")
        project_id = request.get("project_id")

        if not task_description:
            raise HTTPException(status_code=400, detail="Task description is required")

        # Create initial state
        initial_state = BaseState(
            id=f"task_{datetime.now().isoformat()}",
            status=Status.PENDING,
            metadata={
                "task_description": task_description,
                "project_id": project_id,
                "timestamp": datetime.now().isoformat(),
                "multi_agent": True,
            },
        )

        # Store state
        await state_manager.store_state(initial_state)

        # Process with supervisor
        result = await supervisor.process_task(initial_state)

        return {
            "success": True,
            "project_id": project_id or initial_state.id,
            "task_description": task_description,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Multi-agent coordination failed: {str(e)}")
        return {
            "success": False,
            "project_id": project_id,
            "task_description": task_description,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@app.get("/hierarchical/status")
async def get_hierarchical_system_status() -> Dict[str, Any]:
    """Get the status of the hierarchical agent system."""
    logger.info("Getting hierarchical system status")

    try:
        from .hierarchical_agents import validate_full_infrastructure

        # Get validation results
        validation_results = await validate_full_infrastructure()

        # Transform to expected format
        teams = []
        if validation_results.get("overall_status") == "healthy":
            teams = [
                {
                    "name": "Development",
                    "agents": ["fastapi_expert", "nextjs_expert", "code_reviewer"],
                    "status": "healthy",
                },
                {
                    "name": "Research",
                    "agents": ["research", "strategic_expert", "document_analyzer"],
                    "status": "healthy",
                },
                {
                    "name": "Documentation",
                    "agents": ["documentation", "implementation", "prompt_engineer"],
                    "status": "healthy",
                },
            ]

        return {
            "status": validation_results.get("overall_status", "unknown"),
            "teams": teams,
            "infrastructure": validation_results.get("component_status", {}),
            "performance": {
                "healthy_components": len(
                    validation_results.get("healthy_components", [])
                ),
                "failed_components": len(
                    validation_results.get("failed_components", [])
                ),
                "total_components": len(validation_results.get("component_status", {})),
            },
        }

    except Exception as e:
        logger.error(f"Failed to get hierarchical system status: {str(e)}")
        return {
            "status": "error",
            "teams": [],
            "infrastructure": {},
            "performance": {},
            "error": str(e),
        }


@app.post("/hierarchical/process")
async def process_hierarchical_message(request: Dict[str, Any]) -> Dict[str, Any]:
    """Process a message using the hierarchical agent system."""
    logger.info(
        f"Processing hierarchical message: {request.get('message', 'No message')[:100]}..."
    )

    try:
        from .hierarchical_agents import process_with_hierarchical_agents

        # Extract request data
        message = request.get("message", "")
        conversation_id = request.get("conversation_id")
        project_id = request.get("project_id")
        task_priority = request.get("task_priority", "medium")
        context = request.get("context", {})

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Process with hierarchical agents
        response = await process_with_hierarchical_agents(
            message=message,
            conversation_id=conversation_id,
            project_id=project_id,
            task_priority=task_priority,
            context=context,
        )

        # Parse response if it's a string
        if isinstance(response, str):
            return {
                "response": response,
                "conversation_id": conversation_id
                or f"conv_{datetime.now().isoformat()}",
                "project_id": project_id or f"proj_{datetime.now().isoformat()}",
                "agents_used": ["hierarchical_system"],
                "tools_used": [],
                "performance_metrics": {},
                "human_interventions": 0,
            }

        # If response is already structured, return it
        return response

    except Exception as e:
        logger.error(f"Hierarchical message processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/hierarchical/validate")
async def validate_hierarchical_infrastructure() -> Dict[str, Any]:
    """Validate the hierarchical agent infrastructure."""
    logger.info("Validating hierarchical infrastructure")

    try:
        from .hierarchical_agents import validate_full_infrastructure

        validation_results = await validate_full_infrastructure()
        return validation_results

    except Exception as e:
        logger.error(f"Infrastructure validation failed: {str(e)}")
        return {
            "overall_status": "error",
            "component_status": {},
            "failed_components": ["validation_system"],
            "healthy_components": [],
            "error": str(e),
        }


@cli.command()
def run_server(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    workers: int = typer.Option(1, help="Number of worker processes"),
):
    """Run the FastAPI server."""
    console.print("[bold green]Starting Code Hero Server[/bold green]")
    uvicorn.run(
        "code_hero.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )


@cli.command()
def check_config():
    """Check and validate configuration."""
    try:
        # Validate environment
        validate_environment()

        # Initialize configuration
        config = initialize_config()

        table = Table(title="Configuration Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        table.add_column("Validation", style="magenta")

        # Check LLM configurations
        if config.llm_registry.openai:
            table.add_row(
                "OpenAI",
                "✓ Configured",
                config.llm_registry.openai.model,
                (
                    "API Key Valid"
                    if config.llm_registry.openai.api_key
                    else "Missing API Key"
                ),
            )
        if config.llm_registry.deepseek:
            table.add_row(
                "DeepSeek",
                "✓ Configured",
                config.llm_registry.deepseek.model,
                (
                    "API Key Valid"
                    if config.llm_registry.deepseek.api_key
                    else "Missing API Key"
                ),
            )
        if config.llm_registry.groq:
            table.add_row(
                "Groq",
                "✓ Configured",
                config.llm_registry.groq.model,
                (
                    "API Key Valid"
                    if config.llm_registry.groq.api_key
                    else "Missing API Key"
                ),
            )

        # Check database configuration
        db_status = (
            "✓ Configured" if config.database.astra_db_token else "✗ Not Configured"
        )
        table.add_row(
            "AstraDB",
            db_status,
            f"Region: {config.database.astra_db_region}",
            "Token Valid" if config.database.astra_db_token else "Missing Token",
        )

        # Check logging configuration
        table.add_row(
            "Logging",
            "✓ Configured",
            f"Level: {config.logging.level}",
            "Valid Configuration",
        )

        console.print(table)

        # Additional environment checks
        console.print("\n[bold]Environment Checks:[/bold]")
        console.print(f"• Environment: [cyan]{config.environment}[/cyan]")
        console.print(
            f"• Debug Mode: [cyan]{'Yes' if config.api.debug else 'No'}[/cyan]"
        )
        console.print(f"• Workers: [cyan]{config.api.workers}[/cyan]")

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)


@cli.command()
def list_models():
    """List available models and their configurations."""
    try:
        config = initialize_config()
        registry = config.llm_registry

        table = Table(title="Available Models")
        table.add_column("Provider", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Best For", style="magenta")

        if registry.openai:
            table.add_row(
                "OpenAI",
                registry.openai.model,
                "✓ Available",
                "Reasoning, Code Generation",
            )
        if registry.deepseek:
            table.add_row(
                "DeepSeek", registry.deepseek.model, "✓ Available", "Code Generation"
            )
        if registry.groq:
            table.add_row("Groq", registry.groq.model, "✓ Available", "General Purpose")

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)


@cli.command()
def list_agents():
    """List available agents and their capabilities."""
    try:
        import asyncio

        from .agent_manager import agent_manager

        async def get_agents():
            return await agent_manager.get_all_agents()

        agents = asyncio.run(get_agents())

        table = Table(title="Available Agents")
        table.add_column("Agent", style="cyan")
        table.add_column("Role", style="green")
        table.add_column("Team", style="blue")
        table.add_column("Capabilities", style="yellow")
        table.add_column("Tools", style="magenta")

        # Sort agents by team and then by name
        sorted_agents = sorted(agents, key=lambda x: (x.team or "ZZZ", x.name))

        for agent in sorted_agents:
            # Format capabilities (limit to 3 for display)
            capabilities = (
                agent.capabilities[:3]
                if len(agent.capabilities) > 3
                else agent.capabilities
            )
            capabilities_str = ", ".join(capabilities)
            if len(agent.capabilities) > 3:
                capabilities_str += f" (+{len(agent.capabilities) - 3} more)"

            # Format tools (limit to 3 for display)
            tools = agent.tools[:3] if len(agent.tools) > 3 else agent.tools
            tools_str = ", ".join(tools)
            if len(agent.tools) > 3:
                tools_str += f" (+{len(agent.tools) - 3} more)"

            table.add_row(
                agent.id.replace("agent_", ""),
                agent.name,
                agent.team or "General",
                capabilities_str,
                tools_str or "No tools",
            )

        console.print(table)

        # Show team summary
        async def get_team_stats():
            teams = await agent_manager.get_teams()
            team_stats = {}
            for team in teams:
                team_agents = await agent_manager.get_agents_by_team(team)
                team_stats[team] = len(team_agents)
            return team_stats

        team_stats = asyncio.run(get_team_stats())

        console.print(f"\n[bold cyan]Team Summary:[/bold cyan]")
        for team, count in sorted(team_stats.items()):
            console.print(f"  • {team}: {count} agents")

        console.print(
            f"\n[bold green]Total: {len(agents)} agents across {len(team_stats)} teams[/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)


@cli.command()
def query_agent(
    agent: str = typer.Argument(
        ...,
        help="Agent to query (e.g., 'strategic_expert', 'langchain_expert', 'fastapi_expert')",
    ),
    query: str = typer.Argument(..., help="Query to send to the agent"),
    collection: str = typer.Option("", help="Specific collection to search (optional)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Query a specific agent with a question."""
    try:
        # Validate environment first
        validate_environment()

        # Initialize configuration and services
        initialize_config()

        # Import required modules
        import uuid

        from .agent_expert import execute_agent
        from .state import AgentRole, AgentState, Status

        console.print(f"[bold cyan]Querying {agent} agent...[/bold cyan]")

        # Validate agent role
        try:
            agent_role = AgentRole(agent.lower())
        except ValueError:
            console.print(f"[bold red]Error: Unknown agent '{agent}'[/bold red]")
            console.print("Available agents:")
            for role in AgentRole:
                console.print(f"  • {role.value}")
            raise typer.Exit(1)

        # Create agent state
        agent_state = AgentState(
            id=str(uuid.uuid4()),
            agent=agent_role,
            status=Status.PENDING,
            context={
                "query": query,
                "collection": collection if collection else "auto",
                "verbose": verbose,
            },
        )

        if verbose:
            console.print(f"[dim]Agent State ID: {agent_state.id}[/dim]")
            console.print(f"[dim]Query: {query}[/dim]")
            console.print(f"[dim]Collection: {collection or 'auto-detect'}[/dim]")

        # Execute agent asynchronously
        async def run_agent():
            try:
                result = await execute_agent(agent_role, agent_state, query)
                return result
            except Exception as e:
                console.print(f"[bold red]Agent execution failed: {str(e)}[/bold red]")
                raise typer.Exit(1)

        # Run the async function
        import asyncio

        result = asyncio.run(run_agent())

        # Display results
        console.print(f"\n[bold green]✓ Agent Response:[/bold green]")

        if result.status == Status.COMPLETED:
            # Show the main response if available
            if result.artifacts and "response" in result.artifacts:
                console.print(result.artifacts["response"])

            # Show artifacts if available
            if result.artifacts:
                if "search_results" in result.artifacts:
                    search_results = result.artifacts["search_results"]
                    if isinstance(search_results, list) and search_results:
                        console.print(
                            f"\n[bold]Found {len(search_results)} relevant documents:[/bold]"
                        )
                        for i, doc in enumerate(search_results[:3], 1):  # Show top 3
                            similarity = doc.get("metadata", {}).get(
                                "similarity", doc.get("similarity", 0.0)
                            )
                            console.print(
                                f"\n[cyan]{i}. Document (Score: {similarity:.3f})[/cyan]"
                            )
                            content = doc.get("content", "No content available")
                            # Truncate long content
                            if len(content) > 300:
                                content = content[:300] + "..."
                            console.print(f"[dim]{content}[/dim]")

                if "research_results" in result.artifacts and verbose:
                    research_results = result.artifacts["research_results"]
                    console.print(f"\n[bold]Research Details:[/bold]")
                    for source_data in research_results:
                        source = source_data["source"]
                        results = source_data["results"]
                        console.print(
                            f"\n[cyan]{source}: {len(results)} results[/cyan]"
                        )

                if "generated_code" in result.artifacts:
                    code = result.artifacts["generated_code"]
                    console.print(f"\n[bold]Generated Code:[/bold]")
                    console.print(f"```\n{code}\n```")

                if "validation_results" in result.artifacts:
                    validation = result.artifacts["validation_results"]
                    console.print(f"\n[bold]Validation Results:[/bold]")
                    console.print(
                        f"[green]✓ Valid[/green]"
                        if validation.get("valid")
                        else f"[red]✗ Invalid[/red]"
                    )
                    if validation.get("errors"):
                        for error in validation["errors"]:
                            console.print(f"[red]  • {error}[/red]")

            # Show context if verbose
            if verbose and result.context:
                console.print(f"\n[bold]Context:[/bold]")
                for key, value in result.context.items():
                    if key != "chat_history":  # Skip large chat history
                        console.print(f"  {key}: {value}")

        elif result.status == Status.FAILED:
            console.print(f"[bold red]✗ Agent failed: {result.error}[/bold red]")
            raise typer.Exit(1)

        else:
            console.print(f"[yellow]Agent status: {result.status}[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)


@cli.command()
def chat_mode(
    agent: str = typer.Option("supervisor", help="Agent to chat with"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Start an interactive chat session with an agent."""
    try:
        # Validate environment
        validate_environment()

        # Initialize configuration
        initialize_config()

        import uuid

        from .agent_expert import execute_agent
        from .state import AgentRole, AgentState, Status

        # Validate agent role
        try:
            agent_role = AgentRole(agent.lower())
        except ValueError:
            console.print(f"[bold red]Error: Unknown agent '{agent}'[/bold red]")
            console.print("Available agents:")
            for role in AgentRole:
                console.print(f"  • {role.value}")
            raise typer.Exit(1)

        console.print(
            f"[bold green]🤖 Starting chat with {agent_role.value} agent[/bold green]"
        )
        console.print("[dim]Type 'exit', 'quit', or 'bye' to end the session[/dim]")
        console.print("[dim]Type 'help' for available commands[/dim]\n")

        conversation_history = []

        while True:
            try:
                # Get user input
                user_input = typer.prompt("You", prompt_suffix=" > ")

                # Handle special commands
                if user_input.lower() in ["exit", "quit", "bye"]:
                    console.print("[bold yellow]👋 Goodbye![/bold yellow]")
                    break

                if user_input.lower() == "help":
                    console.print("[bold]Available commands:[/bold]")
                    console.print("  • help - Show this help message")
                    console.print("  • clear - Clear conversation history")
                    console.print("  • history - Show conversation history")
                    console.print("  • switch <agent> - Switch to a different agent")
                    console.print("  • exit/quit/bye - End the session")
                    continue

                if user_input.lower() == "clear":
                    conversation_history.clear()
                    console.print("[yellow]Conversation history cleared[/yellow]")
                    continue

                if user_input.lower() == "history":
                    if conversation_history:
                        console.print("[bold]Conversation History:[/bold]")
                        for i, (role, message) in enumerate(conversation_history, 1):
                            console.print(f"{i}. [{role}] {message}")
                    else:
                        console.print("[dim]No conversation history[/dim]")
                    continue

                if user_input.lower().startswith("switch "):
                    new_agent = user_input[7:].strip()
                    try:
                        agent_role = AgentRole(new_agent.lower())
                        console.print(
                            f"[green]Switched to {agent_role.value} agent[/green]"
                        )
                        continue
                    except ValueError:
                        console.print(f"[red]Unknown agent: {new_agent}[/red]")
                        continue

                # Add to conversation history
                conversation_history.append(("user", user_input))

                # Create agent state
                agent_state = AgentState(
                    id=str(uuid.uuid4()),
                    agent=agent_role,
                    status=Status.PENDING,
                    context={
                        "query": user_input,
                        "conversation_history": conversation_history,
                        "verbose": verbose,
                    },
                )

                # Show thinking indicator
                with console.status(
                    f"[bold green]{agent_role.value} is thinking...", spinner="dots"
                ):
                    # Execute agent asynchronously
                    async def run_agent():
                        return await execute_agent(agent_role, agent_state, user_input)

                    import asyncio

                    result = asyncio.run(run_agent())

                # Process and display response
                if result.status == Status.COMPLETED:
                    response = "I've processed your request."

                    # Extract meaningful response from artifacts
                    if result.artifacts:
                        if "search_results" in result.artifacts:
                            search_results = result.artifacts["search_results"]
                            if isinstance(search_results, list) and search_results:
                                response = f"I found {len(search_results)} relevant documents. Here's a summary:\n"
                                for doc in search_results[:2]:  # Top 2 results
                                    content = doc.get("content", "")[:200] + "..."
                                    response += f"• {content}\n"

                        if "generated_code" in result.artifacts:
                            code = result.artifacts["generated_code"]
                            response = f"Here's the generated code:\n```\n{code}\n```"

                    console.print(
                        f"[bold cyan]{agent_role.value}[/bold cyan] > {response}"
                    )
                    conversation_history.append((agent_role.value, response))

                elif result.status == Status.FAILED:
                    error_msg = f"Sorry, I encountered an error: {result.error}"
                    console.print(
                        f"[bold red]{agent_role.value}[/bold red] > {error_msg}"
                    )
                    conversation_history.append((agent_role.value, error_msg))

                else:
                    status_msg = f"Status: {result.status}"
                    console.print(f"[yellow]{agent_role.value}[/yellow] > {status_msg}")

            except KeyboardInterrupt:
                console.print(
                    "\n[bold yellow]👋 Chat interrupted. Goodbye![/bold yellow]"
                )
                break
            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")
                continue

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)


@cli.command()
def search_docs(
    query: str = typer.Argument(..., help="Search query"),
    collection: str = typer.Option("strategy_book", help="Collection to search"),
    limit: int = typer.Option(5, help="Number of results to return"),
):
    """Search documents in AstraDB collections."""
    try:
        # Validate environment
        validate_environment()

        from .tools import astra_retriever

        console.print(f"[bold cyan]Searching '{collection}' for: {query}[/bold cyan]")

        # Search documents asynchronously
        async def search():
            await astra_retriever.initialize()
            return await astra_retriever.search(query, collection, limit)

        import asyncio

        results = asyncio.run(search())

        if results:
            console.print(f"\n[bold green]Found {len(results)} results:[/bold green]")

            for i, doc in enumerate(results, 1):
                similarity = doc.get("metadata", {}).get("similarity", 0.0)
                content = doc.get("content", "No content available")
                title = doc.get("metadata", {}).get("title", f"Document {i}")

                console.print(
                    f"\n[bold cyan]{i}. {title} (Similarity: {similarity:.3f})[/bold cyan]"
                )

                # Truncate long content
                if len(content) > 500:
                    content = content[:500] + "..."

                console.print(f"[dim]{content}[/dim]")
        else:
            console.print("[yellow]No results found[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)


@cli.command()
def search_web(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int = typer.Option(5, help="Maximum number of results"),
):
    """Search the web using Tavily API."""
    try:
        # Validate environment
        validate_environment()

        import asyncio

        import aiohttp

        console.print(f"[bold cyan]Searching the web for: {query}[/bold cyan]")

        # Search web asynchronously
        async def search():
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                raise ValueError("TAVILY_API_KEY not found in environment variables")

            search_data = {
                "api_key": tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "include_images": False,
                "include_raw_content": False,
                "max_results": max_results,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.tavily.com/search",
                    json=search_data,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    return await response.json()

        result = asyncio.run(search())

        # Display AI answer if available
        if result.get("answer"):
            console.print(f"\n[bold green]AI Answer:[/bold green]")
            console.print(f"[yellow]{result['answer']}[/yellow]")

        # Display search results
        results = result.get("results", [])
        if results:
            console.print(
                f"\n[bold green]Found {len(results)} web results:[/bold green]"
            )

            for i, item in enumerate(results, 1):
                title = item.get("title", "No title")
                url = item.get("url", "")
                content = item.get("content", "No content")
                score = item.get("score", 0.0)

                console.print(
                    f"\n[bold cyan]{i}. {title} (Score: {score:.3f})[/bold cyan]"
                )
                console.print(f"[blue]{url}[/blue]")

                # Truncate long content
                if len(content) > 400:
                    content = content[:400] + "..."

                console.print(f"[dim]{content}[/dim]")
        else:
            console.print("[yellow]No web results found[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    cli()
