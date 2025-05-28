"""Code Hero Backend Package.

A comprehensive AI agent system for software development with hierarchical multi-agent
workflows, advanced prompt engineering, and full infrastructure integration.
"""

# Core components
from . import (  # State and configuration; Agent system; Workflow and orchestration; API and services; Tools and utilities
    agent_expert,
    agent_manager,
    agents_api,
    astra_db,
    chat,
    config,
    context,
    graph,
    hierarchical_agents,
    human_loop,
    interfaces,
    langgraph_agents,
    logger,
    manager,
    node,
    prompts,
    services,
    state,
    strategic_agent,
    supervisor,
    tools,
    types,
    utils,
    workflow,
)

# Export all symbols for easy access
__all__ = [
    # Core infrastructure
    "state",
    "config",
    "context",
    "logger",
    "manager",
    "utils",
    # Agent system
    "agent_expert",
    "agent_manager",
    "supervisor",
    "prompts",
    "strategic_agent",
    # Workflow and orchestration
    "workflow",
    "node",
    "graph",
    "hierarchical_agents",
    "langgraph_agents",
    # API and services
    "chat",
    "agents_api",
    "services",
    "interfaces",
    "types",
    # Tools and utilities
    "tools",
    "astra_db",
    "human_loop",
]

# Version information
__version__ = "1.0.0"
__author__ = "Code Hero Team"
__description__ = "AI Agent System for Software Development"
