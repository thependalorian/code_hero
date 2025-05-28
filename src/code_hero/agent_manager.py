"""Agent Manager for tracking and managing agent status and performance."""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .logger import StructuredLogger
from .state import AgentInfoExtended as AgentInfo
from .state import AgentPerformance, AgentRole, AgentStatus, AgentType


class AgentManager:
    """Manages agent status, performance, and real-time information."""

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """Initialize agent manager."""
        self.logger = logger
        self.agents: Dict[str, AgentInfo] = {}
        self.task_history: Dict[str, List[Dict[str, Any]]] = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize agent information from the expert registry and comprehensive agent definitions."""
        # Import here to avoid circular imports
        from .agent_expert import experts
        from .state import AgentRole

        # Helper function to get tool names from an expert agent
        def get_agent_tools(role: AgentRole) -> List[str]:
            """Get tool names for an agent role."""
            expert = experts.get(role)
            if expert and hasattr(expert, "tools"):
                tool_names = []
                for tool in expert.tools:
                    if hasattr(tool, "name"):
                        tool_names.append(tool.name)
                    elif hasattr(tool, "__name__"):
                        tool_names.append(tool.__name__)
                    else:
                        tool_names.append(str(tool))
                return tool_names

            # Fallback: assign tools based on agent role
            role_tool_mapping = {
                AgentRole.RESEARCH: [
                    "search_documents",
                    "search_web",
                    "fetch_web_content",
                ],
                AgentRole.DOCUMENT_ANALYZER: ["search_documents", "fetch_web_content"],
                AgentRole.STRATEGIC_EXPERT: ["search_documents", "search_web"],
                AgentRole.LANGCHAIN_EXPERT: [
                    "generate_code",
                    "validate_code",
                    "analyze_code",
                    "search_documents",
                ],
                AgentRole.FASTAPI_EXPERT: [
                    "generate_code",
                    "validate_code",
                    "analyze_code",
                ],
                AgentRole.NEXTJS_EXPERT: [
                    "generate_code",
                    "validate_code",
                    "analyze_code",
                ],
                AgentRole.IMPLEMENTATION: [
                    "generate_code",
                    "validate_code",
                    "analyze_code",
                ],
                AgentRole.CODE_GENERATOR: ["generate_code", "validate_code"],
                AgentRole.CODE_REVIEWER: ["validate_code", "analyze_code"],
                AgentRole.SUPERVISOR: [
                    "search_documents",
                    "search_web",
                    "fetch_web_content",
                    "generate_code",
                    "validate_code",
                    "analyze_code",
                ],  # All tools
                AgentRole.DOCUMENTATION: ["search_documents", "analyze_code"],
                AgentRole.PROMPT_ENGINEER: ["search_documents", "search_web"],
                AgentRole.TRD_CONVERTER: ["search_documents", "analyze_code"],
                AgentRole.LANGGRAPH_EXPERT: [
                    "search_documents",
                    "generate_code",
                    "validate_code",
                ],
                AgentRole.LLAMAINDEX_EXPERT: [
                    "search_documents",
                    "generate_code",
                    "validate_code",
                ],
                AgentRole.PYDANTIC_EXPERT: [
                    "generate_code",
                    "validate_code",
                    "analyze_code",
                ],
                AgentRole.AGNO_EXPERT: ["search_documents", "generate_code"],
                AgentRole.CREWAI_EXPERT: ["search_documents", "generate_code"],
                AgentRole.STANDARDS_ENFORCER: ["validate_code", "analyze_code"],
            }
            return role_tool_mapping.get(role, [])

        agent_configs = {
            # Core System Agents
            AgentRole.SUPERVISOR: {
                "name": "Supervisor Agent",
                "type": AgentType.SUPERVISOR,
                "description": "Central coordinator that routes tasks to appropriate specialists and manages multi-agent workflows using strategic decision-making.",
                "capabilities": [
                    "Task Routing",
                    "Workflow Coordination",
                    "Agent Management",
                    "Decision Making",
                    "Multi-Agent Orchestration",
                    "Strategic Planning",
                ],
                "team": "Core System",
            },
            # Research and Analysis Team
            AgentRole.RESEARCH: {
                "name": "Research Expert",
                "type": AgentType.RESEARCH,
                "description": "Specialized in information gathering, document search, web research, and comprehensive analysis to provide strategic insights.",
                "capabilities": [
                    "Document Search",
                    "Web Research",
                    "Information Synthesis",
                    "Knowledge Retrieval",
                    "Data Analysis",
                    "Trend Analysis",
                ],
                "team": "Research & Analysis",
            },
            AgentRole.DOCUMENT_ANALYZER: {
                "name": "Document Analyzer",
                "type": AgentType.DOCUMENT_ANALYZER,
                "description": "Expert in analyzing, processing, and extracting insights from various document formats including PDFs, Word docs, and technical specifications.",
                "capabilities": [
                    "Document Processing",
                    "Content Extraction",
                    "Semantic Analysis",
                    "Metadata Extraction",
                    "Format Conversion",
                    "Quality Assessment",
                ],
                "team": "Research & Analysis",
            },
            # Strategic Planning Team
            AgentRole.STRATEGIC_EXPERT: {
                "name": "Strategic Expert",
                "type": AgentType.STRATEGIC,
                "description": "Provides strategic guidance using 'Playing to Win' framework, architecture decisions, and high-level planning for AI systems.",
                "capabilities": [
                    "Strategic Planning",
                    "Architecture Design",
                    "Best Practices",
                    "Technical Leadership",
                    "Framework Analysis",
                    "Decision Making",
                ],
                "team": "Strategic Planning",
            },
            AgentRole.TRD_CONVERTER: {
                "name": "TRD Converter",
                "type": AgentType.TRD_CONVERTER,
                "description": "Converts technical requirements into structured Technical Requirements Documents (TRDs) with comprehensive specifications.",
                "capabilities": [
                    "Requirements Analysis",
                    "TRD Generation",
                    "Specification Writing",
                    "Technical Documentation",
                    "Validation",
                    "Standards Compliance",
                ],
                "team": "Strategic Planning",
            },
            # Framework Experts Team
            AgentRole.LANGCHAIN_EXPERT: {
                "name": "LangChain Expert",
                "type": AgentType.LANGCHAIN,
                "description": "Expert in LangChain framework, chains, agents, and LLM application development with deep knowledge of the ecosystem.",
                "capabilities": [
                    "Chain Creation",
                    "Agent Development",
                    "LLM Integration",
                    "Framework Guidance",
                    "Tool Integration",
                    "Memory Systems",
                ],
                "team": "Framework Experts",
            },
            AgentRole.LANGGRAPH_EXPERT: {
                "name": "LangGraph Expert",
                "type": AgentType.LANGGRAPH,
                "description": "Specialized in LangGraph workflows, state management, and complex multi-agent graph orchestration patterns.",
                "capabilities": [
                    "Graph Workflows",
                    "State Management",
                    "Node Orchestration",
                    "Conditional Logic",
                    "Parallel Processing",
                    "Workflow Optimization",
                ],
                "team": "Framework Experts",
            },
            AgentRole.LLAMAINDEX_EXPERT: {
                "name": "LlamaIndex Expert",
                "type": AgentType.LLAMAINDEX,
                "description": "Expert in LlamaIndex for data ingestion, indexing, and retrieval-augmented generation (RAG) implementations.",
                "capabilities": [
                    "Data Ingestion",
                    "Vector Indexing",
                    "RAG Implementation",
                    "Query Engines",
                    "Document Loaders",
                    "Retrieval Optimization",
                ],
                "team": "Framework Experts",
            },
            AgentRole.PYDANTIC_EXPERT: {
                "name": "Pydantic Expert",
                "type": AgentType.PYDANTIC,
                "description": "Specialized in Pydantic models, data validation, serialization, and type-safe Python development patterns.",
                "capabilities": [
                    "Data Modeling",
                    "Validation Logic",
                    "Serialization",
                    "Type Safety",
                    "Schema Generation",
                    "API Integration",
                ],
                "team": "Framework Experts",
            },
            AgentRole.AGNO_EXPERT: {
                "name": "Agno Expert",
                "type": AgentType.AGNO,
                "description": "Expert in Agno framework for AI agent development, workflow automation, and intelligent system orchestration.",
                "capabilities": [
                    "Agent Development",
                    "Workflow Automation",
                    "System Integration",
                    "AI Orchestration",
                    "Process Optimization",
                    "Framework Implementation",
                ],
                "team": "Framework Experts",
            },
            AgentRole.CREWAI_EXPERT: {
                "name": "CrewAI Expert",
                "type": AgentType.CREWAI,
                "description": "Specialized in CrewAI framework for multi-agent collaboration, role-based AI systems, and crew management.",
                "capabilities": [
                    "Multi-Agent Systems",
                    "Role Definition",
                    "Crew Management",
                    "Task Delegation",
                    "Collaboration Patterns",
                    "Agent Communication",
                ],
                "team": "Framework Experts",
            },
            # Technology Specialists Team
            AgentRole.FASTAPI_EXPERT: {
                "name": "FastAPI Expert",
                "type": AgentType.FASTAPI,
                "description": "Specialized in FastAPI development, API design, async programming, and high-performance backend architecture.",
                "capabilities": [
                    "API Development",
                    "Backend Architecture",
                    "Performance Optimization",
                    "Documentation",
                    "Async Programming",
                    "Security Implementation",
                ],
                "team": "Technology Specialists",
            },
            AgentRole.NEXTJS_EXPERT: {
                "name": "Next.js Expert",
                "type": AgentType.NEXTJS,
                "description": "Expert in Next.js, React, modern frontend development, SSR/SSG, and full-stack JavaScript applications.",
                "capabilities": [
                    "Frontend Development",
                    "React Components",
                    "SSR/SSG",
                    "UI/UX Design",
                    "Performance Optimization",
                    "SEO Implementation",
                ],
                "team": "Technology Specialists",
            },
            # Implementation Team
            AgentRole.IMPLEMENTATION: {
                "name": "Implementation Expert",
                "type": AgentType.IMPLEMENTATION,
                "description": "Focused on translating designs and requirements into working code implementations across multiple technologies.",
                "capabilities": [
                    "Code Implementation",
                    "System Integration",
                    "Technology Translation",
                    "Architecture Realization",
                    "Testing",
                    "Deployment",
                ],
                "team": "Implementation Team",
            },
            AgentRole.CODE_GENERATOR: {
                "name": "Code Generator",
                "type": AgentType.CODE_GENERATOR,
                "description": "Specialized in generating high-quality, production-ready code from specifications using best practices and patterns.",
                "capabilities": [
                    "Code Generation",
                    "Template Creation",
                    "Pattern Implementation",
                    "Best Practices",
                    "Code Optimization",
                    "Framework Integration",
                ],
                "team": "Implementation Team",
            },
            AgentRole.CODE_REVIEWER: {
                "name": "Code Reviewer",
                "type": AgentType.CODE_REVIEWER,
                "description": "Expert in code review, quality assurance, security analysis, and ensuring adherence to coding standards.",
                "capabilities": [
                    "Code Review",
                    "Quality Assurance",
                    "Security Analysis",
                    "Performance Review",
                    "Standards Compliance",
                    "Best Practices Enforcement",
                ],
                "team": "Implementation Team",
            },
            AgentRole.STANDARDS_ENFORCER: {
                "name": "Standards Enforcer",
                "type": AgentType.STANDARDS_ENFORCER,
                "description": "Ensures compliance with coding standards, architectural patterns, and organizational best practices across all implementations.",
                "capabilities": [
                    "Standards Enforcement",
                    "Compliance Checking",
                    "Pattern Validation",
                    "Quality Gates",
                    "Policy Implementation",
                    "Governance",
                ],
                "team": "Implementation Team",
            },
            # Documentation Team
            AgentRole.DOCUMENTATION: {
                "name": "Documentation Expert",
                "type": AgentType.DOCUMENTATION,
                "description": "Specialized in creating comprehensive, clear, and maintainable documentation for technical systems and processes.",
                "capabilities": [
                    "Technical Writing",
                    "API Documentation",
                    "User Guides",
                    "Architecture Documentation",
                    "Process Documentation",
                    "Knowledge Management",
                ],
                "team": "Documentation Team",
            },
            AgentRole.PROMPT_ENGINEER: {
                "name": "Prompt Engineer",
                "type": AgentType.PROMPT_ENGINEER,
                "description": "Expert in creating optimized prompts using industry-leading techniques from Cursor, v0, Claude, and other top AI tools.",
                "capabilities": [
                    "Prompt Optimization",
                    "AI Interaction Design",
                    "Performance Tuning",
                    "Best Practices",
                    "Industry Patterns",
                    "Technique Integration",
                ],
                "team": "Documentation Team",
            },
        }

        # Create agents for all defined configurations
        for role, config in agent_configs.items():
            agent_id = f"agent_{role.value.lower()}"
            tools = get_agent_tools(role)

            self.agents[agent_id] = AgentInfo(
                id=agent_id,
                name=config["name"],
                type=config["type"],
                description=config["description"],
                capabilities=config["capabilities"],
                tools=tools,
                team=config["team"],
                status=AgentStatus.IDLE,
                performance=AgentPerformance(),
            )
            self.task_history[agent_id] = []

        # Also create agents for any AgentRole that doesn't have a config yet
        # This ensures we don't miss any agents defined in the enum
        for role in AgentRole:
            agent_id = f"agent_{role.value.lower()}"
            if agent_id not in self.agents:
                # Create a basic agent configuration for roles not explicitly configured
                tools = get_agent_tools(role)
                self.agents[agent_id] = AgentInfo(
                    id=agent_id,
                    name=role.value.replace("_", " ").title() + " Agent",
                    type=AgentType.CODING,  # Default type
                    description=f"Agent specialized in {role.value.replace('_', ' ').lower()} tasks and operations.",
                    capabilities=[role.value.replace("_", " ").title()],
                    tools=tools,
                    team="General",
                    status=AgentStatus.IDLE,
                    performance=AgentPerformance(),
                )
                self.task_history[agent_id] = []

    async def get_all_agents(self) -> List[AgentInfo]:
        """Get all agent information."""
        return list(self.agents.values())

    async def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get specific agent information."""
        return self.agents.get(agent_id)

    async def update_agent_status(
        self, agent_id: str, status: AgentStatus, current_task: Optional[str] = None
    ):
        """Update agent status."""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            self.agents[agent_id].last_active = datetime.utcnow()
            if current_task:
                self.agents[agent_id].current_task = current_task
            elif status == AgentStatus.IDLE:
                self.agents[agent_id].current_task = None

    async def record_task_completion(
        self, agent_id: str, task_description: str, success: bool, duration: float
    ):
        """Record task completion for performance tracking."""
        if agent_id not in self.agents:
            return

        # Update task history
        task_record = {
            "task": task_description,
            "success": success,
            "duration": duration,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.task_history[agent_id].append(task_record)

        # Keep only last 100 tasks
        if len(self.task_history[agent_id]) > 100:
            self.task_history[agent_id] = self.task_history[agent_id][-100:]

        # Update performance metrics
        await self._update_performance_metrics(agent_id)

        # Update status back to idle
        await self.update_agent_status(agent_id, AgentStatus.IDLE)

    async def _update_performance_metrics(self, agent_id: str):
        """Update performance metrics based on task history."""
        if agent_id not in self.task_history:
            return

        history = self.task_history[agent_id]
        if not history:
            return

        agent = self.agents[agent_id]

        # Calculate metrics
        total_tasks = len(history)
        successful_tasks = sum(1 for task in history if task["success"])
        success_rate = (
            (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 100.0
        )

        # Calculate average response time
        durations = [task["duration"] for task in history if "duration" in task]
        avg_duration = sum(durations) / len(durations) if durations else 0.5
        avg_response_time = f"{avg_duration:.1f}s"

        # Calculate uptime (simplified - based on recent activity)
        recent_tasks = [
            task
            for task in history
            if datetime.fromisoformat(task["timestamp"])
            > datetime.utcnow() - timedelta(hours=24)
        ]
        uptime = min(100.0, len(recent_tasks) * 10)  # Simplified calculation

        # Update performance
        agent.performance = AgentPerformance(
            tasks_completed=total_tasks,
            success_rate=round(success_rate, 1),
            avg_response_time=avg_response_time,
            uptime=f"{uptime:.0f}%",
        )

    async def get_agent_by_role(self, role: AgentRole) -> Optional[AgentInfo]:
        """Get agent by role."""
        agent_id = f"agent_{role.value.lower()}"
        return self.agents.get(agent_id)

    async def get_agents_by_team(self, team: str) -> List[AgentInfo]:
        """Get all agents in a specific team."""
        return [agent for agent in self.agents.values() if agent.team == team]

    async def get_teams(self) -> List[str]:
        """Get all unique team names."""
        teams = set()
        for agent in self.agents.values():
            if agent.team:
                teams.add(agent.team)
        return sorted(list(teams))

    async def start_task(self, agent_id: str, task_description: str) -> str:
        """Start a task for an agent and return task ID."""
        task_id = str(uuid.uuid4())
        await self.update_agent_status(
            agent_id, AgentStatus.PROCESSING, task_description
        )

        if self.logger:
            self.logger.info(f"Agent {agent_id} started task: {task_description}")

        return task_id

    async def get_agent_statistics(self) -> Dict[str, Any]:
        """Get overall agent statistics."""
        total_agents = len(self.agents)
        active_agents = sum(
            1 for agent in self.agents.values() if agent.status == AgentStatus.ACTIVE
        )
        processing_agents = sum(
            1
            for agent in self.agents.values()
            if agent.status == AgentStatus.PROCESSING
        )
        idle_agents = sum(
            1 for agent in self.agents.values() if agent.status == AgentStatus.IDLE
        )
        error_agents = sum(
            1 for agent in self.agents.values() if agent.status == AgentStatus.ERROR
        )

        total_tasks = sum(len(history) for history in self.task_history.values())
        successful_tasks = sum(
            sum(1 for task in history if task["success"])
            for history in self.task_history.values()
        )

        # Team statistics
        teams = await self.get_teams()
        team_stats = {}
        for team in teams:
            team_agents = await self.get_agents_by_team(team)
            team_stats[team] = {
                "agent_count": len(team_agents),
                "active_agents": sum(
                    1 for agent in team_agents if agent.status == AgentStatus.ACTIVE
                ),
                "processing_agents": sum(
                    1 for agent in team_agents if agent.status == AgentStatus.PROCESSING
                ),
                "idle_agents": sum(
                    1 for agent in team_agents if agent.status == AgentStatus.IDLE
                ),
            }

        # Tool usage statistics
        all_tools = set()
        for agent in self.agents.values():
            all_tools.update(agent.tools)

        tool_usage = {}
        for tool in all_tools:
            agents_with_tool = [
                agent for agent in self.agents.values() if tool in agent.tools
            ]
            tool_usage[tool] = {
                "agent_count": len(agents_with_tool),
                "agents": [agent.name for agent in agents_with_tool],
            }

        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "processing_agents": processing_agents,
            "idle_agents": idle_agents,
            "error_agents": error_agents,
            "total_tasks_completed": total_tasks,
            "overall_success_rate": (
                (successful_tasks / total_tasks * 100) if total_tasks > 0 else 100.0
            ),
            "teams": team_stats,
            "total_teams": len(teams),
            "tool_usage": tool_usage,
            "total_tools": len(all_tools),
        }


# Global agent manager instance
agent_manager = AgentManager()

# Export
__all__ = [
    "AgentManager",
    "AgentInfo",
    "AgentType",
    "AgentStatus",
    "AgentPerformance",
    "agent_manager",
]
