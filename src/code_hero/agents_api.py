"""API endpoints for agent management and monitoring."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from .agent_manager import agent_manager
from .context import get_services
from .logger import StructuredLogger
from .manager import StateManager
from .state import AgentInfoExtended as AgentInfo
from .state import AgentStatus
from .supervisor import SupervisorAgent

router = APIRouter()


@router.get("/", response_model=List[AgentInfo])
async def get_all_agents(
    request: Request,
    services: tuple[StateManager, StructuredLogger, SupervisorAgent] = Depends(
        get_services
    ),
) -> List[AgentInfo]:
    """Get all agents with their current status and performance metrics."""
    try:
        agents = await agent_manager.get_all_agents()
        return agents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agents: {str(e)}")


@router.get("/{agent_id}", response_model=AgentInfo)
async def get_agent(
    agent_id: str,
    request: Request,
    services: tuple[StateManager, StructuredLogger, SupervisorAgent] = Depends(
        get_services
    ),
) -> AgentInfo:
    """Get specific agent information."""
    try:
        agent = await agent_manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        return agent
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")


@router.post("/{agent_id}/interact")
async def interact_with_agent(
    agent_id: str,
    request: Request,
    services: tuple[StateManager, StructuredLogger, SupervisorAgent] = Depends(
        get_services
    ),
) -> Dict[str, Any]:
    """Interact with a specific agent directly."""
    state_manager, logger, supervisor = services

    try:
        # Parse request body
        body = await request.json()
        message = body.get("message", "")

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Get agent info
        agent = await agent_manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # Start task tracking
        task_id = await agent_manager.start_task(
            agent_id, f"Direct interaction: {message}"
        )
        start_time = datetime.now()

        try:
            # Map agent ID to role for execution
            role_mapping = {
                "agent_supervisor": "supervisor",
                "agent_research": "research",
                "agent_langchain_expert": "langchain_expert",
                "agent_langgraph_expert": "langgraph_expert",
                "agent_llamaindex_expert": "llamaindex_expert",
                "agent_fastapi_expert": "fastapi_expert",
                "agent_nextjs_expert": "nextjs_expert",
                "agent_pydantic_expert": "pydantic_expert",
                "agent_agno_expert": "agno_expert",
                "agent_crewai_expert": "crewai_expert",
                "agent_strategic_expert": "strategic_expert",
                "agent_prompt_engineer": "prompt_engineer",
                "agent_implementation": "implementation",
                "agent_documentation": "documentation",
                "agent_trd_converter": "trd_converter",
                "agent_code_generator": "code_generator",
                "agent_code_reviewer": "code_reviewer",
                "agent_standards_enforcer": "standards_enforcer",
                "agent_document_analyzer": "document_analyzer",
            }

            agent_role = role_mapping.get(agent_id)
            if not agent_role:
                raise ValueError(f"Unknown agent role for {agent_id}")

            # Execute through supervisor dispatch
            from .state import AgentRole, AgentState, Status

            # Create agent state for execution
            agent_state = AgentState(
                id=task_id,
                agent=AgentRole(agent_role),
                status=Status.PENDING,
                context={"message": message, "direct_interaction": True},
            )

            # Execute the task
            result = await supervisor.dispatch_task(
                project_id="direct_interaction", task=agent_state
            )

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Record completion
            success = isinstance(result, str) and len(result) > 0
            await agent_manager.record_task_completion(
                agent_id, f"Direct interaction: {message}", success, duration
            )

            return {
                "success": True,
                "agent_id": agent_id,
                "task_id": task_id,
                "response": result,
                "duration": f"{duration:.2f}s",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            # Record failure
            duration = (datetime.now() - start_time).total_seconds()
            await agent_manager.record_task_completion(
                agent_id, f"Direct interaction: {message}", False, duration
            )

            # Update agent status to error
            await agent_manager.update_agent_status(agent_id, AgentStatus.ERROR)

            raise HTTPException(
                status_code=500, detail=f"Agent interaction failed: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to interact with agent: {str(e)}"
        )


@router.get("/{agent_id}/history")
async def get_agent_history(
    agent_id: str,
    request: Request,
    limit: int = 10,
    services: tuple[StateManager, StructuredLogger, SupervisorAgent] = Depends(
        get_services
    ),
) -> Dict[str, Any]:
    """Get agent task history."""
    try:
        agent = await agent_manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # Get task history
        history = agent_manager.task_history.get(agent_id, [])

        # Return recent history
        recent_history = history[-limit:] if limit > 0 else history

        return {
            "agent_id": agent_id,
            "total_tasks": len(history),
            "recent_tasks": recent_history,
            "limit": limit,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get agent history: {str(e)}"
        )


@router.get("/statistics/overview")
async def get_agent_statistics(
    request: Request,
    services: tuple[StateManager, StructuredLogger, SupervisorAgent] = Depends(
        get_services
    ),
) -> Dict[str, Any]:
    """Get overall agent statistics."""
    try:
        stats = await agent_manager.get_agent_statistics()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get statistics: {str(e)}"
        )


@router.post("/{agent_id}/status")
async def update_agent_status(
    agent_id: str,
    status: AgentStatus,
    request: Request,
    current_task: Optional[str] = None,
    services: tuple[StateManager, StructuredLogger, SupervisorAgent] = Depends(
        get_services
    ),
) -> Dict[str, Any]:
    """Update agent status (for testing/admin purposes)."""
    try:
        agent = await agent_manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        await agent_manager.update_agent_status(agent_id, status, current_task)

        return {
            "success": True,
            "agent_id": agent_id,
            "new_status": status,
            "current_task": current_task,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update agent status: {str(e)}"
        )


# Export router
__all__ = ["router"]
