"""Graph router implementation for the Strategic Framework.

This module handles all workflow-related endpoints, including workflow creation,
execution, and state management.
"""

from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from .context import get_services, managed_state
from .logger import StructuredLogger
from .manager import StateManager
from .node import BaseNode
from .state import Status, Workflow, WorkflowState
from .workflow import WorkflowConfig, WorkflowRunner

router = APIRouter()


@router.post("/workflow")
async def create_workflow(
    config: WorkflowConfig,
    services: tuple[StateManager, StructuredLogger] = Depends(get_services),
) -> Dict:
    """Create a new workflow.

    Args:
        config: Workflow configuration
        services: Injected services

    Returns:
        New workflow details
    """
    state_manager, logger = services

    try:
        workflow = Workflow(
            name=config.name, description=config.description, nodes=config.nodes
        )

        async with managed_state(WorkflowState(workflow=workflow)) as state:
            workflow_id = await state_manager.create_workflow(state)
            return {
                "workflow_id": workflow_id,
                "status": Status.CREATED,
                "config": config.dict(),
            }
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/{workflow_id}/node")
async def add_node(
    workflow_id: str,
    node: BaseNode,
    services: tuple[StateManager, StructuredLogger] = Depends(get_services),
) -> Dict:
    """Add a node to a workflow.

    Args:
        workflow_id: Workflow ID
        node: Node to add
        services: Injected services

    Returns:
        Updated workflow details
    """
    state_manager, logger = services

    try:
        workflow_state = await state_manager.get_workflow_state(workflow_id)
        if not workflow_state:
            raise HTTPException(status_code=404, detail="Workflow not found")

        async with managed_state(workflow_state) as state:
            state.workflow.nodes.append(node.name)
            await state_manager.update_workflow(state)
            return {
                "workflow_id": workflow_id,
                "node": node.name,
                "status": Status.UPDATED,
            }
    except Exception as e:
        logger.error(f"Failed to add node: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    initial_state: Optional[Dict] = None,
    services: tuple[StateManager, StructuredLogger] = Depends(get_services),
) -> Dict:
    """Execute a workflow.

    Args:
        workflow_id: Workflow ID
        initial_state: Optional initial state
        services: Injected services

    Returns:
        Workflow execution results
    """
    state_manager, logger = services

    try:
        workflow_state = await state_manager.get_workflow_state(workflow_id)
        if not workflow_state:
            raise HTTPException(status_code=404, detail="Workflow not found")

        runner = WorkflowRunner(workflow_state.workflow)

        async with managed_state(workflow_state) as state:
            if initial_state:
                state.update(initial_state)

            final_state = await runner.invoke(state)
            return {
                "workflow_id": workflow_id,
                "status": final_state.status,
                "results": final_state.artifacts,
            }
    except Exception as e:
        logger.error(f"Failed to execute workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    services: tuple[StateManager, StructuredLogger] = Depends(get_services),
) -> Dict:
    """Get workflow details.

    Args:
        workflow_id: Workflow ID
        services: Injected services

    Returns:
        Workflow details
    """
    state_manager, logger = services

    try:
        workflow_state = await state_manager.get_workflow_state(workflow_id)
        if not workflow_state:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {
            "workflow_id": workflow_id,
            "workflow": workflow_state.workflow.dict(),
            "status": workflow_state.status,
        }
    except Exception as e:
        logger.error(f"Failed to get workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/workflow/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    services: tuple[StateManager, StructuredLogger] = Depends(get_services),
) -> Dict:
    """Delete a workflow.

    Args:
        workflow_id: Workflow ID
        services: Injected services

    Returns:
        Deletion status
    """
    state_manager, logger = services

    try:
        success = await state_manager.delete_workflow(workflow_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {"workflow_id": workflow_id, "status": "deleted"}
    except Exception as e:
        logger.error(f"Failed to delete workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router
__all__ = ["router"]
