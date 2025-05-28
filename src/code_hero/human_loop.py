"""Human-in-the-loop integration for the backend system.

This module provides functionality for integrating human feedback into
automated workflows. It handles task escalation, review requests, and
feedback collection from human operators.

Key Components:
    - HumanLoopManager: Main class for managing human interactions
    - request_human_feedback: Function for human task handling
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from langgraph.func import task
    from langgraph.types import StreamWriter, interrupt

    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for older LangGraph versions
    LANGGRAPH_AVAILABLE = False

    def task():
        def decorator(func):
            return func

        return decorator

    class StreamWriter:
        async def write(self, data):
            pass

    def interrupt(data):
        return data


from .interfaces import ServiceInterface
from .state import (
    HumanFeedbackRequest,
    HumanFeedbackResponse,
    StateType,
    Status,
    TaskState,
)


class HumanLoopManager(ServiceInterface):
    """Manager for human-in-the-loop interactions.

    This class handles:
        - Managing feedback requests
        - Notifying human operators
        - Collecting responses
        - Tracking request status
    """

    def __init__(self):
        """Initialize the human loop manager."""
        self._initialized = False
        self._pending_requests: Dict[str, HumanFeedbackRequest] = {}
        self._completed_requests: Dict[str, HumanFeedbackResponse] = {}

    async def initialize(self) -> None:
        """Initialize the manager."""
        if self._initialized:
            return

        # Clear request queues
        self._pending_requests.clear()
        self._completed_requests.clear()

        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up manager resources."""
        # Archive pending requests
        self._pending_requests.clear()
        self._completed_requests.clear()
        self._initialized = False

    async def check_health(self) -> dict:
        """Check manager health status."""
        return {
            "initialized": self._initialized,
            "pending_requests": len(self._pending_requests),
            "completed_requests": len(self._completed_requests),
        }

    async def __call__(
        self,
        project_id: str,
        task: TaskState,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
        deadline: Optional[datetime] = None,
        **kwargs: Any,
    ) -> HumanFeedbackRequest:
        """Execute human loop operation.

        Args:
            project_id: Project identifier
            task: Task requiring feedback
            reason: Reason for feedback request
            context: Optional context information
            deadline: Optional deadline for feedback
            **kwargs: Additional arguments

        Returns:
            Created feedback request

        Raises:
            Exception: If operation fails
        """
        try:
            # Create feedback request
            return await self.create_request(
                project_id=project_id,
                task=task,
                reason=reason,
                context=context,
                deadline=deadline,
            )
        except Exception:
            # Note: Logger should be injected if needed for production use
            raise

    async def create_request(
        self,
        project_id: str,
        task: TaskState,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
        deadline: Optional[datetime] = None,
    ) -> HumanFeedbackRequest:
        """Create a new feedback request."""
        request = HumanFeedbackRequest(
            project_id=project_id,
            task_id=task.id,
            reason=reason,
            context=context or {},
            deadline=deadline,
        )

        self._pending_requests[request.task_id] = request
        return request

    async def get_pending_requests(self) -> List[HumanFeedbackRequest]:
        """Get all pending feedback requests."""
        return list(self._pending_requests.values())

    async def submit_response(
        self,
        request_id: str,
        feedback: str,
        approved: bool,
        comments: Optional[str] = None,
    ) -> HumanFeedbackResponse:
        """Submit a response to a feedback request."""
        if request_id not in self._pending_requests:
            raise ValueError(f"Request {request_id} not found")

        response = HumanFeedbackResponse(
            request_id=request_id,
            feedback=feedback,
            approved=approved,
            comments=comments,
        )

        # Move request from pending to completed
        del self._pending_requests[request_id]
        self._completed_requests[request_id] = response

        return response

    async def get_response(self, request_id: str) -> Optional[HumanFeedbackResponse]:
        """Get the response for a request if available."""
        return self._completed_requests.get(request_id)


@task()
async def request_human_feedback(
    state: StateType, *, writer: Optional[StreamWriter] = None
) -> StateType:
    """Request human feedback using LangGraph interrupt.

    Args:
        state: Current workflow state
        writer: Optional stream writer

    Returns:
        Updated workflow state with feedback
    """
    try:
        if writer:
            await writer.write(
                {
                    "type": "feedback_requested",
                    "data": {
                        "state_id": state.get("id"),
                        "context": state.get("context"),
                        "artifacts": state.get("artifacts"),
                    },
                }
            )

        # Request feedback via interrupt
        feedback = interrupt(
            {
                "state_id": state.get("id"),
                "prompt": state.get("context", {}).get("prompt", ""),
                "context": state.get("context", {}),
                "artifacts": state.get("artifacts", {}),
            }
        )

        # Update state with feedback
        state["context"]["human_feedback"] = feedback
        state["needs_feedback"] = False
        state["status"] = Status.COMPLETED

        if writer:
            await writer.write(
                {"type": "feedback_received", "data": {"feedback": feedback}}
            )

        return state

    except Exception as e:
        if writer:
            await writer.write({"type": "feedback_error", "data": {"error": str(e)}})
        state["status"] = Status.FAILED
        state["error"] = str(e)
        return state


# Export all symbols
__all__ = [
    "HumanLoopManager",
    "request_human_feedback",
    "HumanFeedbackRequest",
    "HumanFeedbackResponse",
]
