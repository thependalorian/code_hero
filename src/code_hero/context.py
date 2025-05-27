"""Context management for state across the application."""

from typing import Dict, Any, Optional, TypeVar, Generic, Tuple, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import Request

from .state import BaseState, Status
from .logger import StructuredLogger
from .manager import StateManager
from .supervisor import SupervisorAgent

T = TypeVar("T", bound=BaseState)


class StateContext(Generic[T]):
    """Context manager for handling state."""

    def __init__(self, state: T, manager: StateManager, logger: StructuredLogger):
        """Initialize the state context.
        
        Args:
            state: State to manage
            manager: State manager instance
            logger: Logger instance
        """
        self.state = state
        self.manager = manager
        self.logger = logger
        self.start_time = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}

    async def __aenter__(self) -> "StateContext[T]":
        """Enter the context.
        
        Returns:
            Self reference for context management
            
        Raises:
            Exception: If context entry fails
        """
        try:
            # Log entry
            self.logger.log_event(
                "context_enter",
                {
                    "state_id": self.state.id,
                    "state_type": self.state.__class__.__name__,
                    "start_time": self.start_time.isoformat(),
                },
                self.state,
            )
            
            # Initialize state if needed
            if hasattr(self.state, "initialize") and callable(self.state.initialize):
                await self.state.initialize()
                
            return self

        except Exception as e:
            self.logger.log_error(e, self.state)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context.
        
        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
            
        Raises:
            Exception: If context exit fails
        """
        try:
            end_time = datetime.utcnow()
            duration = (end_time - self.start_time).total_seconds()

            # Update state
            self.state.updated_at = end_time
            if exc_type:
                self.state.status = Status.FAILED
                self.metadata["error"] = str(exc_val)

            # Cleanup state if needed
            if hasattr(self.state, "cleanup") and callable(self.state.cleanup):
                await self.state.cleanup()

            # Update state in manager
            await self.manager.update_state(self.state)

            # Log exit
            self.logger.log_event(
                "context_exit",
                {
                    "state_id": self.state.id,
                    "state_type": self.state.__class__.__name__,
                    "end_time": end_time.isoformat(),
                    "duration": duration,
                    "status": self.state.status,
                    "metadata": self.metadata,
                    "error": str(exc_val) if exc_val else None,
                },
                self.state,
            )

        except Exception as e:
            self.logger.log_error(e, self.state)
            raise

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the context.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Optional[Any]:
        """Get metadata from the context.
        
        Args:
            key: Metadata key
            
        Returns:
            Metadata value if found, None otherwise
        """
        return self.metadata.get(key)

    def update_state(self, **kwargs: Any) -> None:
        """Update the state with new values.
        
        Args:
            **kwargs: Key-value pairs to update in the state
        """
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
                
    async def refresh_state(self) -> None:
        """Refresh the state from the manager."""
        updated_state = await self.manager.get_state(self.state.id)
        if updated_state:
            self.state = updated_state


@asynccontextmanager
async def managed_state(state: T) -> AsyncGenerator[StateContext[T], None]:
    """Create a managed state context.

    Args:
        state: State to manage

    Yields:
        Managed state context
        
    Raises:
        Exception: If context management fails
    """
    if not hasattr(state, "request"):
        raise ValueError("State must have a 'request' attribute")
        
    services = await get_services(state.request)
    manager, logger, _ = services
    context = StateContext(state, manager, logger)
    
    try:
        async with context as ctx:
            yield ctx
    except Exception as e:
        logger.log_error(e, state)
        raise


async def get_services(
    request: Request,
) -> Tuple[StateManager, StructuredLogger, SupervisorAgent]:
    """Get service instances from app state.

    Args:
        request: FastAPI request object

    Returns:
        Tuple of service instances
        
    Raises:
        ValueError: If required services are not found
    """
    if not hasattr(request.app.state, "state_manager"):
        raise ValueError("State manager not initialized")
    if not hasattr(request.app.state, "logger"):
        raise ValueError("Logger not initialized")
    if not hasattr(request.app.state, "supervisor"):
        raise ValueError("Supervisor not initialized")
        
    return (
        request.app.state.state_manager,
        request.app.state.logger,
        request.app.state.supervisor,
    )


# Export all symbols
__all__ = ["StateContext", "managed_state", "get_services"]
