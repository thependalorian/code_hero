"""Service type definitions for the backend system.

This module provides type definitions and protocols for the service architecture.
It helps maintain type safety and enables proper dependency injection throughout
the system.

Key Components:
    - Service type variables for dependency injection
    - ServiceProvider protocol for service instantiation

Example:
    ```python
    class MyServiceProvider(ServiceProvider):
        async def get_services(
            self, request: Request
        ) -> Tuple[StateManager, StructuredLogger, SupervisorAgent]:
            # Initialize and return service instances
            return state_manager, logger, supervisor
    ```
"""

from typing import Protocol, Tuple, TypeVar

from fastapi import Request

# Forward declarations to avoid circular imports
try:
    from .logger import StructuredLogger
    from .manager import StateManager
    from .supervisor import SupervisorAgent
except ImportError:
    # Handle circular import gracefully
    StateManager = None
    StructuredLogger = None
    SupervisorAgent = None

# Type variables for service instances
StateManagerType = TypeVar("StateManagerType")
LoggerType = TypeVar("LoggerType")
SupervisorType = TypeVar("SupervisorType")


class ServiceProvider(Protocol):
    """Protocol for service providers.

    This protocol defines the interface for service instantiation and
    dependency injection. It ensures consistent service initialization
    and proper typing across the system.

    Methods:
        get_services: Retrieve service instances for dependency injection
    """

    async def get_services(self, request: Request) -> Tuple:
        """Get service instances.

        This method should:
            1. Initialize required services
            2. Validate service health
            3. Return service tuple in correct order

        Args:
            request: FastAPI request object for context

        Returns:
            Tuple containing service instances

        Raises:
            ServiceError: If service initialization fails
        """
        ...


# Export all symbols
__all__ = ["StateManagerType", "LoggerType", "SupervisorType", "ServiceProvider"]
