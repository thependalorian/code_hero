"""Service interfaces for the backend system.

This module defines the core interfaces and protocols that all services must implement.
It provides the foundation for our service architecture, ensuring consistency and
maintainability across the system.

Key Components:
    - ServiceInterface: Base protocol for all services
    - ServiceStatus: Health status model for services

Example:
    ```python
    @runtime_checkable
    class MyService(ServiceInterface):
        async def initialize(self) -> None:
            # Initialize resources
            pass
            
        async def cleanup(self) -> None:
            # Clean up resources
            pass
            
        async def check_health(self) -> dict:
            return {"status": "healthy"}
            
        async def __call__(self, *args, **kwargs) -> Any:
            # Service execution logic
            pass
    ```
"""

from typing import Protocol, runtime_checkable, Any
from pydantic import BaseModel

from .state import Status

@runtime_checkable
class ServiceInterface(Protocol):
    """Base interface that all services must implement.
    
    This protocol defines the core lifecycle methods that every service
    must provide. It ensures consistent initialization, cleanup, and
    health monitoring across all services.
    
    Methods:
        initialize: Set up service resources
        cleanup: Release service resources
        check_health: Monitor service health
        __call__: Execute service logic
    """
    
    async def initialize(self) -> None:
        """Initialize the service.
        
        This method should:
            1. Set up any required resources
            2. Initialize internal state
            3. Establish connections if needed
            4. Set service as initialized
            
        Raises:
            Exception: If initialization fails
        """
        ...
        
    async def cleanup(self) -> None:
        """Clean up service resources.
        
        This method should:
            1. Release any held resources
            2. Close connections
            3. Save state if needed
            4. Mark service as uninitialized
            
        Raises:
            Exception: If cleanup fails
        """
        ...
        
    async def check_health(self) -> dict:
        """Check service health status.
        
        Returns:
            dict: Health check information including:
                - initialization status
                - resource status
                - error information if any
                
        Raises:
            Exception: If health check fails
        """
        ...
        
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the service's main logic.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Service execution result
            
        Raises:
            Exception: If execution fails
        """
        ...

class ServiceStatus(BaseModel):
    """Service health status model.
    
    This model represents the health status of a service at a point in time.
    It includes the service name, current status, additional details, and
    any error information.
    
    Attributes:
        name: Service identifier
        status: Current service status
        details: Additional status information
        error: Error message if service is unhealthy
    """
    
    name: str
    status: Status
    details: dict = {}
    error: str | None = None

# Export all symbols
__all__ = ["ServiceInterface", "ServiceStatus"] 