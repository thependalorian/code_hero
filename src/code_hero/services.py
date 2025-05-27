"""Service management for the application."""

from typing import Tuple, TYPE_CHECKING
from fastapi import Request, HTTPException

from .state import Status
from .interfaces import ServiceInterface, ServiceStatus
from .types import StateManagerType, LoggerType, SupervisorType

if TYPE_CHECKING:
    from .manager import StateManager
    from .logger import StructuredLogger
    from .supervisor import SupervisorAgent

class ServiceError(Exception):
    """Base exception for service-related errors."""
    pass

class ServiceNotInitializedError(ServiceError):
    """Exception raised when a required service is not initialized."""
    pass

class ServiceHealthCheckError(ServiceError):
    """Exception raised when a service health check fails."""
    pass

async def validate_services(
    state_manager: StateManagerType,
    logger: LoggerType,
    supervisor: SupervisorType,
) -> list[ServiceStatus]:
    """Validate all services are properly initialized and healthy.
    
    Args:
        state_manager: State manager instance
        logger: Logger instance
        supervisor: Supervisor instance
        
    Returns:
        List of service status checks
        
    Raises:
        ServiceError: If service validation fails
    """
    services = [
        ("state_manager", state_manager),
        ("logger", logger),
        ("supervisor", supervisor)
    ]
    
    statuses = []
    for name, service in services:
        try:
            if not isinstance(service, ServiceInterface):
                raise ServiceError(f"Service {name} does not implement ServiceInterface")
                
            health = await service.check_health()
            statuses.append(
                ServiceStatus(
                    name=name,
                    status=Status.RUNNING,
                    details=health
                )
            )
        except Exception as e:
            statuses.append(
                ServiceStatus(
                    name=name,
                    status=Status.FAILED,
                    error=str(e)
                )
            )
    
    return statuses

async def get_services(
    request: Request,
) -> Tuple[StateManagerType, LoggerType, SupervisorType]:
    """Get service instances from app state.

    Args:
        request: FastAPI request object

    Returns:
        Tuple of service instances
        
    Raises:
        HTTPException: If services are not properly initialized
    """
    try:
        if not hasattr(request.app.state, "state_manager"):
            raise ServiceNotInitializedError("State manager not initialized")
        if not hasattr(request.app.state, "logger"):
            raise ServiceNotInitializedError("Logger not initialized")
        if not hasattr(request.app.state, "supervisor"):
            raise ServiceNotInitializedError("Supervisor not initialized")
            
        services = (
            request.app.state.state_manager,
            request.app.state.logger,
            request.app.state.supervisor,
        )
        
        # Validate services
        statuses = await validate_services(*services)
        failed = [s for s in statuses if s.status == Status.FAILED]
        
        if failed:
            errors = "; ".join(f"{s.name}: {s.error}" for s in failed)
            raise ServiceHealthCheckError(f"Service validation failed: {errors}")
            
        return services
        
    except ServiceError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service error: {str(e)}")

# Export all symbols
__all__ = [
    "ServiceError",
    "ServiceNotInitializedError", 
    "ServiceHealthCheckError",
    "validate_services",
    "get_services"
] 