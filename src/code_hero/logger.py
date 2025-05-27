"""Structured logging for the backend system."""

import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

from .config import LoggingConfig
from .state import BaseState
from .interfaces import ServiceInterface


class StructuredLogger(ServiceInterface):
    """Structured logging with state tracking."""

    def __init__(self, config: LoggingConfig):
        """Initialize structured logger."""
        self.config = config
        self._initialized = False
        self.logger = logging.getLogger(__name__)

        # Configure base logging
        logging.basicConfig(
            level=config.level,
            format=config.format,
            filename=config.file
        )

    async def initialize(self) -> None:
        """Initialize the logger."""
        if self._initialized:
            return
            
        try:
            # Set up log directory if needed
            if self.config.file:
                log_path = Path(self.config.file).parent
                log_path.mkdir(parents=True, exist_ok=True)
            
            self._initialized = True
            self.info("Logger initialized successfully")
        except Exception as e:
            self.error(f"Failed to initialize logger: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up logger resources."""
        try:
            # Flush any pending logs
            for handler in self.logger.handlers:
                handler.flush()
                handler.close()
            
            self._initialized = False
            self.info("Logger cleaned up successfully")
        except Exception as e:
            self.error(f"Failed to clean up logger: {e}")
            raise

    async def check_health(self) -> dict:
        """Check logger health status."""
        return {
            "initialized": self._initialized,
            "level": self.config.level,
            "has_file_handler": bool(self.config.file),
            "handler_count": len(self.logger.handlers)
        }

    def log_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        state: Optional[BaseState] = None,
    ) -> None:
        """Log a structured event.

        Args:
            event_type: Type of event
            event_data: Event data
            state: Optional state context
        """
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "data": event_data,
            }
            
            if state:
                log_entry["state_id"] = state.id
                log_entry["state_type"] = state.__class__.__name__
                
            self.logger.info(json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Failed to log event: {e}")

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message.

        Args:
            message: Info message to log
            **kwargs: Additional logging arguments
        """
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "message": message,
                **kwargs
            }
            self.logger.info(json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Failed to log info: {e}")

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message.

        Args:
            message: Error message to log
            **kwargs: Additional logging arguments
        """
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "ERROR",
                "message": message,
                **kwargs
            }
            self.logger.error(json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Failed to log error: {e}")

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message.

        Args:
            message: Warning message to log
            **kwargs: Additional logging arguments
        """
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "WARNING",
                "message": message,
                **kwargs
            }
            self.logger.warning(json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Failed to log warning: {e}")

    def log_warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message with extra context.

        Args:
            message: Warning message to log
            extra: Additional context data
        """
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "WARNING",
                "message": message,
            }
            
            if extra:
                log_entry.update(extra)
                
            self.logger.warning(json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Failed to log warning: {e}")

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message.

        Args:
            message: Debug message to log
            **kwargs: Additional logging arguments
        """
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "DEBUG",
                "message": message,
                **kwargs
            }
            self.logger.debug(json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Failed to log debug: {e}")

    def log_error(
        self,
        error: Exception,
        state: Optional[BaseState] = None,
    ) -> None:
        """Log an error with context.

        Args:
            error: Exception to log
            state: Optional state context
        """
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "error_type": error.__class__.__name__,
                "error_message": str(error),
            }
            
            if state:
                log_entry["state_id"] = state.id
                log_entry["state_type"] = state.__class__.__name__
                
            self.logger.error(json.dumps(log_entry))
        except Exception as e:
            self.logger.error(f"Failed to log error: {e}")

    def _format_state(self, state: BaseState) -> Dict[str, Any]:
        """Format a state object for logging."""
        state_dict = state.dict()
        state_dict["timestamp"] = datetime.utcnow().isoformat()
        return state_dict

    def log_state(
        self, state: BaseState, level: str = "INFO", message: Optional[str] = None
    ) -> None:
        """Log a state object with optional message."""
        try:
            # Format state for logging
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "state_id": state.id,
                "state_type": state.__class__.__name__,
                "state_data": self._format_state(state),
                "message": message,
            }

            # Log at appropriate level
            log_func = getattr(self.logger, level.lower())
            log_func(json.dumps(log_data))

        except Exception as e:
            self.logger.error(f"Failed to log state: {e}")

    def log_state_change(
        self, old_state: BaseState, new_state: BaseState, level: str = "INFO"
    ) -> None:
        """Log a state change."""
        try:
            # Format states for comparison
            old_data = self._format_state(old_state)
            new_data = self._format_state(new_state)

            # Find changed fields
            changes = {}
            for key in old_data:
                if key in new_data and old_data[key] != new_data[key]:
                    changes[key] = {"old": old_data[key], "new": new_data[key]}

            # Log the changes
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "state_id": new_state.id,
                "state_type": new_state.__class__.__name__,
                "changes": changes,
            }

            log_func = getattr(self.logger, level.lower())
            log_func(json.dumps(log_data))

        except Exception as e:
            self.logger.error(f"Failed to log state change: {e}")

    def log_metric(
        self, metric_name: str, value: Any, state: Optional[BaseState] = None
    ) -> None:
        """Log a metric with optional state context."""
        try:
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "metric": metric_name,
                "value": value,
                "state_context": self._format_state(state) if state else None,
            }

            self.logger.info(json.dumps(log_data))

        except Exception as e:
            self.logger.error(f"Failed to log metric: {e}")

    async def __call__(self, event_type: str, event_data: Dict[str, Any], state: Optional[BaseState] = None, **kwargs: Any) -> None:
        """Execute logging operation.
        
        Args:
            event_type: Type of event to log
            event_data: Event data to log
            state: Optional state context
            **kwargs: Additional arguments
            
        Raises:
            Exception: If logging fails
        """
        try:
            # Log the event
            self.log_event(event_type, event_data, state)
        except Exception as e:
            self.logger.error(f"Logging operation failed: {e}")
            raise

# Export all symbols
__all__ = ["StructuredLogger"]
