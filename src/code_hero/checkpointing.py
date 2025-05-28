"""Enhanced Checkpointing System for Code Hero.

This module implements a comprehensive checkpointing system that integrates with
the existing Code Hero infrastructure, following LangGraph best practices and
supporting multiple storage backends for production use.

Features:
- Multiple checkpointer types (InMemory, SQLite, AsyncSQLite, PostgreSQL)
- Automatic fallback mechanisms
- Integration with existing memory management
- Production-ready configurations
- Thread-safe operations
- Comprehensive error handling and logging
"""

import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

# Import LangGraph checkpointing components with fallbacks
try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
    )
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    # Try to import PostgreSQL checkpointers (optional)
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        POSTGRES_AVAILABLE = True
    except ImportError:
        POSTGRES_AVAILABLE = False
        PostgresSaver = None
        AsyncPostgresSaver = None

    LANGGRAPH_CHECKPOINTING_AVAILABLE = True

except ImportError:
    LANGGRAPH_CHECKPOINTING_AVAILABLE = False

    # Fallback implementations
    class BaseCheckpointSaver:
        def __init__(self):
            pass

    class MemorySaver(BaseCheckpointSaver):
        def __init__(self):
            super().__init__()
            self.storage = {}

    SqliteSaver = None
    AsyncSqliteSaver = None
    PostgresSaver = None
    AsyncPostgresSaver = None
    POSTGRES_AVAILABLE = False

from .logger import StructuredLogger
from .memory import CodeHeroMemoryManager
from .utils import generate_id

logger = logging.getLogger(__name__)


class CheckpointerType:
    """Enumeration of available checkpointer types."""

    MEMORY = "memory"
    SQLITE = "sqlite"
    ASYNC_SQLITE = "async_sqlite"
    POSTGRES = "postgres"
    ASYNC_POSTGRES = "async_postgres"


class CheckpointerConfig:
    """Configuration for checkpointer initialization."""

    def __init__(
        self,
        checkpointer_type: str = CheckpointerType.MEMORY,
        connection_string: Optional[str] = None,
        database_path: Optional[str] = None,
        enable_async: bool = True,
        auto_setup: bool = True,
        thread_safe: bool = True,
        **kwargs,
    ):
        self.checkpointer_type = checkpointer_type
        self.connection_string = connection_string
        self.database_path = database_path
        self.enable_async = enable_async
        self.auto_setup = auto_setup
        self.thread_safe = thread_safe
        self.extra_config = kwargs


class EnhancedCheckpointerManager:
    """Enhanced checkpointer manager with multiple backend support and fallback mechanisms."""

    def __init__(
        self,
        config: Optional[CheckpointerConfig] = None,
        logger_service: Optional[StructuredLogger] = None,
        memory_manager: Optional[CodeHeroMemoryManager] = None,
    ):
        """Initialize the enhanced checkpointer manager.

        Args:
            config: Checkpointer configuration
            logger_service: Structured logger service
            memory_manager: Memory manager for integration
        """
        self.config = config or CheckpointerConfig()
        self.logger = logger_service
        self.memory_manager = memory_manager
        self.checkpointer = None
        self.fallback_checkpointer = None
        self.is_initialized = False

        # Track checkpointer health
        self.health_status = {
            "primary_healthy": False,
            "fallback_healthy": False,
            "last_health_check": None,
            "error_count": 0,
            "last_error": None,
        }

    async def initialize(self) -> bool:
        """Initialize the checkpointer with fallback support.

        Returns:
            True if initialization successful, False otherwise
        """
        if not LANGGRAPH_CHECKPOINTING_AVAILABLE:
            logger.warning("LangGraph checkpointing not available, using fallback")
            self.checkpointer = MemorySaver()
            self.is_initialized = True
            return True

        try:
            # Initialize primary checkpointer
            self.checkpointer = await self._create_checkpointer(self.config)

            if self.checkpointer:
                self.health_status["primary_healthy"] = True
                if self.logger:
                    await self.logger.log_info(
                        f"Primary checkpointer initialized: {self.config.checkpointer_type}"
                    )

            # Initialize fallback checkpointer (always use MemorySaver as fallback)
            if self.config.checkpointer_type != CheckpointerType.MEMORY:
                try:
                    fallback_config = CheckpointerConfig(
                        checkpointer_type=CheckpointerType.MEMORY
                    )
                    self.fallback_checkpointer = await self._create_checkpointer(
                        fallback_config
                    )
                    self.health_status["fallback_healthy"] = True
                    if self.logger:
                        await self.logger.log_info(
                            "Fallback checkpointer (MemorySaver) initialized"
                        )
                except Exception as e:
                    logger.warning(f"Failed to initialize fallback checkpointer: {e}")

            self.is_initialized = True
            self.health_status["last_health_check"] = datetime.now()

            return True

        except Exception as e:
            logger.error(f"Failed to initialize checkpointer: {e}")
            self.health_status["error_count"] += 1
            self.health_status["last_error"] = str(e)

            # Try to use fallback
            try:
                self.checkpointer = MemorySaver()
                self.is_initialized = True
                logger.info("Using MemorySaver as emergency fallback")
                return True
            except Exception as fallback_e:
                logger.error(f"Even fallback checkpointer failed: {fallback_e}")
                return False

    async def _create_checkpointer(
        self, config: CheckpointerConfig
    ) -> Optional[BaseCheckpointSaver]:
        """Create a checkpointer based on configuration.

        Args:
            config: Checkpointer configuration

        Returns:
            Initialized checkpointer or None if failed
        """
        try:
            if config.checkpointer_type == CheckpointerType.MEMORY:
                return MemorySaver()

            elif config.checkpointer_type == CheckpointerType.SQLITE:
                if not SqliteSaver:
                    raise ImportError("SqliteSaver not available")

                db_path = config.database_path or "checkpoints.sqlite"
                conn = sqlite3.connect(
                    db_path, check_same_thread=not config.thread_safe
                )
                checkpointer = SqliteSaver(conn)

                if config.auto_setup:
                    checkpointer.setup()

                return checkpointer

            elif config.checkpointer_type == CheckpointerType.ASYNC_SQLITE:
                if not AsyncSqliteSaver:
                    raise ImportError("AsyncSqliteSaver not available")

                db_path = config.database_path or "checkpoints.sqlite"

                # Use async context manager
                async def create_async_sqlite():
                    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
                        if config.auto_setup:
                            await saver.setup()
                        return saver

                return await create_async_sqlite()

            elif config.checkpointer_type == CheckpointerType.POSTGRES:
                if not POSTGRES_AVAILABLE or not PostgresSaver:
                    raise ImportError("PostgresSaver not available")

                if not config.connection_string:
                    raise ValueError("PostgreSQL connection string required")

                with PostgresSaver.from_conn_string(config.connection_string) as saver:
                    if config.auto_setup:
                        saver.setup()
                    return saver

            elif config.checkpointer_type == CheckpointerType.ASYNC_POSTGRES:
                if not POSTGRES_AVAILABLE or not AsyncPostgresSaver:
                    raise ImportError("AsyncPostgresSaver not available")

                if not config.connection_string:
                    raise ValueError("PostgreSQL connection string required")

                async with AsyncPostgresSaver.from_conn_string(
                    config.connection_string
                ) as saver:
                    if config.auto_setup:
                        await saver.setup()
                    return saver

            else:
                raise ValueError(
                    f"Unknown checkpointer type: {config.checkpointer_type}"
                )

        except Exception as e:
            logger.error(
                f"Failed to create {config.checkpointer_type} checkpointer: {e}"
            )
            return None

    def get_checkpointer(self) -> Optional[BaseCheckpointSaver]:
        """Get the active checkpointer with automatic fallback.

        Returns:
            Active checkpointer or fallback if primary fails
        """
        if not self.is_initialized:
            logger.warning("Checkpointer not initialized")
            return None

        # Try primary checkpointer first
        if self.checkpointer and self.health_status["primary_healthy"]:
            return self.checkpointer

        # Fall back to fallback checkpointer
        if self.fallback_checkpointer and self.health_status["fallback_healthy"]:
            logger.warning("Using fallback checkpointer due to primary failure")
            return self.fallback_checkpointer

        # Emergency fallback
        logger.error("Both primary and fallback checkpointers unavailable")
        return MemorySaver()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on checkpointers.

        Returns:
            Health status information
        """
        health_info = {
            "timestamp": datetime.now().isoformat(),
            "primary_checkpointer": {
                "type": self.config.checkpointer_type,
                "healthy": False,
                "error": None,
            },
            "fallback_checkpointer": {
                "healthy": False,
                "error": None,
            },
            "overall_status": "unhealthy",
        }

        # Test primary checkpointer
        if self.checkpointer:
            try:
                # Create a test configuration
                test_config = {
                    "configurable": {"thread_id": f"health_check_{generate_id()}"}
                }

                # Try to get a checkpoint (should return None for non-existent thread)
                if hasattr(self.checkpointer, "get"):
                    self.checkpointer.get(test_config)
                    health_info["primary_checkpointer"]["healthy"] = True
                    self.health_status["primary_healthy"] = True
                elif hasattr(self.checkpointer, "aget"):
                    await self.checkpointer.aget(test_config)
                    health_info["primary_checkpointer"]["healthy"] = True
                    self.health_status["primary_healthy"] = True
                else:
                    # Basic checkpointer without get method
                    health_info["primary_checkpointer"]["healthy"] = True
                    self.health_status["primary_healthy"] = True

            except Exception as e:
                health_info["primary_checkpointer"]["error"] = str(e)
                self.health_status["primary_healthy"] = False
                self.health_status["error_count"] += 1
                self.health_status["last_error"] = str(e)

        # Test fallback checkpointer
        if self.fallback_checkpointer:
            try:
                test_config = {
                    "configurable": {
                        "thread_id": f"health_check_fallback_{generate_id()}"
                    }
                }

                if hasattr(self.fallback_checkpointer, "get"):
                    self.fallback_checkpointer.get(test_config)
                    health_info["fallback_checkpointer"]["healthy"] = True
                    self.health_status["fallback_healthy"] = True

            except Exception as e:
                health_info["fallback_checkpointer"]["error"] = str(e)
                self.health_status["fallback_healthy"] = False

        # Determine overall status
        if health_info["primary_checkpointer"]["healthy"]:
            health_info["overall_status"] = "healthy"
        elif health_info["fallback_checkpointer"]["healthy"]:
            health_info["overall_status"] = "degraded"
        else:
            health_info["overall_status"] = "unhealthy"

        self.health_status["last_health_check"] = datetime.now()

        return health_info

    async def cleanup(self) -> None:
        """Clean up checkpointer resources."""
        try:
            if self.checkpointer and hasattr(self.checkpointer, "close"):
                await self.checkpointer.close()

            if self.fallback_checkpointer and hasattr(
                self.fallback_checkpointer, "close"
            ):
                await self.fallback_checkpointer.close()

            if self.logger:
                await self.logger.log_info("Checkpointer cleanup completed")

        except Exception as e:
            logger.error(f"Error during checkpointer cleanup: {e}")

    def create_thread_config(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        **kwargs,
    ) -> RunnableConfig:
        """Create a thread configuration for checkpointing.

        Args:
            thread_id: Thread identifier
            user_id: User identifier
            project_id: Project identifier
            **kwargs: Additional configuration

        Returns:
            RunnableConfig for LangGraph
        """
        config = {"configurable": {"thread_id": thread_id, **kwargs}}

        if user_id:
            config["configurable"]["user_id"] = user_id

        if project_id:
            config["configurable"]["project_id"] = project_id

        return config

    async def list_threads(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List available threads with optional filtering.

        Args:
            user_id: Filter by user ID
            project_id: Filter by project ID
            limit: Maximum number of threads to return

        Returns:
            List of thread information
        """
        checkpointer = self.get_checkpointer()
        if not checkpointer:
            return []

        try:
            threads = []

            # This is a simplified implementation
            # In a real scenario, you'd need to implement proper thread listing
            # based on the specific checkpointer type

            if hasattr(checkpointer, "list"):
                # Try to list checkpoints
                config = {"configurable": {}}
                if user_id:
                    config["configurable"]["user_id"] = user_id
                if project_id:
                    config["configurable"]["project_id"] = project_id

                if hasattr(checkpointer, "alist"):
                    async for checkpoint_tuple in checkpointer.alist(
                        config, limit=limit
                    ):
                        thread_info = {
                            "thread_id": checkpoint_tuple.config.get(
                                "configurable", {}
                            ).get("thread_id"),
                            "checkpoint_id": checkpoint_tuple.checkpoint.get("id"),
                            "timestamp": checkpoint_tuple.checkpoint.get("ts"),
                            "metadata": checkpoint_tuple.metadata,
                        }
                        threads.append(thread_info)
                else:
                    for checkpoint_tuple in checkpointer.list(config, limit=limit):
                        thread_info = {
                            "thread_id": checkpoint_tuple.config.get(
                                "configurable", {}
                            ).get("thread_id"),
                            "checkpoint_id": checkpoint_tuple.checkpoint.get("id"),
                            "timestamp": checkpoint_tuple.checkpoint.get("ts"),
                            "metadata": checkpoint_tuple.metadata,
                        }
                        threads.append(thread_info)

            return threads

        except Exception as e:
            logger.error(f"Failed to list threads: {e}")
            return []

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and all its checkpoints.

        Args:
            thread_id: Thread identifier to delete

        Returns:
            True if deletion successful, False otherwise
        """
        checkpointer = self.get_checkpointer()
        if not checkpointer:
            return False

        try:
            if hasattr(checkpointer, "adelete_thread"):
                await checkpointer.adelete_thread(thread_id)
            elif hasattr(checkpointer, "delete_thread"):
                checkpointer.delete_thread(thread_id)
            else:
                logger.warning(f"Checkpointer does not support thread deletion")
                return False

            if self.logger:
                await self.logger.log_info(f"Deleted thread: {thread_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete thread {thread_id}: {e}")
            return False


# Factory functions for easy checkpointer creation


def create_memory_checkpointer() -> CheckpointerConfig:
    """Create configuration for in-memory checkpointer."""
    return CheckpointerConfig(checkpointer_type=CheckpointerType.MEMORY)


def create_sqlite_checkpointer(
    database_path: str = "checkpoints.sqlite",
    async_mode: bool = False,
    thread_safe: bool = True,
) -> CheckpointerConfig:
    """Create configuration for SQLite checkpointer.

    Args:
        database_path: Path to SQLite database file
        async_mode: Whether to use async SQLite
        thread_safe: Whether to enable thread safety

    Returns:
        CheckpointerConfig for SQLite
    """
    checkpointer_type = (
        CheckpointerType.ASYNC_SQLITE if async_mode else CheckpointerType.SQLITE
    )

    return CheckpointerConfig(
        checkpointer_type=checkpointer_type,
        database_path=database_path,
        thread_safe=thread_safe,
        auto_setup=True,
    )


def create_postgres_checkpointer(
    connection_string: str, async_mode: bool = True
) -> CheckpointerConfig:
    """Create configuration for PostgreSQL checkpointer.

    Args:
        connection_string: PostgreSQL connection string
        async_mode: Whether to use async PostgreSQL

    Returns:
        CheckpointerConfig for PostgreSQL
    """
    if not POSTGRES_AVAILABLE:
        raise ImportError(
            "PostgreSQL checkpointer not available. Install langgraph-checkpoint-postgres"
        )

    checkpointer_type = (
        CheckpointerType.ASYNC_POSTGRES if async_mode else CheckpointerType.POSTGRES
    )

    return CheckpointerConfig(
        checkpointer_type=checkpointer_type,
        connection_string=connection_string,
        auto_setup=True,
    )


# Integration with existing memory system


async def integrate_with_memory_manager(
    checkpointer_manager: EnhancedCheckpointerManager,
    memory_manager: CodeHeroMemoryManager,
) -> bool:
    """Integrate checkpointer with existing memory manager.

    Args:
        checkpointer_manager: Enhanced checkpointer manager
        memory_manager: Existing memory manager

    Returns:
        True if integration successful
    """
    try:
        # Update memory manager's checkpointer
        checkpointer = checkpointer_manager.get_checkpointer()
        if checkpointer:
            memory_manager.checkpointer = checkpointer
            logger.info("Successfully integrated checkpointer with memory manager")
            return True
        else:
            logger.warning("No checkpointer available for integration")
            return False

    except Exception as e:
        logger.error(f"Failed to integrate checkpointer with memory manager: {e}")
        return False


# Utility functions for production deployment


def get_production_checkpointer_config() -> CheckpointerConfig:
    """Get production-ready checkpointer configuration based on environment.

    Returns:
        Production checkpointer configuration
    """
    # Check for PostgreSQL configuration
    postgres_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
    if postgres_url and POSTGRES_AVAILABLE:
        logger.info("Using PostgreSQL checkpointer for production")
        return create_postgres_checkpointer(postgres_url, async_mode=True)

    # Check for SQLite configuration
    sqlite_path = os.getenv("SQLITE_DB_PATH", "data/checkpoints.sqlite")

    # Ensure directory exists
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using SQLite checkpointer for production: {sqlite_path}")
    return create_sqlite_checkpointer(sqlite_path, async_mode=True, thread_safe=True)


def get_development_checkpointer_config() -> CheckpointerConfig:
    """Get development-friendly checkpointer configuration.

    Returns:
        Development checkpointer configuration
    """
    # Use in-memory for development unless explicitly configured
    if os.getenv("DEV_USE_SQLITE", "false").lower() == "true":
        return create_sqlite_checkpointer("dev_checkpoints.sqlite", async_mode=True)
    else:
        logger.info("Using in-memory checkpointer for development")
        return create_memory_checkpointer()


# Main initialization function


async def initialize_enhanced_checkpointing(
    environment: str = "development",
    custom_config: Optional[CheckpointerConfig] = None,
    logger_service: Optional[StructuredLogger] = None,
    memory_manager: Optional[CodeHeroMemoryManager] = None,
) -> EnhancedCheckpointerManager:
    """Initialize enhanced checkpointing system.

    Args:
        environment: Environment type ("development", "production", "testing")
        custom_config: Custom checkpointer configuration
        logger_service: Structured logger service
        memory_manager: Memory manager for integration

    Returns:
        Initialized checkpointer manager
    """
    try:
        # Determine configuration
        if custom_config:
            config = custom_config
        elif environment == "production":
            config = get_production_checkpointer_config()
        elif environment == "testing":
            config = create_memory_checkpointer()
        else:  # development
            config = get_development_checkpointer_config()

        # Create and initialize manager
        manager = EnhancedCheckpointerManager(
            config=config,
            logger_service=logger_service,
            memory_manager=memory_manager,
        )

        success = await manager.initialize()
        if not success:
            raise RuntimeError("Failed to initialize checkpointer manager")

        # Integrate with memory manager if provided
        if memory_manager:
            await integrate_with_memory_manager(manager, memory_manager)

        logger.info(f"Enhanced checkpointing initialized for {environment} environment")
        return manager

    except Exception as e:
        logger.error(f"Failed to initialize enhanced checkpointing: {e}")
        raise


# Export all public components
__all__ = [
    # Core classes
    "EnhancedCheckpointerManager",
    "CheckpointerConfig",
    "CheckpointerType",
    # Factory functions
    "create_memory_checkpointer",
    "create_sqlite_checkpointer",
    "create_postgres_checkpointer",
    # Configuration functions
    "get_production_checkpointer_config",
    "get_development_checkpointer_config",
    # Integration functions
    "integrate_with_memory_manager",
    "initialize_enhanced_checkpointing",
    # Availability flags
    "LANGGRAPH_CHECKPOINTING_AVAILABLE",
    "POSTGRES_AVAILABLE",
]
