"""Utility functions for the backend system.

This module provides helper functions for tool execution, ID generation,
file handling, and other common operations.
"""

import asyncio
import functools
import json
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Union

from .tools import tool_registry

logger = logging.getLogger(__name__)


async def call_tool(tool_name: str, **kwargs: Any) -> Dict[str, Any]:
    """Call a tool with error handling and logging.

    Args:
        tool_name: Name of the tool to call
        **kwargs: Tool arguments

    Returns:
        Tool result with status and response
    """
    try:
        # Find tool by name
        tool = tool_registry.get_tool(tool_name)
        if not tool:
            logger.warning(f"Tool {tool_name} not found")
            return {
                "status": "error",
                "response": f"Tool {tool_name} not found",
                "error": f"Tool {tool_name} not found",
            }

        if asyncio.iscoroutinefunction(tool.run):
            result = await tool.run(**kwargs)
        else:
            result = await asyncio.to_thread(tool.run, **kwargs)

        return {"status": "success", "response": result, "error": None}

    except Exception as e:
        error_msg = f"Error executing tool {tool_name}: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "response": error_msg, "error": str(e)}


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix.

    Args:
        prefix: Optional prefix for the ID

    Returns:
        Unique ID string
    """
    return f"{prefix}_{uuid.uuid4()}" if prefix else str(uuid.uuid4())


def retry_with_backoff(
    retries: int = 3, backoff_in_seconds: int = 1, max_backoff_in_seconds: int = 10
) -> Callable:
    """Retry decorator with exponential backoff.

    Args:
        retries: Maximum number of retries
        backoff_in_seconds: Initial backoff time
        max_backoff_in_seconds: Maximum backoff time

    Returns:
        Decorator function
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Convert to async function if it's not already
            if not asyncio.iscoroutinefunction(func):
                func_to_execute = asyncio.coroutine(func)
            else:
                func_to_execute = func

            last_exception = None
            for attempt in range(retries):
                try:
                    return await func_to_execute(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    last_exception = e

                    if attempt == retries - 1:
                        logger.error(f"Max retries ({retries}) exceeded")
                        raise last_exception

                    # Calculate backoff with jitter
                    backoff = min(
                        max_backoff_in_seconds,
                        (backoff_in_seconds * 2**attempt)
                        + (random.randint(0, 1000) / 1000),
                    )
                    logger.info(f"Backing off for {backoff:.2f} seconds")
                    await asyncio.sleep(backoff)

            raise last_exception

        return wrapper

    return decorator


def safe_file_path(base_dir: Union[str, Path], filename: str) -> Path:
    """Create a safe file path within a base directory.

    Args:
        base_dir: Base directory path
        filename: Target filename

    Returns:
        Safe Path object
    """
    base_path = Path(base_dir)
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ")
    return base_path / safe_name


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file safely.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON file {file_path}: {e}")
        return {}


def save_json_file(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """Save data to a JSON file safely.

    Args:
        data: Data to save
        file_path: Target file path

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        return False


# Export all symbols
__all__ = [
    "call_tool",
    "generate_id",
    "retry_with_backoff",
    "safe_file_path",
    "load_json_file",
    "save_json_file",
]
