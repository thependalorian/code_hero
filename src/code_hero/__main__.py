"""Main entry point for the Code Hero CLI.

This module provides the command-line interface for the Code Hero AI agent system,
allowing users to interact with the system through various commands and modes.

Usage:
    python -m code_hero [command] [options]

Examples:
    python -m code_hero chat                    # Start interactive chat
    python -m code_hero server                  # Start API server
    python -m code_hero agent fastapi_expert   # Chat with specific agent
    python -m code_hero --help                 # Show help
"""

import logging
import sys
from typing import Optional

try:
    from .main import cli
except ImportError as e:
    print(f"Error importing Code Hero CLI: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)


def main() -> Optional[int]:
    """Main entry point with error handling."""
    try:
        return cli()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logging.error(f"Unexpected error in Code Hero CLI: {e}")
        print(f"An unexpected error occurred: {e}")
        print("Please check the logs for more details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)
