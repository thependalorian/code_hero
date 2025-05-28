"""Tools for the Strategic Framework.

This module provides reusable tools for agents to perform various tasks
like web content fetching, text processing, and code operations.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from langchain_core.tools import BaseTool, Tool, tool
from langchain_openai import OpenAIEmbeddings

try:
    from langgraph.prebuilt import ToolNode
    from langgraph.types import StreamWriter

    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for older LangGraph versions
    LANGGRAPH_AVAILABLE = False

    class StreamWriter:
        async def write(self, data):
            pass

    class ToolNode:
        def __init__(self, tools):
            self.tools = tools


import os

from astrapy import DataAPIClient
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import consolidated state models
from .state import (
    CodeAnalysisArgs,
    CodeGenArgs,
    CodeValidationArgs,
    DocumentManagementArgs,
    FileOperationArgs,
    REPLArgs,
    SearchArgs,
    WebFetchArgs,
    WebSearchArgs,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────


@tool(args_schema=WebFetchArgs)
async def fetch_web_content(url: str) -> Dict[str, Any]:
    """Fetch webpage content asynchronously.

    Args:
        url: URL to fetch content from

    Returns:
        Webpage content and metadata
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                content = await response.text()

                result = {
                    "content": (
                        content[:2000] + "..." if len(content) > 2000 else content
                    ),  # Truncate for LLM
                    "status": response.status,
                    "url": str(response.url),
                    "content_length": len(content),
                }

                return result

    except Exception as e:
        error = f"Failed to fetch content from {url}: {str(e)}"
        logger.error(error)
        return {"error": error, "status": "failed"}


class AstraDBRetriever:
    """Document retriever for AstraDB."""

    def __init__(self):
        """Initialize the retriever."""
        self.embeddings = None
        self.client = None
        self.db = None
        self.text_splitter = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the retriever."""
        if self._initialized:
            return

        try:
            # Validate required environment variables
            required_vars = [
                "OPENAI_API_KEY",
                "ASTRA_DB_APPLICATION_TOKEN",
                "ASTRA_DB_ID",
                "ASTRA_DB_REGION",
            ]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(
                    f"Missing required environment variables: {', '.join(missing_vars)}"
                )

            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
            )
            logger.info("OpenAI embeddings initialized successfully")

            # Initialize AstraDB client
            self.client = DataAPIClient(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))
            api_endpoint = f"https://{os.getenv('ASTRA_DB_ID')}-{os.getenv('ASTRA_DB_REGION')}.apps.astra.datastax.com"
            self.db = self.client.get_database_by_api_endpoint(api_endpoint)
            logger.info(f"AstraDB client initialized with endpoint: {api_endpoint}")

            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            logger.info("Text splitter initialized successfully")

            self._initialized = True
            logger.info("AstraDB retriever initialization completed")

        except Exception as e:
            error_msg = f"Failed to initialize AstraDB retriever: {str(e)}"
            logger.error(error_msg)
            raise

    async def search(
        self, query: str, collection_name: str = "strategy_book", k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for documents in a collection using vector similarity.

        Args:
            query: Search query
            collection_name: Name of collection
            k: Number of results to return

        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info(f"Searching collection '{collection_name}' with query: {query}")

            # Get collection
            collection = self.db.get_collection(collection_name)

            # Get embedding for query
            query_embedding = await self.embeddings.aembed_query(query)

            # Search using vector similarity
            results = collection.find(
                sort={"$vector": query_embedding},
                limit=k,
                projection={"*": 1},
                include_similarity=True,
            )

            # Convert results to list and sort by similarity
            docs = sorted(
                list(results), key=lambda x: x.get("$similarity", 0.0), reverse=True
            )
            logger.info(
                f"Found {len(docs)} documents in collection '{collection_name}'"
            )

            # Format results
            formatted_results = []
            for doc in docs:
                similarity_score = doc.get("$similarity", 0.0)
                result = {
                    "content": doc.get("content", ""),
                    "metadata": {
                        "title": doc.get("title", ""),
                        "url": doc.get("url", ""),
                        "source": doc.get("source", ""),
                        "similarity": round(
                            similarity_score, 4
                        ),  # Round to 4 decimal places
                        **doc.get("metadata", {}),
                    },
                    "collection": collection_name,
                }
                formatted_results.append(result)
                logger.debug(f"Added result with similarity score: {similarity_score}")

            return formatted_results

        except Exception as e:
            error_msg = f"Error searching collection {collection_name}: {str(e)}"
            logger.error(error_msg)
            return []

    async def list_collections(self) -> List[str]:
        """List all available collections."""
        if not self._initialized:
            await self.initialize()

        try:
            logger.info("Listing available collections")
            collections = self.db.list_collection_names()
            logger.info(f"Found collections: {collections}")
            return collections

        except Exception as e:
            error_msg = f"Error listing collections: {str(e)}"
            logger.error(error_msg)
            return []


# Initialize retriever
astra_retriever = AstraDBRetriever()


@tool(args_schema=SearchArgs)
async def search_documents(
    query: str, collection: str = "strategy_book", limit: int = 5
) -> List[Dict[str, Any]]:
    """Search documents in vector store using semantic similarity.

    Args:
        query: Search query
        collection: Collection to search in
        limit: Maximum number of results to return

    Returns:
        List of relevant documents with similarity scores
    """
    try:
        results = await astra_retriever.search(query, collection, k=limit)

        # Format results for LLM consumption
        formatted_results = []
        for result in results:
            formatted_result = {
                "content": (
                    result.get("content", "")[:1000] + "..."
                    if len(result.get("content", "")) > 1000
                    else result.get("content", "")
                ),
                "title": result.get("metadata", {}).get("title", ""),
                "source": result.get("metadata", {}).get("source", ""),
                "similarity": result.get("metadata", {}).get("similarity", 0.0),
                "collection": collection,
            }
            formatted_results.append(formatted_result)

        return formatted_results

    except Exception as e:
        error = f"Search failed: {str(e)}"
        logger.error(error)
        return [{"error": error, "status": "failed"}]


@tool(args_schema=WebSearchArgs)
async def search_web(
    query: str,
    max_results: int = 5,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search the web using Tavily API for current information.

    Args:
        query: Search query
        max_results: Maximum number of results
        include_domains: Domains to include in search
        exclude_domains: Domains to exclude from search

    Returns:
        List of web search results with content and metadata
    """
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            error = "TAVILY_API_KEY not found in environment variables"
            logger.error(error)
            return [{"error": error, "status": "failed"}]

        # Prepare search request
        search_data = {
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_images": False,
            "include_raw_content": False,
            "max_results": max_results,
        }

        if include_domains:
            search_data["include_domains"] = include_domains
        if exclude_domains:
            search_data["exclude_domains"] = exclude_domains

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.tavily.com/search",
                json=search_data,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                result = await response.json()

                # Format results for LLM consumption
                formatted_results = []
                for item in result.get("results", []):
                    content = item.get("content", "")
                    formatted_result = {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": (
                            content[:800] + "..." if len(content) > 800 else content
                        ),  # Truncate for LLM
                        "score": item.get("score", 0.0),
                        "published_date": item.get("published_date", ""),
                        "source": "web_search",
                    }
                    formatted_results.append(formatted_result)

                # Include answer if available
                if result.get("answer"):
                    answer_content = result["answer"]
                    formatted_results.insert(
                        0,
                        {
                            "title": "AI Answer",
                            "url": "",
                            "content": (
                                answer_content[:800] + "..."
                                if len(answer_content) > 800
                                else answer_content
                            ),
                            "score": 1.0,
                            "published_date": "",
                            "source": "ai_answer",
                        },
                    )

                logger.info(
                    f"Web search completed: {len(formatted_results)} results for '{query}'"
                )
                return formatted_results

    except Exception as e:
        error = f"Web search failed: {str(e)}"
        logger.error(error)
        return [{"error": error, "status": "failed"}]


@tool(args_schema=CodeGenArgs)
async def generate_code(template: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate code from template and context with best practices.

    Args:
        template: Template name (fastapi, nextjs, python, etc.)
        context: Generation context with requirements

    Returns:
        Generated code with metadata and best practices
    """
    try:
        requirements = context.get("requirements", "")
        features = context.get("features", [])

        if template.lower() == "fastapi":
            title = context.get("title", "API")
            description = context.get("description", "Generated API")
            code = f"""from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging

# Initialize FastAPI app
app = FastAPI(
    title="{title}",
    description="{description}",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Item(BaseModel):
    name: str
    description: Optional[str] = None

@app.get("/")
async def root():
    return {{"message": "Hello World", "requirements": "{requirements}"}}

@app.get("/health")
async def health_check():
    return {{"status": "healthy"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        elif template.lower() == "nextjs":
            component_name = context.get("component_name", "Component")
            title = context.get("title", "Hello World")
            code = f""""use client";

import {{ useState, useEffect }} from "react";

interface Props {{
  title?: string;
  description?: string;
}}

export default function {component_name}({{ title = "{title}", description }}: Props) {{
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {{
        // Initialize component
        console.log("Component mounted");
    }}, []);

    return (
        <div className="container mx-auto p-4">
            <h1 className="text-3xl font-bold mb-4">{{title}}</h1>
            {{description && (
                <p className="text-gray-600 mb-6">{{description}}</p>
            )}}
            
            <div className="grid gap-4">
                <div className="p-4 border rounded-lg">
                    <p>Requirements: {requirements}</p>
                </div>
            </div>
        </div>
    );
}}
"""
        elif template.lower() == "python":
            class_name = context.get("class_name", "GeneratedClass")
            module_description = context.get("description", "Generated Python module")
            title = context.get("title", "generated functionality")
            code = f'''#!/usr/bin/env python3
"""
{module_description}

Requirements: {requirements}
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class {class_name}:
    """Main class for {title}."""
    
    def __init__(self):
        """Initialize the class."""
        self.created_at = datetime.utcnow()
        logger.info("Initialized {class_name}")
    
    def process(self, data: Any) -> Dict[str, Any]:
        """Process data according to requirements.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed result
        """
        try:
            result = {{
                "status": "success",
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }}
            return result
        except Exception as e:
            logger.error(f"Processing failed: {{e}}")
            return {{"status": "error", "error": str(e)}}

def main():
    """Main function."""
    instance = {class_name}()
    result = instance.process("test data")
    print(f"Result: {{result}}")

if __name__ == "__main__":
    main()
'''
        else:
            # Generic template
            features_str = ", ".join(features) if features else "None specified"
            code = f'''# Generated code for {template}
# Requirements: {requirements}
# Features: {features_str}

def main():
    """Main function for {template} template."""
    print("Generated code template: {template}")
    return {{"status": "success", "template": "{template}"}}

if __name__ == "__main__":
    main()
'''

        result = {
            "code": code.strip(),
            "template": template,
            "context": context,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "language": (
                "python"
                if template.lower() in ["python", "fastapi"]
                else "javascript" if template.lower() == "nextjs" else "text"
            ),
        }

        return result

    except Exception as e:
        error = f"Code generation failed: {str(e)}"
        logger.error(error)
        return {"error": error, "status": "failed", "template": template}


@tool(args_schema=CodeValidationArgs)
async def validate_code(code: str, language: str) -> Dict[str, Any]:
    """Validate code syntax and structure with detailed analysis.

    Args:
        code: Code to validate
        language: Programming language (python, javascript, typescript, etc.)

    Returns:
        Comprehensive validation results with errors and suggestions
    """
    try:
        errors = []
        warnings = []
        suggestions = []

        if language.lower() == "python":
            try:
                compile(code, "<string>", "exec")
                # Additional Python-specific checks
                if "import *" in code:
                    warnings.append("Avoid wildcard imports for better code clarity")
                if "eval(" in code or "exec(" in code:
                    warnings.append("Avoid eval() and exec() for security reasons")
                if len(code.split("\n")) > 100:
                    suggestions.append(
                        "Consider breaking large files into smaller modules"
                    )

            except SyntaxError as e:
                errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            except Exception as e:
                errors.append(f"Compilation error: {str(e)}")

        elif language.lower() in ["javascript", "typescript", "js", "ts"]:
            # Basic JavaScript/TypeScript validation
            if "var " in code:
                suggestions.append("Consider using 'let' or 'const' instead of 'var'")
            if "==" in code and "===" not in code:
                suggestions.append(
                    "Use strict equality (===) instead of loose equality (==)"
                )
            if "console.log" in code:
                warnings.append("Remove console.log statements before production")

        result = {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "language": language,
            "lines_of_code": len(code.split("\n")),
            "status": "success",
        }

        return result

    except Exception as e:
        error = f"Validation failed: {str(e)}"
        return {
            "is_valid": False,
            "errors": [error],
            "warnings": [],
            "suggestions": [],
            "language": language,
            "status": "failed",
        }


@tool(args_schema=CodeAnalysisArgs)
async def analyze_code(code: str, language: str) -> Dict[str, Any]:
    """Analyze code for patterns, complexity, and quality metrics.

    Args:
        code: Code to analyze
        language: Programming language

    Returns:
        Detailed analysis results with metrics and recommendations
    """
    try:
        lines = code.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        # Basic metrics
        metrics = {
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "comment_lines": len(
                [
                    line
                    for line in lines
                    if line.strip().startswith("#") or line.strip().startswith("//")
                ]
            ),
            "blank_lines": len(lines) - len(non_empty_lines),
        }

        # Complexity analysis
        complexity = "low"
        if len(non_empty_lines) > 50:
            complexity = "medium"
        if len(non_empty_lines) > 200:
            complexity = "high"

        # Pattern detection
        patterns = []
        if language.lower() == "python":
            if "class " in code:
                patterns.append("object_oriented")
            if "def " in code:
                patterns.append("functions")
            if "async def" in code:
                patterns.append("async_programming")
            if "import " in code:
                patterns.append("modular_design")
        elif language.lower() in ["javascript", "typescript"]:
            if "class " in code:
                patterns.append("es6_classes")
            if "function" in code or "=>" in code:
                patterns.append("functions")
            if "async" in code or "await" in code:
                patterns.append("async_programming")
            if "import " in code or "require(" in code:
                patterns.append("modular_design")

        # Quality suggestions
        suggestions = []
        if metrics["comment_lines"] / max(metrics["code_lines"], 1) < 0.1:
            suggestions.append("Add more comments to improve code documentation")
        if complexity == "high":
            suggestions.append(
                "Consider breaking down into smaller functions or modules"
            )
        if not patterns:
            suggestions.append(
                "Consider using functions or classes for better code organization"
            )

        result = {
            "complexity": complexity,
            "patterns": patterns,
            "suggestions": suggestions,
            "metrics": metrics,
            "language": language,
            "quality_score": min(100, max(0, 100 - len(suggestions) * 10)),
            "status": "success",
        }

        return result

    except Exception as e:
        error = f"Analysis failed: {str(e)}"
        logger.error(error)
        return {"error": error, "status": "failed", "language": language}


# ─────────────────────────────────────────────────────────────────────────────
# REPL and File Management Tools
# ─────────────────────────────────────────────────────────────────────────────


@tool(args_schema=REPLArgs)
async def execute_python_repl(
    code: str, timeout: int = 30, capture_output: bool = True
) -> Dict[str, Any]:
    """Execute Python code in a safe REPL environment with security checks.

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds
        capture_output: Whether to capture stdout/stderr

    Returns:
        Execution results with output, errors, and status
    """
    import os
    import subprocess
    import sys
    import tempfile

    try:
        # Security checks - block dangerous operations
        dangerous_patterns = [
            "import os",
            "import subprocess",
            "import sys",
            "import shutil",
            "eval(",
            "exec(",
            "__import__",
            "open(",
            "file(",
            "input(",
            "raw_input(",
            "compile(",
            "globals(",
            "locals(",
            "setattr(",
            "getattr(",
            "hasattr(",
            "delattr(",
            "exit(",
            "quit(",
            "reload(",
            "help(",
            "dir(",
            "vars(",
            "type(",
            "isinstance(",
            "rm ",
            "del ",
            "rmdir",
            "unlink",
            "remove",
        ]

        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return {
                    "status": "blocked",
                    "error": f"Security violation: '{pattern}' is not allowed",
                    "output": "",
                    "execution_time": 0,
                }

        # Additional security: limit code length
        if len(code) > 5000:
            return {
                "status": "blocked",
                "error": "Code too long (max 5000 characters)",
                "output": "",
                "execution_time": 0,
            }

        # Create safe execution environment
        safe_code = f"""
import sys
import io
import contextlib
import time
from typing import Any, Dict, List, Optional, Union
import json
import math
import random
import datetime
import re
import collections
import itertools
import functools

# Capture output
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

start_time = time.time()
result = None
error = None

try:
    with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
        # User code execution
{chr(10).join('        ' + line for line in code.split(chr(10)))}
        
except Exception as e:
    error = str(e)
    import traceback
    stderr_capture.write(traceback.format_exc())

end_time = time.time()
execution_time = end_time - start_time

# Prepare results
stdout_content = stdout_capture.getvalue()
stderr_content = stderr_capture.getvalue()

print("EXECUTION_RESULTS_START")
print(json.dumps({{
    "stdout": stdout_content,
    "stderr": stderr_content,
    "error": error,
    "execution_time": execution_time,
    "status": "error" if error else "success"
}}))
print("EXECUTION_RESULTS_END")
"""

        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(safe_code)
            temp_file = f.name

        try:
            # Execute with timeout
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tempfile.gettempdir(),
            )

            stdout, stderr = process.communicate(timeout=timeout)

            # Parse results
            if (
                "EXECUTION_RESULTS_START" in stdout
                and "EXECUTION_RESULTS_END" in stdout
            ):
                start_idx = stdout.find("EXECUTION_RESULTS_START") + len(
                    "EXECUTION_RESULTS_START\n"
                )
                end_idx = stdout.find("EXECUTION_RESULTS_END")
                results_json = stdout[start_idx:end_idx].strip()

                try:
                    results = json.loads(results_json)
                    return {
                        "status": results.get("status", "success"),
                        "output": results.get("stdout", ""),
                        "error": results.get("error"),
                        "stderr": results.get("stderr", ""),
                        "execution_time": results.get("execution_time", 0),
                        "code_executed": code,
                    }
                except json.JSONDecodeError:
                    pass

            # Fallback if JSON parsing fails
            return {
                "status": "completed",
                "output": stdout,
                "error": stderr if stderr else None,
                "stderr": stderr,
                "execution_time": 0,
                "code_executed": code,
            }

        except subprocess.TimeoutExpired:
            process.kill()
            return {
                "status": "timeout",
                "error": f"Code execution timed out after {timeout} seconds",
                "output": "",
                "stderr": "",
                "execution_time": timeout,
                "code_executed": code,
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass

    except Exception as e:
        error = f"REPL execution failed: {str(e)}"
        logger.error(error)
        return {
            "status": "failed",
            "error": error,
            "output": "",
            "stderr": "",
            "execution_time": 0,
            "code_executed": code,
        }


@tool(args_schema=FileOperationArgs)
async def read_file_content(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """Read content from a file with proper error handling.

    Args:
        file_path: Path to the file to read
        encoding: File encoding

    Returns:
        File content and metadata
    """
    try:
        from pathlib import Path

        path = Path(file_path)

        # Security check - prevent reading outside allowed directories
        allowed_dirs = [
            Path.cwd(),
            Path.home() / "Documents",
            Path("/tmp"),
            Path("/var/tmp"),
        ]

        # Check if path is within allowed directories
        is_allowed = False
        try:
            resolved_path = path.resolve()
            for allowed_dir in allowed_dirs:
                try:
                    resolved_path.relative_to(allowed_dir.resolve())
                    is_allowed = True
                    break
                except ValueError:
                    continue
        except:
            pass

        if not is_allowed:
            return {
                "status": "blocked",
                "error": "File access denied - path not in allowed directories",
                "content": "",
                "file_info": {},
            }

        if not path.exists():
            return {
                "status": "not_found",
                "error": f"File not found: {file_path}",
                "content": "",
                "file_info": {},
            }

        if not path.is_file():
            return {
                "status": "invalid",
                "error": f"Path is not a file: {file_path}",
                "content": "",
                "file_info": {},
            }

        # Read file content
        with open(path, "r", encoding=encoding) as f:
            content = f.read()

        # Get file info
        stat = path.stat()
        file_info = {
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "extension": path.suffix,
            "name": path.name,
            "parent": str(path.parent),
        }

        return {
            "status": "success",
            "content": content,
            "file_info": file_info,
            "encoding": encoding,
            "file_path": str(path),
        }

    except UnicodeDecodeError as e:
        return {
            "status": "encoding_error",
            "error": f"Failed to decode file with {encoding}: {str(e)}",
            "content": "",
            "file_info": {},
        }
    except Exception as e:
        error = f"Failed to read file: {str(e)}"
        logger.error(error)
        return {"status": "failed", "error": error, "content": "", "file_info": {}}


@tool(args_schema=FileOperationArgs)
async def write_file_content(
    file_path: str, content: str, encoding: str = "utf-8"
) -> Dict[str, Any]:
    """Write content to a file with proper error handling and safety checks.

    Args:
        file_path: Path to the file to write
        content: Content to write
        encoding: File encoding

    Returns:
        Write operation results
    """
    try:
        from pathlib import Path

        path = Path(file_path)

        # Security check - prevent writing outside allowed directories
        allowed_dirs = [
            Path.cwd(),
            Path.home() / "Documents",
            Path("/tmp"),
            Path("/var/tmp"),
        ]

        # Check if path is within allowed directories
        is_allowed = False
        try:
            resolved_path = path.resolve()
            for allowed_dir in allowed_dirs:
                try:
                    resolved_path.relative_to(allowed_dir.resolve())
                    is_allowed = True
                    break
                except ValueError:
                    continue
        except:
            pass

        if not is_allowed:
            return {
                "status": "blocked",
                "error": "File write denied - path not in allowed directories",
                "bytes_written": 0,
            }

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        with open(path, "w", encoding=encoding) as f:
            f.write(content)

        # Get file info after writing
        stat = path.stat()

        return {
            "status": "success",
            "file_path": str(path),
            "bytes_written": stat.st_size,
            "encoding": encoding,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    except Exception as e:
        error = f"Failed to write file: {str(e)}"
        logger.error(error)
        return {"status": "failed", "error": error, "bytes_written": 0}


@tool(args_schema=DocumentManagementArgs)
async def create_document(
    title: str,
    content: str,
    format: str = "markdown",
    tags: List[str] = None,
    metadata: Dict[str, Any] = None,
    document_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or update a document in the document management system.

    Args:
        title: Document title
        content: Document content
        format: Document format (markdown, html, text)
        tags: Document tags
        metadata: Additional metadata
        document_id: Optional document ID for updates

    Returns:
        Document creation/update results
    """
    try:
        import uuid
        from datetime import datetime

        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}

        # Generate document ID if not provided
        if not document_id:
            document_id = str(uuid.uuid4())

        # Create document object
        document = {
            "id": document_id,
            "title": title,
            "content": content,
            "format": format,
            "tags": tags,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "version": 1,
            "word_count": len(content.split()),
            "character_count": len(content),
        }

        # Store in AstraDB if available
        try:
            # This would integrate with the actual document storage system
            # For now, we'll simulate storage
            logger.info(f"Document created: {document_id} - {title}")

            return {
                "status": "success",
                "document": document,
                "message": f"Document '{title}' created successfully",
            }

        except Exception as e:
            logger.warning(f"Failed to store in database: {e}")
            # Return success anyway since document is created in memory
            return {
                "status": "success",
                "document": document,
                "message": f"Document '{title}' created (local only)",
                "warning": "Database storage failed",
            }

    except Exception as e:
        error = f"Failed to create document: {str(e)}"
        logger.error(error)
        return {"status": "failed", "error": error, "document": None}


@tool
async def list_files_in_directory(
    directory_path: str = ".", file_extension: Optional[str] = None, max_files: int = 50
) -> Dict[str, Any]:
    """List files in a directory with optional filtering.

    Args:
        directory_path: Directory to list files from
        file_extension: Optional file extension filter (e.g., '.py', '.md')
        max_files: Maximum number of files to return

    Returns:
        List of files with metadata
    """
    try:
        from pathlib import Path

        path = Path(directory_path)

        # Security check
        allowed_dirs = [
            Path.cwd(),
            Path.home() / "Documents",
            Path("/tmp"),
            Path("/var/tmp"),
        ]

        is_allowed = False
        try:
            resolved_path = path.resolve()
            for allowed_dir in allowed_dirs:
                try:
                    resolved_path.relative_to(allowed_dir.resolve())
                    is_allowed = True
                    break
                except ValueError:
                    continue
        except:
            pass

        if not is_allowed:
            return {
                "status": "blocked",
                "error": "Directory access denied",
                "files": [],
            }

        if not path.exists():
            return {
                "status": "not_found",
                "error": f"Directory not found: {directory_path}",
                "files": [],
            }

        if not path.is_dir():
            return {
                "status": "invalid",
                "error": f"Path is not a directory: {directory_path}",
                "files": [],
            }

        # List files
        files = []
        for item in path.iterdir():
            if item.is_file():
                if file_extension and not item.name.endswith(file_extension):
                    continue

                stat = item.stat()
                files.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "size": stat.st_size,
                        "extension": item.suffix,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    }
                )

                if len(files) >= max_files:
                    break

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "status": "success",
            "directory": str(path),
            "files": files,
            "total_files": len(files),
            "filtered_by": file_extension if file_extension else "none",
        }

    except Exception as e:
        error = f"Failed to list directory: {str(e)}"
        logger.error(error)
        return {"status": "failed", "error": error, "files": []}


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._initialize_tools()

    def _initialize_tools(self):
        """Initialize default tools."""
        self._tools = {
            "fetch_web_content": Tool(
                name="fetch_web_content",
                description="Fetch content from a web URL",
                func=fetch_web_content,
            ),
            "search_documents": Tool(
                name="search_documents",
                description="Search through documents in AstraDB collections",
                func=search_documents,
            ),
            "search_web": Tool(
                name="search_web",
                description="Search the web using Tavily API for current information",
                func=search_web,
            ),
            "generate_code": Tool(
                name="generate_code",
                description="Generate code based on specifications",
                func=generate_code,
            ),
            "validate_code": Tool(
                name="validate_code",
                description="Validate code for correctness",
                func=validate_code,
            ),
            "analyze_code": Tool(
                name="analyze_code",
                description="Analyze code for patterns and issues",
                func=analyze_code,
            ),
            "execute_python_repl": Tool(
                name="execute_python_repl",
                description="Execute Python code in a safe REPL environment",
                func=execute_python_repl,
            ),
            "read_file_content": Tool(
                name="read_file_content",
                description="Read content from files with security checks",
                func=read_file_content,
            ),
            "write_file_content": Tool(
                name="write_file_content",
                description="Write content to files with security checks",
                func=write_file_content,
            ),
            "create_document": Tool(
                name="create_document",
                description="Create or update documents in the document management system",
                func=create_document,
            ),
            "list_files_in_directory": Tool(
                name="list_files_in_directory",
                description="List files in a directory with optional filtering",
                func=list_files_in_directory,
            ),
        }

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def register_tool(self, name: str, tool: BaseTool):
        """Register a new tool."""
        if not isinstance(tool, BaseTool):
            tool = Tool(name=name, description=f"Tool for {name}", func=tool)
        self._tools[name] = tool

    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())

    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools.

        Returns:
            List of all tools
        """
        return list(self._tools.values())

    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get tools by category for specialized agent binding.

        Args:
            category: Tool category (e.g., 'research', 'content', 'coding')

        Returns:
            List of tools in the category
        """
        category_mapping = {
            "research": ["search_documents", "search_web", "fetch_web_content"],
            "content": [
                "generate_code",
                "search_documents",
            ],  # Content creation and research
            "coding": [
                "generate_code",
                "validate_code",
                "analyze_code",
                "execute_python_repl",
            ],
            "validation": ["validate_code", "analyze_code"],
            "web": ["fetch_web_content", "search_web"],
            "analysis": ["analyze_code", "validate_code", "search_documents"],
            "documentation": [
                "create_document",
                "read_file_content",
                "write_file_content",
                "list_files_in_directory",
                "search_documents",
            ],
            "file_management": [
                "read_file_content",
                "write_file_content",
                "list_files_in_directory",
            ],
            "repl": ["execute_python_repl", "validate_code", "analyze_code"],
            "all": list(self._tools.keys()),  # All tools for supervisor
        }

        tool_names = category_mapping.get(category, [])
        return [
            self.get_tool(name)
            for name in tool_names
            if self.get_tool(name) is not None
        ]

    def create_tool_node(self, category: Optional[str] = None) -> ToolNode:
        """Create a LangGraph ToolNode for specific category or all tools.

        Args:
            category: Optional category to filter tools

        Returns:
            Configured ToolNode for LangGraph workflows
        """
        if category:
            tools = self.get_tools_by_category(category)
        else:
            tools = self.get_all_tools()

        return ToolNode(tools)

    def bind_tools_to_llm(self, llm, category: Optional[str] = None):
        """Bind tools to an LLM for LangGraph agent creation.

        Args:
            llm: Language model to bind tools to
            category: Optional category to filter tools

        Returns:
            LLM with bound tools
        """
        if category:
            tools = self.get_tools_by_category(category)
        else:
            tools = self.get_all_tools()

        if hasattr(llm, "bind_tools"):
            return llm.bind_tools(tools)
        else:
            # Fallback for older LangChain versions
            return llm

    def get_tool_descriptions(self, category: Optional[str] = None) -> Dict[str, str]:
        """Get tool descriptions for prompt engineering.

        Args:
            category: Optional category to filter tools

        Returns:
            Dictionary mapping tool names to descriptions
        """
        if category:
            tools = self.get_tools_by_category(category)
        else:
            tools = self.get_all_tools()

        return {
            tool.name: tool.description
            for tool in tools
            if hasattr(tool, "name") and hasattr(tool, "description")
        }


# Create global tool registry
tool_registry = ToolRegistry()

# Export all symbols
__all__ = [
    "fetch_web_content",
    "search_documents",
    "search_web",
    "generate_code",
    "validate_code",
    "analyze_code",
    "execute_python_repl",
    "read_file_content",
    "write_file_content",
    "create_document",
    "list_files_in_directory",
    "tool_registry",
    "ToolRegistry",
    "astra_retriever",
    # Tool schemas
    "WebFetchArgs",
    "SearchArgs",
    "WebSearchArgs",
    "CodeGenArgs",
    "CodeValidationArgs",
    "CodeAnalysisArgs",
    "REPLArgs",
    "FileOperationArgs",
    "DocumentManagementArgs",
]
