"""Tools for the Strategic Framework.

This module provides reusable tools for agents to perform various tasks
like web content fetching, text processing, and code operations.
"""

from typing import Dict, Any, Optional, List, Union, Callable
import logging
from datetime import datetime
import aiohttp
import json
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import tool, BaseTool, Tool, StructuredTool
from langchain_openai import OpenAIEmbeddings

try:
    from langgraph.types import StreamWriter
    from langgraph.prebuilt import ToolNode
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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from astrapy import DataAPIClient
import os

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tool Schemas
# ─────────────────────────────────────────────────────────────────────────────

class WebFetchArgs(BaseModel):
    """Arguments for web content fetching."""
    url: str = Field(..., description="URL to fetch content from")

class SearchArgs(BaseModel):
    """Arguments for document search."""
    query: str = Field(..., description="Search query")
    collection: str = Field(default="strategy_book", description="Collection to search in")
    limit: int = Field(default=5, description="Maximum number of results to return")

class CodeGenArgs(BaseModel):
    """Arguments for code generation."""
    template: str = Field(..., description="Template name (e.g. 'fastapi', 'nextjs')")
    context: Dict[str, Any] = Field(..., description="Generation context with requirements")

class CodeValidationArgs(BaseModel):
    """Arguments for code validation."""
    code: str = Field(..., description="Code to validate")
    language: str = Field(..., description="Programming language")

class CodeAnalysisArgs(BaseModel):
    """Arguments for code analysis."""
    code: str = Field(..., description="Code to analyze")
    language: str = Field(..., description="Programming language")

class WebSearchArgs(BaseModel):
    """Arguments for web search."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results")
    include_domains: Optional[List[str]] = Field(default=None, description="Domains to include")
    exclude_domains: Optional[List[str]] = Field(default=None, description="Domains to exclude")

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
                    "content": content[:2000] + "..." if len(content) > 2000 else content,  # Truncate for LLM
                    "status": response.status,
                    "url": str(response.url),
                    "content_length": len(content)
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
                "ASTRA_DB_REGION"
            ]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-3-small"
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

    async def search(self, query: str, collection_name: str = "strategy_book", k: int = 5) -> List[Dict[str, Any]]:
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
                include_similarity=True
            )
            
            # Convert results to list and sort by similarity
            docs = sorted(
                list(results), 
                key=lambda x: x.get("$similarity", 0.0), 
                reverse=True
            )
            logger.info(f"Found {len(docs)} documents in collection '{collection_name}'")
            
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
                        "similarity": round(similarity_score, 4),  # Round to 4 decimal places
                        **doc.get("metadata", {})
                    },
                    "collection": collection_name
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
    query: str,
    collection: str = "strategy_book",
    limit: int = 5
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
                "content": result.get("content", "")[:1000] + "..." if len(result.get("content", "")) > 1000 else result.get("content", ""),
                "title": result.get("metadata", {}).get("title", ""),
                "source": result.get("metadata", {}).get("source", ""),
                "similarity": result.get("metadata", {}).get("similarity", 0.0),
                "collection": collection
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
    exclude_domains: Optional[List[str]] = None
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
            "max_results": max_results
        }

        if include_domains:
            search_data["include_domains"] = include_domains
        if exclude_domains:
            search_data["exclude_domains"] = exclude_domains

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.tavily.com/search",
                json=search_data,
                headers={"Content-Type": "application/json"}
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
                        "content": content[:800] + "..." if len(content) > 800 else content,  # Truncate for LLM
                        "score": item.get("score", 0.0),
                        "published_date": item.get("published_date", ""),
                        "source": "web_search"
                    }
                    formatted_results.append(formatted_result)

                # Include answer if available
                if result.get("answer"):
                    answer_content = result["answer"]
                    formatted_results.insert(0, {
                        "title": "AI Answer",
                        "url": "",
                        "content": answer_content[:800] + "..." if len(answer_content) > 800 else answer_content,
                        "score": 1.0,
                        "published_date": "",
                        "source": "ai_answer"
                    })

                logger.info(f"Web search completed: {len(formatted_results)} results for '{query}'")
                return formatted_results

    except Exception as e:
        error = f"Web search failed: {str(e)}"
        logger.error(error)
        return [{"error": error, "status": "failed"}]

@tool(args_schema=CodeGenArgs)
async def generate_code(
    template: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate code from template and context with best practices.
    
    Args:
        template: Template name (fastapi, nextjs, python, etc.)
        context: Generation context with requirements and specifications
        
    Returns:
        Generated code with metadata and best practices
    """
    try:
        requirements = context.get("requirements", "")
        features = context.get("features", [])
        
        if template.lower() == "fastapi":
            title = context.get("title", "API")
            description = context.get("description", "Generated API")
            code = f'''from fastapi import FastAPI, HTTPException, Depends
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
'''
        elif template.lower() == "nextjs":
            component_name = context.get("component_name", "Component")
            title = context.get("title", "Hello World")
            code = f'''"use client";

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
'''
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
            "language": "python" if template.lower() in ["python", "fastapi"] else "javascript" if template.lower() == "nextjs" else "text"
        }
            
        return result
        
    except Exception as e:
        error = f"Code generation failed: {str(e)}"
        logger.error(error)
        return {"error": error, "status": "failed", "template": template}

@tool(args_schema=CodeValidationArgs)
async def validate_code(
    code: str,
    language: str
) -> Dict[str, Any]:
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
                if len(code.split('\n')) > 100:
                    suggestions.append("Consider breaking large files into smaller modules")
                    
            except SyntaxError as e:
                errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            except Exception as e:
                errors.append(f"Compilation error: {str(e)}")
                
        elif language.lower() in ["javascript", "typescript", "js", "ts"]:
            # Basic JavaScript/TypeScript validation
            if "var " in code:
                suggestions.append("Consider using 'let' or 'const' instead of 'var'")
            if "==" in code and "===" not in code:
                suggestions.append("Use strict equality (===) instead of loose equality (==)")
            if "console.log" in code:
                warnings.append("Remove console.log statements before production")
                
        result = {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "language": language,
            "lines_of_code": len(code.split('\n')),
            "status": "success"
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
            "status": "failed"
        }

@tool(args_schema=CodeAnalysisArgs)
async def analyze_code(
    code: str,
    language: str
) -> Dict[str, Any]:
    """Analyze code for patterns, complexity, and quality metrics.
    
    Args:
        code: Code to analyze
        language: Programming language
        
    Returns:
        Detailed analysis results with metrics and recommendations
    """
    try:
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Basic metrics
        metrics = {
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "comment_lines": len([line for line in lines if line.strip().startswith('#') or line.strip().startswith('//')]),
            "blank_lines": len(lines) - len(non_empty_lines)
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
            suggestions.append("Consider breaking down into smaller functions or modules")
        if not patterns:
            suggestions.append("Consider using functions or classes for better code organization")
            
        result = {
            "complexity": complexity,
            "patterns": patterns,
            "suggestions": suggestions,
            "metrics": metrics,
            "language": language,
            "quality_score": min(100, max(0, 100 - len(suggestions) * 10)),
            "status": "success"
        }
            
        return result
        
    except Exception as e:
        error = f"Analysis failed: {str(e)}"
        logger.error(error)
        return {
            "error": error,
            "status": "failed",
            "language": language
        }

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
                func=fetch_web_content
            ),
            "search_documents": Tool(
                name="search_documents",
                description="Search through documents in AstraDB collections",
                func=search_documents
            ),
            "search_web": Tool(
                name="search_web",
                description="Search the web using Tavily API for current information",
                func=search_web
            ),
            "generate_code": Tool(
                name="generate_code",
                description="Generate code based on specifications",
                func=generate_code
            ),
            "validate_code": Tool(
                name="validate_code",
                description="Validate code for correctness",
                func=validate_code
            ),
            "analyze_code": Tool(
                name="analyze_code",
                description="Analyze code for patterns and issues",
                func=analyze_code
            ),
        }
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def register_tool(self, name: str, tool: BaseTool):
        """Register a new tool."""
        if not isinstance(tool, BaseTool):
            tool = Tool(
                name=name,
                description=f"Tool for {name}",
                func=tool
            )
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
            "content": ["generate_code", "search_documents"],  # Content creation and research
            "coding": ["generate_code", "validate_code", "analyze_code"],
            "validation": ["validate_code", "analyze_code"],
            "web": ["fetch_web_content", "search_web"],
            "analysis": ["analyze_code", "validate_code", "search_documents"],
            "all": list(self._tools.keys())  # All tools for supervisor
        }
        
        tool_names = category_mapping.get(category, [])
        return [self.get_tool(name) for name in tool_names if self.get_tool(name) is not None]
    
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
            
        if hasattr(llm, 'bind_tools'):
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
            if hasattr(tool, 'name') and hasattr(tool, 'description')
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
    "tool_registry",
    "ToolRegistry",
    "astra_retriever",
    # Tool schemas
    "WebFetchArgs",
    "SearchArgs",
    "WebSearchArgs",
    "CodeGenArgs",
    "CodeValidationArgs",
    "CodeAnalysisArgs"
]
