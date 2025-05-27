# Code Hero CLI Setup Guide

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file in the project root with the following variables:

```bash
# Required Variables
OPENAI_API_KEY=your_openai_api_key_here
ASTRA_DB_ID=your_astra_db_id_here
ASTRA_DB_REGION=your_astra_db_region_here
ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token_here

# Optional LLM Providers
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional Services
TAVILY_API_KEY=your_tavily_api_key_here

# LangChain Tracing (Optional)
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_TRACING_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=code-hero

# Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### 2. Install Dependencies

```bash
# Install the package in development mode
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

### 3. Test the CLI

```bash
# Run the test script
python test_cli.py

# Or test commands manually
python -m code_hero --help
```

## 📋 Available Commands

### Basic Commands

```bash
# Show help
python -m code_hero --help

# List all available agents
python -m code_hero list-agents

# List available LLM models
python -m code_hero list-models

# Check configuration and environment
python -m code_hero check-config
```

### Agent Interaction

```bash
# Query a specific agent
python -m code_hero query-agent <agent_name> "<your_question>"

# Start interactive chat with an agent
python -m code_hero chat-mode [--agent <agent_name>]

# Search documents in collections
python -m code_hero search-docs "<search_query>" [--collection <collection_name>]
```

### Multi-Agent Workflows

```bash
# Coordinate multi-agent tasks
curl -X POST "http://localhost:8000/multi-agent/coordinate" \
  -H "Content-Type: application/json" \
  -d '{"task_description": "research and write about LangGraph best practices", "project_id": "test_001"}'
```

### Server Management

```bash
# Run the FastAPI server
python -m code_hero run-server [--host 0.0.0.0] [--port 8000]
```

## 🤖 Available Agents

| Agent | Capabilities | Collections | Tools |
|-------|-------------|-------------|-------|
| `supervisor` | Orchestrates workflows, manages tasks | All collections | All tools |
| `strategic_expert` | Strategic planning, decision-making | strategy_book, framework docs | Research, analysis |
| `langchain_expert` | LangChain development, chains, agents | langchain_docs, langgraph_docs | Coding, validation |
| `fastapi_expert` | FastAPI development, REST APIs | fastapi_docs, pydantic_docs | Code generation, validation |
| `nextjs_expert` | Next.js development, React components | nextjs_docs | Code generation, web tools |
| `prompt_engineer` | Enhanced prompt engineering | framework_docs, strategy_book | Research, content |
| `research` | Information gathering, analysis | strategy_book, all docs | Search, web research |
| `implementation` | Code generation, implementation | All technical docs | Coding tools |
| `code_reviewer` | Code review, quality assurance | All technical docs | Validation, analysis |
| `documentation` | Documentation generation, writing | All collections | Content, research |

## 🛠️ Enhanced Tools System

### Tool Categories

- **Research Tools**: `search_documents`, `search_web`, `fetch_web_content`
- **Coding Tools**: `generate_code`, `validate_code`, `analyze_code`
- **Content Tools**: `generate_code`, `search_documents`
- **Validation Tools**: `validate_code`, `analyze_code`
- **Web Tools**: `fetch_web_content`, `search_web`

### Tool Features

- **LangGraph Integration**: Proper tool binding with `invoke` method
- **Content Optimization**: Automatic truncation for LLM efficiency
- **Enhanced Validation**: Comprehensive code analysis with quality metrics
- **Smart Code Generation**: Context-aware templates for FastAPI, Next.js, Python
- **Error Handling**: Structured error responses for workflow management

## 💡 Usage Examples

### Query Specific Agents

```bash
# Ask LangChain expert about chains
python -m code_hero query-agent langchain_expert "How do I create a sequential chain?"

# Ask FastAPI expert about routing
python -m code_hero query-agent fastapi_expert "How do I create nested routers with authentication?"

# Ask Next.js expert about components
python -m code_hero query-agent nextjs_expert "How do I create a reusable component with TypeScript and DaisyUI?"

# Use verbose mode for detailed output
python -m code_hero query-agent research "What are the best practices for AI agents?" --verbose
```

### Interactive Chat Mode

```bash
# Start chat with supervisor agent
python -m code_hero chat-mode

# Start chat with specific agent
python -m code_hero chat-mode --agent langchain_expert

# Chat commands:
# - help: Show available commands
# - clear: Clear conversation history
# - history: Show conversation history
# - switch <agent>: Switch to different agent
# - exit/quit/bye: End session
```

### Document Search

```bash
# Search in default collection (strategy_book)
python -m code_hero search-docs "AI agent patterns"

# Search in specific collection
python -m code_hero search-docs "FastAPI middleware" --collection fastapi_docs

# Limit number of results
python -m code_hero search-docs "LangChain tools" --collection langchain_docs --limit 3
```

### Multi-Agent Coordination

```bash
# Research and write task
curl -X POST "http://localhost:8000/multi-agent/coordinate" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "research FastAPI best practices and generate a sample application with authentication",
    "project_id": "fastapi_research_001"
  }'

# Code generation task
curl -X POST "http://localhost:8000/multi-agent/coordinate" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "generate a Next.js component with TypeScript and validate the code",
    "project_id": "nextjs_component_001"
  }'
```

## 🔧 Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   ```bash
   # Check your configuration
   python -m code_hero check-config
   ```

2. **Import Errors**
   ```bash
   # Make sure you're in the project directory
   cd /path/to/code-hero
   
   # Install in development mode
   pip install -e .
   ```

3. **AstraDB Connection Issues**
   - Verify your `ASTRA_DB_ID`, `ASTRA_DB_REGION`, and `ASTRA_DB_APPLICATION_TOKEN`
   - Check that your AstraDB instance is running
   - Ensure the collections exist in your database

4. **OpenAI API Issues**
   - Verify your `OPENAI_API_KEY` is valid
   - Check your OpenAI account has sufficient credits

5. **LangGraph Integration Issues**
   - Ensure LangGraph is properly installed: `pip install langgraph`
   - Check for version compatibility issues
   - Verify tool binding is working correctly

### Debug Mode

```bash
# Run with verbose output
python -m code_hero query-agent <agent> "<query>" --verbose

# Check logs
tail -f code_hero.log

# Test tool functionality
python -c "from src.code_hero.tools import tool_registry; print(tool_registry.list_tools())"
```

### Server Issues

```bash
# Check server health
curl http://localhost:8000/health

# Check AstraDB health
curl http://localhost:8000/api/astra/health

# Test multi-agent coordination
curl -X POST "http://localhost:8000/multi-agent/coordinate" \
  -H "Content-Type: application/json" \
  -d '{"task_description": "test task", "project_id": "test_001"}'
```

## 🎯 Next Steps

1. **Set up your environment variables** in `.env`
2. **Test basic commands** with `python test_cli.py`
3. **Try querying agents** with simple questions
4. **Explore interactive chat mode** for longer conversations
5. **Search your document collections** to verify data access
6. **Test multi-agent workflows** for complex tasks
7. **Experiment with tool binding** and enhanced code generation

## 📚 Collections Available

Your AstraDB should have these collections:
- `strategy_book` - Strategic framework documentation
- `langchain_docs` - LangChain documentation
- `langgraph_docs` - LangGraph documentation
- `llamaindex_docs` - LlamaIndex documentation
- `nextjs_docs` - Next.js documentation
- `fastapi_docs` - FastAPI documentation
- `pydantic_docs` - Pydantic documentation
- `crewai_docs` - CrewAI documentation
- `langsmith_docs` - LangSmith documentation
- `agno_phidata_docs` - Agno/PhiData documentation
- `framework_docs` - General framework documentation

## 🚀 Advanced Features

### Tool Binding
- Agents automatically get appropriate tools based on their role
- Research agents get search and web tools
- Coding agents get generation, validation, and analysis tools
- Supervisor gets access to all tools for coordination

### Workflow Management
- Multi-agent coordination with proper state management
- Error handling and recovery mechanisms
- Human-in-the-loop integration for complex decisions
- Streaming responses for real-time updates

### Code Generation
- Context-aware templates for different frameworks
- Comprehensive validation with quality metrics
- Best practices integration for FastAPI, Next.js, Python
- Automatic code analysis and suggestions 