# Code Hero Quick Start Guide

Get up and running with Code Hero's LangGraph-powered multi-agent system in minutes.

## Prerequisites

- Python 3.11+
- OpenAI API key (required)
- Additional API keys for enhanced features (optional)

## Installation

### 1. Clone and Setup
```bash
git clone <repository-url>
cd code-hero
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory:

```env
# Required - Get from OpenAI
OPENAI_API_KEY=sk-your-openai-api-key

# Optional but recommended for enhanced features
ASTRA_DB_ID=your-astra-db-id
ASTRA_DB_REGION=your-astra-db-region  
ASTRA_DB_APPLICATION_TOKEN=your-astra-db-token
TAVILY_API_KEY=your-tavily-api-key
GROQ_API_KEY=your-groq-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key

# Optional - LangChain tracing for debugging
LANGCHAIN_API_KEY=your-langchain-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=code-hero
```

### 3. Start the Server
```bash
python -m src.code_hero run-server --port 8000
```

You should see:
```
INFO     Startup completed successfully
INFO     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## First Steps

### 1. Health Check
Verify everything is working:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "logger": {"initialized": true},
    "state_manager": {"initialized": true},
    "supervisor": {"initialized": true}
  }
}
```

### 2. Try the Chat Interface

#### Code Generation
```bash
curl -X POST "http://localhost:8000/api/chat/?message=create%20a%20python%20function%20to%20calculate%20fibonacci%20numbers"
```

#### Research Query
```bash
curl -X POST "http://localhost:8000/api/chat/?message=find%20information%20about%20FastAPI"
```

#### Code Analysis
```bash
curl -X POST "http://localhost:8000/api/chat/?message=analyze%20this%20code%20for%20performance"
```

#### General Help
```bash
curl -X POST "http://localhost:8000/api/chat/?message=hello%20how%20can%20you%20help%20me"
```

## Understanding the Response

Each response includes:
- **Agent routing**: Which specialist handled your request
- **Tools used**: What tools were employed
- **Structured output**: Consistent, typed responses
- **Conversation tracking**: Persistent chat history

Example response:
```json
{
  "response": "I'll help you with code generation. Here's the python code:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n\n---\n*Handled by: code_expert* | *Tools: generate_code*",
  "conversation_id": "conv_1748349115.976745",
  "status": "completed",
  "active_agent": "code_expert"
}
```

## Agent Types

Code Hero automatically routes your requests to the right specialist:

### 🔧 Code Expert
**Triggers:** "create", "build", "implement", "develop", "generate", "code", "function"
- Generates code in multiple languages
- Creates APIs and applications
- Provides implementation guidance

### 🔍 Research Expert  
**Triggers:** "search", "find", "research", "information", "about"
- Searches documents and web
- Gathers information
- Provides research summaries

### 📊 Analysis Expert
**Triggers:** "analyze", "review", "check", "examine", "validate"
- Analyzes code quality
- Reviews performance
- Validates implementations

### 🎯 Supervisor (General)
**Handles:** Everything else
- General conversation
- Help and guidance
- Task clarification

## Common Use Cases

### 1. Code Generation
```bash
# Python function
curl -X POST "http://localhost:8000/api/chat/?message=create%20a%20python%20function%20for%20sorting%20a%20list"

# FastAPI endpoint
curl -X POST "http://localhost:8000/api/chat/?message=build%20a%20FastAPI%20endpoint%20for%20user%20registration"

# JavaScript function
curl -X POST "http://localhost:8000/api/chat/?message=implement%20a%20JavaScript%20function%20for%20form%20validation"
```

### 2. Research and Information
```bash
# Technology research
curl -X POST "http://localhost:8000/api/chat/?message=find%20information%20about%20GraphQL%20vs%20REST"

# Best practices
curl -X POST "http://localhost:8000/api/chat/?message=search%20for%20Python%20async%20programming%20best%20practices"

# Documentation lookup
curl -X POST "http://localhost:8000/api/chat/?message=look%20up%20FastAPI%20dependency%20injection"
```

### 3. Code Analysis
```bash
# Performance analysis
curl -X POST "http://localhost:8000/api/chat/?message=analyze%20this%20algorithm%20for%20time%20complexity"

# Code review
curl -X POST "http://localhost:8000/api/chat/?message=review%20this%20function%20for%20best%20practices"

# Validation
curl -X POST "http://localhost:8000/api/chat/?message=validate%20this%20API%20design"
```

## CLI Commands

### Interactive Chat Mode
```bash
python -m src.code_hero chat-mode --agent supervisor
```

### Query Specific Agent
```bash
python -m src.code_hero query-agent langchain_expert "How do I use LangChain with FastAPI?"
```

### Document Search
```bash
python -m src.code_hero search-docs "FastAPI best practices" --collection strategy_book
```

### Web Search
```bash
python -m src.code_hero search-web "Python async programming" --max-results 5
```

## Troubleshooting

### Server Won't Start
1. Check Python version: `python --version` (need 3.11+)
2. Verify dependencies: `pip install -r requirements.txt`
3. Check environment variables: ensure `OPENAI_API_KEY` is set

### API Errors
1. Check health endpoint: `curl http://localhost:8000/health`
2. Verify API keys are valid
3. Check logs for detailed error messages

### No Response from Agents
1. Ensure LangGraph is installed: `pip install langgraph`
2. Check if OpenAI API key is working
3. Try a simple query first: "hello"

### Common Error Messages

#### "LangGraph not available"
```bash
pip install langgraph
```

#### "OpenAI API key not found"
```bash
export OPENAI_API_KEY=your-key-here
# or add to .env file
```

#### "Connection refused"
```bash
# Check if server is running
curl http://localhost:8000/health
```

## Next Steps

### 1. Explore the API
- Check out the full [API Reference](API_REFERENCE.md)
- Try different types of queries
- Experiment with conversation continuity

### 2. Learn the Architecture
- Read the [LangGraph Implementation Guide](LANGGRAPH_IMPLEMENTATION.md)
- Understand agent routing logic
- Explore Pydantic structured outputs

### 3. Customize and Extend
- Add new agents for specific domains
- Integrate additional tools
- Customize response formats

### 4. Production Deployment
- Set up proper environment variables
- Configure logging and monitoring
- Implement rate limiting and authentication

## Getting Help

### Documentation
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [LangGraph Implementation](LANGGRAPH_IMPLEMENTATION.md) - Technical details
- [Architecture Overview](../README.md) - System overview

### Health Monitoring
```bash
# Check system status
curl http://localhost:8000/health

# Monitor logs
tail -f logs/code_hero.log
```

### Debug Mode
Enable detailed tracing:
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-key
```

## Example Session

Here's a complete example session:

```bash
# 1. Start server
python -m src.code_hero run-server --port 8000

# 2. Health check
curl http://localhost:8000/health

# 3. Code generation
curl -X POST "http://localhost:8000/api/chat/?message=create%20a%20REST%20API%20for%20todo%20management"

# 4. Research
curl -X POST "http://localhost:8000/api/chat/?message=find%20best%20practices%20for%20API%20design"

# 5. Analysis
curl -X POST "http://localhost:8000/api/chat/?message=analyze%20the%20security%20of%20this%20API"

# 6. General help
curl -X POST "http://localhost:8000/api/chat/?message=what%20else%20can%20you%20help%20me%20with"
```

You're now ready to use Code Hero's powerful multi-agent system! 🚀 