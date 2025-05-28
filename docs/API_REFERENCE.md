# Code Hero API Reference

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, no authentication is required for local development. API keys are configured via environment variables.

## Endpoints

### Health Check
Check the system health and status.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "logger": {
      "initialized": true
    },
    "state_manager": {
      "initialized": true,
      "project_count": 1,
      "chat_count": 3,
      "agent_count": 0,
      "graph_count": 0
    },
    "supervisor": {
      "initialized": true,
      "status": null,
      "active_workflows": 0,
      "state_manager_connected": true,
      "logger_connected": true
    }
  },
  "config": {
    "api": true,
    "database": true,
    "llm_registry": true,
    "logging": true
  },
  "environment": "development",
  "timestamp": "2025-05-27T08:36:25.480720Z"
}
```

### Chat Endpoint
Main chat interface with LangGraph supervisor routing.

**Endpoint:** `POST /api/chat/`

**Parameters:**
- `message` (required): The user's message/query
- `conversation_id` (optional): Existing conversation ID to continue

**Example Requests:**

#### Code Generation
```bash
curl -X POST "http://localhost:8000/api/chat/?message=create%20a%20python%20function%20to%20calculate%20fibonacci%20numbers"
```

#### Research Query
```bash
curl -X POST "http://localhost:8000/api/chat/?message=find%20information%20about%20machine%20learning"
```

#### Code Analysis
```bash
curl -X POST "http://localhost:8000/api/chat/?message=analyze%20this%20code%20for%20performance"
```

#### General Query
```bash
curl -X POST "http://localhost:8000/api/chat/?message=hello%20how%20are%20you"
```

**Response Format:**
```json
{
  "response": "I'll help you with code generation. Here's the python code:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n\n**Explanation:** This code addresses your request...\n\n---\n*Handled by: code_expert* | *Tools: generate_code*",
  "conversation_id": "conv_1748349115.976745",
  "messages": [
    {
      "role": "user",
      "content": "create a python function to calculate fibonacci numbers",
      "timestamp": "2025-05-27T08:31:56.445563",
      "metadata": {
        "source": "api"
      }
    },
    {
      "role": "assistant",
      "content": "I'll help you with code generation...",
      "timestamp": "2025-05-27T08:31:56.445588",
      "metadata": {
        "source": "langgraph",
        "task_analysis": {
          "task_type": "code_generation",
          "complexity": "medium",
          "required_tools": ["generate_code", "validate_code"],
          "estimated_steps": 2
        },
        "tools_available": ["fetch_web_content", "search_documents", "search_web", "generate_code", "validate_code", "analyze_code"]
      }
    }
  ],
  "status": "completed",
  "active_agent": "code_expert"
}
```

**Response Fields:**
- `response`: The formatted response from the agent
- `conversation_id`: Unique identifier for the conversation
- `messages`: Array of all messages in the conversation
- `status`: Current status ("completed", "failed", "pending")
- `active_agent`: The agent that handled the request

### Get Chat History
Retrieve the history of a specific conversation.

**Endpoint:** `GET /api/chat/{conversation_id}`

**Parameters:**
- `conversation_id` (path): The conversation ID

**Response:**
```json
{
  "conversation_id": "conv_1748349115.976745",
  "messages": [
    {
      "role": "user",
      "content": "create a python function to calculate fibonacci numbers",
      "timestamp": "2025-05-27T08:31:56.445563",
      "metadata": {
        "source": "api"
      }
    },
    {
      "role": "assistant", 
      "content": "I'll help you with code generation...",
      "timestamp": "2025-05-27T08:31:56.445588",
      "metadata": {
        "source": "langgraph",
        "task_analysis": {
          "task_type": "code_generation",
          "complexity": "medium",
          "required_tools": ["generate_code", "validate_code"],
          "estimated_steps": 2
        }
      }
    }
  ],
  "status": "completed",
  "active_agent": "code_expert"
}
```

### Multi-Agent Coordination
Coordinate complex tasks across multiple agents.

**Endpoint:** `POST /multi-agent/coordinate`

**Request Body:**
```json
{
  "task_description": "Create a FastAPI application with user authentication",
  "project_id": "optional_project_id"
}
```

**Response:**
```json
{
  "success": true,
  "project_id": "project_1748349115.976745",
  "task_description": "Create a FastAPI application with user authentication",
  "result": {
    "status": "completed",
    "artifacts": {
      "generated_code": "...",
      "documentation": "...",
      "tests": "..."
    },
    "workflow_id": "multi_agent_project_1748349115.976745",
    "execution_time": "2025-05-27T08:31:56.445588Z"
  },
  "timestamp": "2025-05-27T08:31:56.445588Z"
}
```

### Root Information
Get basic API information.

**Endpoint:** `GET /`

**Response:**
```json
{
  "name": "Code Hero API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/api/docs",
  "redoc": "/api/redoc", 
  "openapi": "/api/openapi.json",
  "endpoints": {
    "multi_agent": "/multi-agent/coordinate",
    "health": "/health"
  }
}
```

## Agent Routing

The system uses intelligent routing based on message content:

### Code Expert
**Triggers:** "create", "build", "implement", "develop", "generate", "write", "make", "code", "function", "class", "api"

**Capabilities:**
- Code generation in multiple languages
- API development
- Function and class creation
- Implementation guidance

**Example Response Structure:**
```json
{
  "task_analysis": {
    "task_type": "code_generation",
    "complexity": "medium",
    "required_tools": ["generate_code", "validate_code"],
    "estimated_steps": 2
  },
  "agent_used": "code_expert",
  "tools_used": ["generate_code"],
  "confidence": 0.9
}
```

### Research Expert
**Triggers:** "search", "find", "research", "look up", "information", "about", "learn about", "tell me about"

**Capabilities:**
- Document search
- Web research
- Information gathering
- Knowledge retrieval

**Example Response Structure:**
```json
{
  "task_analysis": {
    "task_type": "research",
    "complexity": "medium", 
    "required_tools": ["search_documents", "search_web"],
    "estimated_steps": 2
  },
  "agent_used": "research_expert",
  "tools_used": ["search_documents", "search_web"],
  "confidence": 0.8
}
```

### Analysis Expert
**Triggers:** "analyze", "review", "check", "examine", "validate", "audit", "inspect", "evaluate"

**Capabilities:**
- Code analysis
- Performance reviews
- Validation tasks
- Quality assessment

**Example Response Structure:**
```json
{
  "task_analysis": {
    "task_type": "analysis",
    "complexity": "medium",
    "required_tools": ["analyze_code", "validate_code"],
    "estimated_steps": 2
  },
  "agent_used": "analysis_expert", 
  "tools_used": ["analyze_code", "validate_code"],
  "confidence": 0.85
}
```

### Supervisor (General)
**Triggers:** All other queries

**Capabilities:**
- General conversation
- Help and guidance
- Task clarification
- System information

## Error Handling

### Error Response Format
```json
{
  "detail": {
    "error": "Error message",
    "message": "Human-readable description"
  }
}
```

### Common Error Codes
- `400`: Bad Request - Invalid parameters
- `404`: Not Found - Resource not found
- `422`: Unprocessable Entity - Validation error
- `500`: Internal Server Error - System error

### Example Error Responses

#### Missing Required Parameter
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["query", "message"],
      "msg": "Field required",
      "input": null
    }
  ]
}
```

#### Chat Not Found
```json
{
  "detail": "Chat conv_123456 not found"
}
```

#### System Error
```json
{
  "detail": {
    "error": "Failed to process chat request",
    "message": "Internal system error occurred"
  }
}
```

## Rate Limiting

Currently, no rate limiting is implemented for local development. In production, consider implementing:
- Request rate limits per IP
- Conversation limits per user
- Resource usage monitoring

## WebSocket Support

WebSocket support is not currently implemented but could be added for:
- Real-time chat streaming
- Live agent status updates
- Progress notifications for long-running tasks

## SDK Examples

### Python
```python
import requests

# Chat request
response = requests.post(
    "http://localhost:8000/api/chat/",
    params={"message": "create a python function for sorting"}
)
result = response.json()
print(result["response"])

# Health check
health = requests.get("http://localhost:8000/health")
print(health.json()["status"])
```

### JavaScript
```javascript
// Chat request
const response = await fetch(
  'http://localhost:8000/api/chat/?message=create%20a%20javascript%20function',
  { method: 'POST' }
);
const result = await response.json();
console.log(result.response);

// Health check
const health = await fetch('http://localhost:8000/health');
const status = await health.json();
console.log(status.status);
```

### cURL
```bash
# Chat request
curl -X POST "http://localhost:8000/api/chat/?message=help%20me%20with%20coding"

# Health check
curl http://localhost:8000/health

# Get chat history
curl http://localhost:8000/api/chat/conv_1748349115.976745
```

## Development Notes

### Environment Variables
Required environment variables for full functionality:
```env
OPENAI_API_KEY=your_openai_key
ASTRA_DB_ID=your_astra_db_id
ASTRA_DB_REGION=your_astra_region
ASTRA_DB_APPLICATION_TOKEN=your_astra_token
TAVILY_API_KEY=your_tavily_key
GROQ_API_KEY=your_groq_key
DEEPSEEK_API_KEY=your_deepseek_key
```

### Testing
```bash
# Start server
python -m src.code_hero run-server --port 8000

# Test different agent types
curl -X POST "http://localhost:8000/api/chat/?message=create%20a%20function"
curl -X POST "http://localhost:8000/api/chat/?message=search%20for%20information"
curl -X POST "http://localhost:8000/api/chat/?message=analyze%20performance"
```

### Monitoring
- Check `/health` endpoint for system status
- Monitor logs for agent execution details
- Use LangChain tracing for workflow visibility 