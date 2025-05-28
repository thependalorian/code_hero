# Code Hero API Documentation

## 🚀 Overview

The Code Hero API provides a comprehensive set of endpoints for interacting with the multi-agent AI system. The API is built with FastAPI and includes real-time chat, workflow management, document processing, and system monitoring capabilities.

**Base URL**: `http://localhost:8000`  
**API Documentation**: `http://localhost:8000/api/docs`  
**Alternative Docs**: `http://localhost:8000/api/redoc`

## 🆕 Recent Updates

### ✅ **Enhanced Document Management** (Just Added)
- **File upload with TRD conversion** - Technical Requirements Document generation
- **Multi-file processing** - Batch upload and analysis capabilities
- **Document lifecycle management** - Upload, analyze, convert, download, delete
- **Project-based organization** - Associate documents with specific projects

### ✅ **Devstral Model Integration** (Just Added)
- **Mistral Devstral 24B** available for specialized Python coding tasks
- **Enhanced coding capabilities** - 46.8% score on SWE-Bench Verified
- **Multi-file editing support** - Advanced codebase exploration

### ✅ **Build System Improvements** (Just Added)
- **Production-ready frontend** - All TypeScript errors resolved
- **Code formatting applied** - Black + isort for consistent Python code style
- **Enhanced API client** - Full document management support

## 🔐 Authentication

Currently, the API operates without authentication for development purposes. In production, implement:
- API Key authentication
- JWT tokens
- OAuth 2.0 integration

## 📋 API Endpoints

### System Endpoints

#### Health Check
```http
GET /health
```

**Description**: Check system health and service status

**Response**:
```json
{
  "status": "healthy",
  "services": {
    "logger": true,
    "state_manager": true,
    "supervisor": true
  },
  "config": {
    "api": true,
    "database": true,
    "llm_registry": true,
    "logging": true
  },
  "environment": "development",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Root Information
```http
GET /
```

**Description**: Get API information and available endpoints

**Response**:
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

### Multi-Agent Coordination

#### Coordinate Multi-Agent Task
```http
POST /multi-agent/coordinate
```

**Description**: Coordinate a complex task across multiple AI agents

**Request Body**:
```json
{
  "task_description": "Build a FastAPI application with authentication",
  "project_id": "project_123"
}
```

**Response**:
```json
{
  "success": true,
  "project_id": "project_123",
  "task_description": "Build a FastAPI application with authentication",
  "result": {
    "agents_involved": ["strategic_expert", "fastapi_expert"],
    "artifacts": {
      "code": "...",
      "documentation": "...",
      "tests": "..."
    },
    "status": "completed"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 💬 Chat API (`/api/chat`)

### Send Message
```http
POST /api/chat/
```

**Description**: Send a message to an AI agent and get a response

**Request Body**:
```json
{
  "message": "How do I implement authentication in FastAPI?",
  "conversation_id": "conv_123"
}
```

**Response**:
```json
{
  "response": "To implement authentication in FastAPI, you can use...",
  "conversation_id": "conv_123",
  "messages": [
    {
      "role": "user",
      "content": "How do I implement authentication in FastAPI?",
      "timestamp": "2024-01-15T10:30:00Z",
      "metadata": {"source": "api"}
    },
    {
      "role": "assistant",
      "content": "To implement authentication in FastAPI, you can use...",
      "timestamp": "2024-01-15T10:30:01Z",
      "metadata": {"agent": "fastapi_expert", "status": "completed"}
    }
  ],
  "status": "completed",
  "active_agent": "fastapi_expert"
}
```

### Get Chat History
```http
GET /api/chat/{conversation_id}
```

**Description**: Retrieve the complete chat history for a conversation

**Parameters**:
- `conversation_id` (path): The conversation identifier

**Response**:
```json
{
  "conversation_id": "conv_123",
  "messages": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2024-01-15T10:30:00Z",
      "metadata": {"source": "api"}
    }
  ],
  "status": "active",
  "active_agent": "supervisor",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

## 📄 Documents API (`/documents`)

### Upload Documents
```http
POST /documents/upload
```

**Description**: Upload one or more documents for processing and analysis

**Request**: Multipart form data
- `files` (file[]): One or more files to upload
- `project_id` (string, optional): Associate documents with a project
- `description` (string, optional): Description of the documents

**Response**:
```json
{
  "success": true,
  "uploaded_files": [
    {
      "id": "doc_123",
      "filename": "requirements.pdf",
      "size": 1024000,
      "status": "uploaded"
    }
  ],
  "failed_files": [],
  "total_uploaded": 1,
  "total_failed": 0,
  "project_id": "project_123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### List Documents
```http
GET /documents
```

**Description**: Retrieve a list of uploaded documents

**Query Parameters**:
- `project_id` (string, optional): Filter by project
- `limit` (integer, optional): Number of documents to return (default: 50)
- `offset` (integer, optional): Number of documents to skip (default: 0)

**Response**:
```json
{
  "documents": [
    {
      "id": "doc_123",
      "filename": "requirements.pdf",
      "size": 1024000,
      "content_type": "application/pdf",
      "format": "pdf",
      "processing_status": "completed",
      "upload_timestamp": "2024-01-15T10:30:00Z",
      "project_id": "project_123",
      "description": "Project requirements document"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0,
  "project_id": "project_123"
}
```

### Get Document Details
```http
GET /documents/{document_id}
```

**Description**: Get detailed information about a specific document

**Parameters**:
- `document_id` (path): The document identifier

**Response**:
```json
{
  "id": "doc_123",
  "filename": "requirements.pdf",
  "size": 1024000,
  "content_type": "application/pdf",
  "format": "pdf",
  "processing_status": "completed",
  "upload_timestamp": "2024-01-15T10:30:00Z",
  "project_id": "project_123",
  "description": "Project requirements document"
}
```

### Download Document
```http
GET /documents/{document_id}/download
```

**Description**: Download the original document file

**Parameters**:
- `document_id` (path): The document identifier

**Response**: Binary file content with appropriate headers

### Analyze Document
```http
POST /documents/{document_id}/analyze
```

**Description**: Perform AI-powered analysis on a document

**Parameters**:
- `document_id` (path): The document identifier

**Request Body**:
```json
{
  "analysis_type": "general"
}
```

**Response**:
```json
{
  "success": true,
  "document_id": "doc_123",
  "task_id": "task_456",
  "analysis_type": "general",
  "result": "This document contains project requirements including...",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Convert to TRD (Technical Requirements Document)
```http
POST /documents/{document_id}/convert-trd
```

**Description**: Convert a document to a Technical Requirements Document format

**Parameters**:
- `document_id` (path): The document identifier

**Request Body**:
```json
{
  "target_format": "technical_requirements",
  "stakeholders": "Development Team, Product Manager, QA Team",
  "compliance_requirements": "GDPR, HIPAA, SOX"
}
```

**Response**:
```json
{
  "success": true,
  "document_id": "doc_123",
  "trd_id": "trd_789",
  "task_id": "task_456",
  "target_format": "technical_requirements",
  "result": "# Technical Requirements Document\n\n## Overview\n...",
  "stakeholders": "Development Team, Product Manager, QA Team",
  "compliance_requirements": "GDPR, HIPAA, SOX",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Delete Document
```http
DELETE /documents/{document_id}
```

**Description**: Delete a document and its associated data

**Parameters**:
- `document_id` (path): The document identifier

**Response**:
```json
{
  "success": true,
  "message": "Document deleted successfully"
}
```

## 🔄 Workflow API (`/api/graph`)

### Create Workflow
```http
POST /api/graph/workflow
```

**Description**: Create a new workflow for task automation

**Request Body**:
```json
{
  "name": "Code Review Workflow",
  "description": "Automated code review process",
  "nodes": ["analyze", "review", "suggest"]
}
```

**Response**:
```json
{
  "workflow_id": "workflow_123",
  "status": "created",
  "config": {
    "name": "Code Review Workflow",
    "description": "Automated code review process",
    "nodes": ["analyze", "review", "suggest"]
  }
}
```

### Execute Workflow
```http
POST /api/graph/workflow/{workflow_id}/execute
```

**Description**: Execute a workflow with optional initial state

**Parameters**:
- `workflow_id` (path): The workflow identifier

**Request Body**:
```json
{
  "code": "def hello(): return 'world'",
  "language": "python"
}
```

**Response**:
```json
{
  "workflow_id": "workflow_123",
  "status": "completed",
  "results": {
    "analysis": "Code quality: Good",
    "review": "No issues found",
    "suggestions": ["Add type hints", "Add docstring"]
  }
}
```

### Get Workflow Details
```http
GET /api/graph/workflow/{workflow_id}
```

**Description**: Get workflow configuration and status

**Parameters**:
- `workflow_id` (path): The workflow identifier

**Response**:
```json
{
  "workflow_id": "workflow_123",
  "workflow": {
    "name": "Code Review Workflow",
    "description": "Automated code review process",
    "nodes": ["analyze", "review", "suggest"]
  },
  "status": "completed"
}
```

### Delete Workflow
```http
DELETE /api/graph/workflow/{workflow_id}
```

**Description**: Delete a workflow

**Parameters**:
- `workflow_id` (path): The workflow identifier

**Response**:
```json
{
  "workflow_id": "workflow_123",
  "status": "deleted"
}
```

## 🗄️ Database API (`/api/astra`)

### List Collections
```http
GET /api/astra/collections
```

**Description**: List all available document collections

**Response**:
```json
{
  "collections": [
    "strategy_book",
    "langchain_docs",
    "fastapi_docs",
    "nextjs_docs",
    "pydantic_docs"
  ],
  "status": "success",
  "message": "Found 5 collections"
}
```

### Search Documents
```http
POST /api/astra/search
```

**Description**: Search for documents using semantic similarity

**Request Body**:
```json
{
  "query": "FastAPI authentication best practices",
  "collection": "fastapi_docs",
  "limit": 5
}
```

**Response**:
```json
{
  "results": [
    {
      "content": "FastAPI provides several authentication methods...",
      "metadata": {
        "source": "fastapi_docs",
        "title": "Authentication Guide",
        "url": "https://fastapi.tiangolo.com/tutorial/security/"
      },
      "score": 0.95
    }
  ],
  "status": "success",
  "message": "Found 5 documents"
}
```

### Database Health Check
```http
GET /api/astra/health
```

**Description**: Check AstraDB connection health

**Response**:
```json
{
  "status": "healthy",
  "collections_available": 5,
  "initialized": true,
  "message": "AstraDB connection is healthy"
}
```

## 🤖 Agent Types

### Available Agents

| Agent | Role | Capabilities |
|-------|------|-------------|
| `supervisor` | Orchestrator | Task routing, workflow management |
| `strategic_expert` | Strategist | High-level planning, architecture decisions |
| `langchain_expert` | LangChain Developer | Chain building, agent development |
| `fastapi_expert` | Backend Developer | API design, async programming |
| `nextjs_expert` | Frontend Developer | React development, SSR/SSG |
| `pydantic_expert` | Data Modeler | Schema design, validation |

### Agent Capabilities

#### Strategic Expert
- Project planning and roadmapping
- Architecture decision making
- Technology stack recommendations
- Risk assessment and mitigation

#### LangChain Expert
- LangChain application development
- Custom chain creation
- Agent orchestration
- LLM integration patterns

#### FastAPI Expert
- REST API design and implementation
- Async programming patterns
- Database integration
- Authentication and authorization

#### Next.js Expert
- React component development
- Server-side rendering (SSR)
- Static site generation (SSG)
- Modern frontend practices

#### Pydantic Expert
- Data model design
- Validation schema creation
- Serialization/deserialization
- Type safety implementation

## 📊 Response Formats

### Success Response
```json
{
  "success": true,
  "data": { /* response data */ },
  "message": "Operation completed successfully",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": { /* error details */ }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Streaming Response
For real-time operations, responses may be streamed:

```json
{"type": "message", "data": {"content": "Processing your request..."}}
{"type": "progress", "data": {"step": 1, "total": 3, "description": "Analyzing code"}}
{"type": "result", "data": {"analysis": "Code quality: Good"}}
{"type": "complete", "data": {"status": "success"}}
```

## 🔧 Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Invalid request parameters | 400 |
| `AGENT_NOT_FOUND` | Specified agent doesn't exist | 404 |
| `WORKFLOW_NOT_FOUND` | Workflow ID not found | 404 |
| `COLLECTION_NOT_FOUND` | Database collection not found | 404 |
| `PROCESSING_ERROR` | Error during task processing | 500 |
| `DATABASE_ERROR` | Database connection or query error | 503 |
| `AGENT_TIMEOUT` | Agent response timeout | 504 |

## 🚀 Rate Limiting

Current rate limits (development):
- **Chat API**: 60 requests per minute
- **Search API**: 100 requests per minute
- **Workflow API**: 30 requests per minute
- **System API**: 200 requests per minute

## 📝 Request Examples

### cURL Examples

#### Send Chat Message
```bash
curl -X POST "http://localhost:8000/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I create a FastAPI endpoint?",
    "conversation_id": "conv_123"
  }'
```

#### Search Documents
```bash
curl -X POST "http://localhost:8000/api/astra/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication middleware",
    "collection": "fastapi_docs",
    "limit": 3
  }'
```

#### Coordinate Multi-Agent Task
```bash
curl -X POST "http://localhost:8000/multi-agent/coordinate" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Create a React component with TypeScript",
    "project_id": "react_project_001"
  }'
```

### JavaScript Examples

#### Using Fetch API
```javascript
// Send chat message
const response = await fetch('http://localhost:8000/api/chat/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: 'How do I implement authentication?',
    conversation_id: 'conv_123'
  })
});

const data = await response.json();
console.log(data.response);
```

#### Using Axios
```javascript
import axios from 'axios';

// Search documents
const searchResults = await axios.post('http://localhost:8000/api/astra/search', {
  query: 'FastAPI best practices',
  collection: 'fastapi_docs',
  limit: 5
});

console.log(searchResults.data.results);
```

### Python Examples

#### Using Requests
```python
import requests

# Coordinate multi-agent task
response = requests.post('http://localhost:8000/multi-agent/coordinate', json={
    'task_description': 'Build a REST API with authentication',
    'project_id': 'api_project_001'
})

result = response.json()
print(result['result'])
```

#### Using httpx (Async)
```python
import httpx
import asyncio

async def chat_with_agent():
    async with httpx.AsyncClient() as client:
        response = await client.post('http://localhost:8000/api/chat/', json={
            'message': 'Explain LangChain agents',
            'conversation_id': 'conv_456'
        })
        return response.json()

result = asyncio.run(chat_with_agent())
print(result['response'])
```

## 🔍 Monitoring & Debugging

### Health Check Endpoints
- `/health` - Overall system health
- `/api/astra/health` - Database health
- `/api/docs` - Interactive API documentation

### Logging
All API requests are logged with:
- Request ID
- Timestamp
- Endpoint
- Response time
- Status code
- Error details (if any)

### Debug Mode
Enable debug mode by setting `DEBUG=true` in environment variables for:
- Detailed error messages
- Request/response logging
- Performance metrics
- Agent interaction traces

## 🔮 Future Enhancements

### Planned Features
- [ ] WebSocket support for real-time chat
- [ ] Batch processing endpoints
- [ ] File upload and processing
- [ ] Advanced workflow builder
- [ ] Plugin system API
- [ ] Team collaboration endpoints
- [ ] Analytics and metrics API

### API Versioning
Future versions will use URL versioning:
- `v1`: Current API (default)
- `v2`: Enhanced features (planned)
- `v3`: Advanced capabilities (future)

---

**For more information, visit the interactive API documentation at `/api/docs`** 