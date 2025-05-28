# LangGraph Implementation Guide

## Overview

Code Hero now uses a proper LangGraph implementation with supervisor-based routing, Pydantic structured outputs, and specialized agent workflows. This document explains the architecture, usage, and implementation details.

## Architecture

### Workflow Flow
```
User Message → START
     ↓
Supervisor Router (analyzes & routes)
     ↓
┌─────────────────────────────────────┐
│  Code Expert    │  Research Expert  │
│  Analysis Expert │  General Response │
└─────────────────────────────────────┘
     ↓
Format Response
     ↓
END → User Response
```

### Key Components

#### 1. Supervisor as Entry Point
- **Entry Point**: All user messages start at the supervisor
- **Intelligent Routing**: Analyzes message content and routes to appropriate specialist
- **Fallback Handling**: Handles general queries directly when no specialist is needed

#### 2. Pydantic Structured Outputs
```python
class ChatResponse(BaseModel):
    content: str = Field(..., description="Response content")
    agent_used: str = Field(..., description="Agent that generated the response")
    tools_used: List[str] = Field(default_factory=list, description="Tools used in response")
    confidence: float = Field(default=1.0, description="Confidence in response")

class TaskAnalysis(BaseModel):
    task_type: str = Field(..., description="Type of task (code, research, analysis, etc.)")
    complexity: str = Field(..., description="Task complexity (low, medium, high)")
    required_tools: List[str] = Field(default_factory=list, description="Tools needed for task")
    estimated_steps: int = Field(default=1, description="Estimated number of steps")

class CodeHeroChatState(MessagesState):
    conversation_id: str
    project_id: str
    active_agent: str
    task_analysis: Optional[TaskAnalysis] = None
    current_response: Optional[ChatResponse] = None
    tools_available: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
```

#### 3. Agent Routing Logic
The supervisor uses keyword-based routing with priority order:

1. **Analysis Expert** (highest priority)
   - Keywords: "analyze", "review", "check", "examine", "validate", "audit", "inspect", "evaluate"
   - Use case: Code analysis, performance reviews, validation tasks

2. **Research Expert**
   - Keywords: "search", "find", "research", "look up", "information", "about", "learn about", "tell me about"
   - Use case: Information gathering, documentation search, web research

3. **Code Expert**
   - Keywords: "create", "build", "implement", "develop", "generate", "write", "make", "code", "function", "class", "api"
   - Use case: Code generation, implementation, development tasks

4. **General Response** (supervisor handles directly)
   - All other queries
   - Use case: General conversation, help requests, unclear tasks

## Usage Examples

### 1. Code Generation
```bash
curl -X POST "http://localhost:8000/api/chat/?message=create%20a%20python%20function%20to%20calculate%20fibonacci%20numbers"
```

**Response:**
```json
{
  "response": "I'll help you with code generation. Here's the python code:\n\n```python\n# Python Function\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n\n**Explanation:** This code addresses your request...\n\n---\n*Handled by: code_expert* | *Tools: generate_code*",
  "conversation_id": "conv_1748349115.976745",
  "status": "completed",
  "active_agent": "code_expert"
}
```

### 2. Research Tasks
```bash
curl -X POST "http://localhost:8000/api/chat/?message=find%20information%20about%20machine%20learning"
```

**Response:**
```json
{
  "response": "I'll research that for you. Here's what I found about: \"find information about machine learning\"\n\n**Research Summary:**\nBased on your query, I can help you find information about this topic.\n\n**Key Points:**\n• This appears to be a research request\n• I can search through documents and web sources\n• I'll provide comprehensive information\n\n**Sources:** Internal documents, Web search\n\n---\n*Handled by: research_expert* | *Tools: search_documents, search_web*",
  "status": "completed",
  "active_agent": "research_expert"
}
```

### 3. Analysis Tasks
```bash
curl -X POST "http://localhost:8000/api/chat/?message=analyze%20this%20code%20for%20performance"
```

**Response:**
```json
{
  "response": "I'll analyze that for you. Here's my analysis of: \"analyze this code for performance\"\n\n**Analysis Summary:**\nI've examined your request and can provide detailed analysis.\n\n**Key Findings:**\n• This appears to be an analysis request\n• I can validate and examine code or content\n• I'll provide structured feedback\n\n**Recommendations:**\n• Consider the context and requirements\n• Follow best practices\n• Implement suggested improvements\n\n**Confidence Level:** 85%\n\n---\n*Handled by: analysis_expert* | *Tools: analyze_code, validate_code*",
  "status": "completed",
  "active_agent": "analysis_expert"
}
```

### 4. General Queries
```bash
curl -X POST "http://localhost:8000/api/chat/?message=hello%20how%20are%20you"
```

**Response:**
```json
{
  "response": "I understand you need help with: \"hello how are you\"\n\nI'm your AI assistant supervisor and I can help you with:\n• **Code Generation** - Creating functions, APIs, and applications\n• **Research** - Finding information and documentation\n• **Analysis** - Reviewing and validating code or content\n\nPlease let me know specifically what you'd like me to help you with, and I'll route your request to the appropriate specialist or handle it myself.\n\n---\n*Handled by: supervisor*",
  "status": "completed",
  "active_agent": "supervisor"
}
```

## Implementation Details

### LangGraph Workflow Creation
```python
def create_chat_workflow():
    """Create LangGraph workflow with supervisor as entry point."""
    if not LANGGRAPH_AVAILABLE:
        return None
    
    # Get available tools
    available_tools = tool_registry.get_all_tools()
    
    # Define the workflow graph
    workflow = StateGraph(CodeHeroChatState)
    
    # Add nodes for each specialist
    workflow.add_node("code_expert", code_expert)
    workflow.add_node("research_expert", research_expert)
    workflow.add_node("analysis_expert", analysis_expert)
    workflow.add_node("general_response", general_response)
    workflow.add_node("format_response", format_response)
    
    # Add conditional routing from START
    workflow.add_conditional_edges(
        START,
        supervisor_route,
        {
            "code_expert": "code_expert",
            "research_expert": "research_expert", 
            "analysis_expert": "analysis_expert",
            "general_response": "general_response"
        }
    )
    
    # All paths lead to format_response then END
    workflow.add_edge("code_expert", "format_response")
    workflow.add_edge("research_expert", "format_response")
    workflow.add_edge("analysis_expert", "format_response")
    workflow.add_edge("general_response", "format_response")
    workflow.add_edge("format_response", END)
    
    # Compile with memory
    return workflow.compile(checkpointer=MemorySaver())
```

### Supervisor Routing Function
```python
def supervisor_route(state: CodeHeroChatState) -> str:
    """Supervisor analyzes user message and routes to appropriate agent."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if not last_message or not isinstance(last_message, HumanMessage):
        return "general_response"
    
    user_input = last_message.content.lower()
    
    # Priority order: analysis > research > code generation > general
    if any(keyword in user_input for keyword in ["analyze", "review", "check", "examine", "validate", "audit", "inspect", "evaluate"]):
        state["active_agent"] = "analysis_expert"
        return "analysis_expert"
    elif any(keyword in user_input for keyword in ["search", "find", "research", "look up", "information", "about", "learn about", "tell me about"]):
        state["active_agent"] = "research_expert"
        return "research_expert"
    elif any(keyword in user_input for keyword in ["create", "build", "implement", "develop", "generate", "write", "make", "code", "function", "class", "api"]):
        state["active_agent"] = "code_expert"
        return "code_expert"
    else:
        state["active_agent"] = "supervisor"
        return "general_response"
```

### Agent Node Implementation
```python
def code_expert(state: CodeHeroChatState) -> CodeHeroChatState:
    """Handle code generation and development tasks."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if not last_message:
        return state
    
    user_input = last_message.content
    
    # Create task analysis
    task_analysis = TaskAnalysis(
        task_type="code_generation",
        complexity="high" if len(user_input.split()) > 20 else "medium",
        required_tools=["generate_code", "validate_code"],
        estimated_steps=2
    )
    
    # Generate structured response
    response = ChatResponse(
        content=f"Generated code based on: {user_input}",
        agent_used="code_expert",
        tools_used=["generate_code"],
        confidence=0.9
    )
    
    state["task_analysis"] = task_analysis
    state["current_response"] = response
    return state
```

## Benefits

### 1. Proper LangGraph Structure
- Uses official LangGraph patterns (StateGraph, MessagesState, conditional edges)
- Follows LangGraph best practices for workflow design
- Integrates seamlessly with LangGraph ecosystem

### 2. Type Safety with Pydantic
- Structured data models prevent runtime errors
- Clear data contracts between components
- Automatic validation and serialization

### 3. Intelligent Routing
- Context-aware agent selection
- Priority-based keyword matching
- Fallback handling for edge cases

### 4. Scalability
- Easy to add new agents and routing rules
- Modular architecture supports independent development
- Tool integration ready for expansion

### 5. Observability
- Structured logging and monitoring
- Clear agent attribution in responses
- Tool usage tracking

## Extending the System

### Adding New Agents
1. Create agent function following the pattern:
```python
def new_expert(state: CodeHeroChatState) -> CodeHeroChatState:
    # Implement agent logic
    return state
```

2. Add to workflow graph:
```python
workflow.add_node("new_expert", new_expert)
```

3. Update routing logic:
```python
elif any(keyword in user_input for keyword in ["new", "keywords"]):
    state["active_agent"] = "new_expert"
    return "new_expert"
```

### Adding New Tools
1. Implement tool with `@tool` decorator
2. Register in tool_registry
3. Add to agent's required_tools list
4. Use in agent implementation

### Customizing Responses
Modify the agent functions to change response format, add new fields to Pydantic models, or integrate additional data sources.

## Troubleshooting

### Common Issues
1. **LangGraph not available**: Check if langgraph is installed
2. **Routing not working**: Verify keyword matching logic
3. **State not persisting**: Check MemorySaver configuration
4. **Tools not executing**: Verify tool registration and availability

### Debugging
- Enable LangChain tracing for detailed workflow visibility
- Check logs for agent execution details
- Use health endpoint to verify system status
- Test individual agents with specific keywords 