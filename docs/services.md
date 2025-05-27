# Service Architecture Documentation

## Overview

The Code Hero service architecture is designed with modularity, testability, and maintainability in mind. It follows a clean separation of concerns and uses dependency injection for service management, enhanced with LangGraph integration and multi-agent coordination capabilities.

## Core Components

### 1. Service Interfaces (`interfaces.py`)

The base interface that all services must implement:

```python
@runtime_checkable
class ServiceInterface(Protocol):
    async def initialize(self) -> None
    async def cleanup(self) -> None
    async def check_health(self) -> dict
```

Service health status model:

```python
class ServiceStatus(BaseModel):
    name: str
    status: Status
    details: dict = {}
    error: str | None = None
```

### 2. Service Types (`types.py`)

Type definitions for service dependencies:

```python
StateManagerType = TypeVar("StateManagerType")
LoggerType = TypeVar("LoggerType")
SupervisorType = TypeVar("SupervisorType")
```

### 3. Service Management (`services.py`)

Enhanced service management functionality:
- Service validation with health monitoring
- Error handling and recovery
- Dependency injection with proper lifecycle management
- Multi-agent coordination support

Key components:
- `get_services()`: Main service provider with enhanced error handling
- `validate_services()`: Comprehensive service validation
- Custom exceptions for service errors with detailed context

## Core Services

### 1. State Manager (`manager.py`)

**Enhanced Responsibilities:**
- Manages application state with persistence
- Handles state transitions with checkpointing
- Maintains state consistency across multi-agent workflows
- Project management with enhanced tracking
- Agent state coordination
- LangGraph workflow state management

**Key Features:**
```python
class StateManager(ServiceInterface):
    async def initialize(self) -> None
    async def add_state(self, state_id: str, state: BaseState) -> None
    async def get_state(self, state_id: str) -> Optional[BaseState]
    async def update_state(self, state_id: str, state: BaseState) -> None
    async def create_project_state(self, project_id: str, **kwargs) -> ProjectState
    async def get_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]
```

### 2. Structured Logger (`logger.py`)

**Enhanced Responsibilities:**
- Structured logging with JSON format
- Event tracking across multi-agent workflows
- Error logging with context preservation
- State change logging for debugging
- Metrics logging for performance monitoring
- LangGraph workflow event logging

**Key Features:**
```python
class StructuredLogger(ServiceInterface):
    def log_event(self, event: str, data: Dict[str, Any], context: Optional[BaseState] = None)
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None)
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None)
    def log_agent_interaction(self, agent_role: AgentRole, action: str, result: Dict[str, Any])
    def log_tool_execution(self, tool_name: str, result: Dict[str, Any], duration: float)
```

### 3. Supervisor Agent (`supervisor.py`)

**Enhanced Responsibilities:**
- Multi-agent workflow orchestration
- Task management and routing
- Agent coordination with tool binding
- Error handling and recovery
- Human-in-the-loop integration
- LangGraph workflow execution

**Key Features:**
```python
class SupervisorAgent(ServiceInterface):
    async def coordinate_multi_agent_task(self, task_description: str, project_id: str) -> Dict[str, Any]
    async def start_workflow(self, workflow_id: str, agents: List[AgentRole]) -> WorkflowState
    async def route_task_to_agent(self, task: str, available_agents: List[AgentRole]) -> AgentRole
    async def execute_workflow_graph(self, initial_state: Dict[str, Any]) -> Dict[str, Any]
```

### 4. Enhanced Tools Registry (`tools.py`)

**New Service Component:**
- Tool management and categorization
- LangGraph tool binding
- Content optimization for LLM efficiency
- Error handling for tool execution
- Category-based tool assignment

**Key Features:**
```python
class ToolRegistry:
    def get_tools_by_category(self, category: str) -> List[BaseTool]
    def create_tool_node(self, category: Optional[str] = None) -> ToolNode
    def bind_tools_to_llm(self, llm, category: Optional[str] = None)
    def get_tool_descriptions(self, category: Optional[str] = None) -> Dict[str, str]
```

### 5. Agent Expert System (`agent_expert.py`)

**Enhanced Multi-Agent System:**
- 10 specialized agents with distinct roles
- Tool binding per agent category
- State management across agents
- Error handling and recovery
- Performance optimization

**Available Agents:**
```python
experts = {
    AgentRole.SUPERVISOR: SupervisorExpert(),
    AgentRole.STRATEGIC_EXPERT: StrategicExpert(),
    AgentRole.LANGCHAIN_EXPERT: LangChainExpert(),
    AgentRole.FASTAPI_EXPERT: FastAPIExpert(),
    AgentRole.NEXTJS_EXPERT: NextJSExpert(),
    AgentRole.RESEARCH: ResearchExpert(),
    AgentRole.IMPLEMENTATION: ImplementationExpert(),
    AgentRole.CODE_REVIEWER: CodeReviewerExpert(),
    AgentRole.DOCUMENTATION: DocumentationExpert(),
    AgentRole.PROMPT_ENGINEER: PromptEngineerExpert(),
}
```

## Service Lifecycle

### 1. **Enhanced Initialization**
- Services are initialized during application startup with dependency validation
- Each service implements `initialize()` with proper error handling
- Resources are allocated with monitoring
- Initial state is set up with persistence
- Tool registry is initialized with category mapping
- Agent system is configured with tool binding

### 2. **Runtime Operations**
- Services are accessed via dependency injection with health checks
- Health checks ensure service availability with detailed status
- Error handling manages service failures with recovery mechanisms
- State is maintained consistently across multi-agent workflows
- Tool execution is monitored and optimized
- Agent coordination is managed through supervisor

### 3. **Enhanced Cleanup**
- Services implement `cleanup()` with proper resource deallocation
- Resources are released with verification
- State is saved with persistence validation
- Graceful shutdown is ensured with timeout handling
- Tool connections are properly closed
- Agent states are persisted for recovery

## Enhanced Error Handling

### Custom Exceptions
```python
class ServiceError(Exception): """Base exception for service errors"""
class ServiceNotInitializedError(ServiceError): """Service not properly initialized"""
class ServiceHealthCheckError(ServiceError): """Health check failed with details"""
class MultiAgentCoordinationError(ServiceError): """Multi-agent coordination failed"""
class ToolExecutionError(ServiceError): """Tool execution failed"""
class WorkflowExecutionError(ServiceError): """LangGraph workflow execution failed"""
```

### Error Recovery Mechanisms
- **Automatic Retry**: Failed operations are retried with exponential backoff
- **Fallback Services**: Alternative implementations for critical services
- **Circuit Breakers**: Prevent cascade failures in multi-agent workflows
- **State Rollback**: Restore previous known good state on failures

## Multi-Agent Coordination

### Service Integration Pattern
```python
async def coordinate_multi_agent_task(
    task_description: str,
    project_id: str,
    services: Tuple[StateManager, StructuredLogger, SupervisorAgent]
) -> Dict[str, Any]:
    """Coordinate multi-agent tasks with proper service integration."""
    state_manager, logger, supervisor = services
    
    try:
        # Create project state
        project_state = await state_manager.create_project_state(project_id)
        
        # Log task initiation
        logger.log_event("multi_agent_task_started", {
            "task_description": task_description,
            "project_id": project_id
        })
        
        # Execute workflow
        result = await supervisor.coordinate_multi_agent_task(
            task_description, project_id
        )
        
        # Update state
        await state_manager.update_state(project_id, project_state)
        
        return result
        
    except Exception as e:
        logger.log_error(e, {"project_id": project_id})
        raise ServiceError(f"Multi-agent coordination failed: {str(e)}")
```

## Tool Integration Services

### Tool Binding Service
```python
class ToolBindingService:
    """Service for managing tool binding to agents."""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
    
    async def bind_tools_to_agent(self, agent_role: AgentRole) -> List[BaseTool]:
        """Bind appropriate tools to agent based on role."""
        category_mapping = {
            AgentRole.RESEARCH: "research",
            AgentRole.IMPLEMENTATION: "coding",
            AgentRole.CODE_REVIEWER: "validation",
            AgentRole.SUPERVISOR: "all"
        }
        
        category = category_mapping.get(agent_role, "research")
        return self.tool_registry.get_tools_by_category(category)
```

## Performance Monitoring Services

### Metrics Collection
```python
class MetricsService(ServiceInterface):
    """Service for collecting and reporting performance metrics."""
    
    async def record_agent_execution_time(self, agent_role: AgentRole, duration: float)
    async def record_tool_execution_time(self, tool_name: str, duration: float)
    async def record_workflow_completion_time(self, workflow_id: str, duration: float)
    async def get_performance_summary(self) -> Dict[str, Any]
```

## Best Practices

### 1. **Enhanced Service Implementation**
- Implement `ServiceInterface` with proper error handling
- Handle initialization and cleanup with resource verification
- Provide comprehensive health checks with detailed status
- Use structured error handling with context preservation
- Implement proper logging with correlation IDs

### 2. **Dependency Management**
- Use type hints with proper validation
- Avoid circular dependencies through interface design
- Use dependency injection with lifecycle management
- Follow interface contracts with comprehensive testing
- Implement proper service discovery mechanisms

### 3. **State Management**
- Keep state consistent across multi-agent workflows
- Use atomic operations with proper locking
- Handle edge cases with comprehensive validation
- Validate state changes with schema enforcement
- Implement proper state persistence with backup

### 4. **Error Handling**
- Use custom exceptions with detailed context
- Provide detailed error messages with actionable information
- Log errors properly with correlation tracking
- Handle cleanup on errors with resource verification
- Implement proper retry mechanisms with circuit breakers

## Enhanced Usage Example

```python
# Get services with enhanced error handling
async def get_enhanced_services(request):
    """Get services with comprehensive validation."""
    try:
        services = await get_services(request)
        state_manager, logger, supervisor = services
        
        # Validate service health
        health_checks = await asyncio.gather(
            state_manager.check_health(),
            logger.check_health(),
            supervisor.check_health(),
            return_exceptions=True
        )
        
        # Check for any failed health checks
        for i, health in enumerate(health_checks):
            if isinstance(health, Exception):
                service_name = ["state_manager", "logger", "supervisor"][i]
                raise ServiceHealthCheckError(f"{service_name} health check failed: {health}")
        
        return services
        
    except Exception as e:
        raise ServiceError(f"Service initialization failed: {str(e)}")

# Use services with multi-agent coordination
async def execute_multi_agent_workflow(request):
    """Execute multi-agent workflow with proper service integration."""
    try:
        # Get validated services
        services = await get_enhanced_services(request)
        state_manager, logger, supervisor = services
        
        # Initialize workflow with tool binding
        workflow_id = f"workflow_{generate_id()}"
        
        # Create workflow state
        workflow_state = await state_manager.create_workflow_state(workflow_id)
        
        # Log workflow initiation
        logger.log_event("workflow_started", {
            "workflow_id": workflow_id,
            "agents": ["research", "implementation", "review"]
        })
        
        # Execute workflow with enhanced coordination
        result = await supervisor.execute_workflow_graph({
            "workflow_id": workflow_id,
            "task": "Generate and validate FastAPI application",
            "agents": [AgentRole.RESEARCH, AgentRole.IMPLEMENTATION, AgentRole.CODE_REVIEWER]
        })
        
        # Update final state
        await state_manager.update_workflow_state(workflow_id, result)
        
        # Log completion
        logger.log_event("workflow_completed", {
            "workflow_id": workflow_id,
            "status": result.get("status"),
            "duration": result.get("duration")
        })
        
        return result
        
    except ServiceError as e:
        # Handle service-specific errors
        logger.log_error(e, {"workflow_id": workflow_id})
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Handle unexpected errors
        logger.log_error(e, {"workflow_id": workflow_id})
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Testing Strategy

### 1. **Unit Testing**
- Mock service dependencies with proper interfaces
- Test service interfaces with comprehensive scenarios
- Validate error handling with edge cases
- Check state transitions with various inputs
- Test tool binding with different agent roles

### 2. **Integration Testing**
- Test service interactions with real dependencies
- Validate multi-agent workflows end-to-end
- Check error propagation across service boundaries
- Test cleanup procedures with failure scenarios
- Validate performance under load

### 3. **Health Check Testing**
- Test health check endpoints with various states
- Validate service recovery mechanisms
- Test circuit breaker functionality
- Validate monitoring and alerting systems

## Current Implementation Status

### ✅ **Fully Implemented Services**
- **State Manager**: Complete with persistence and checkpointing
- **Structured Logger**: Enhanced with multi-agent workflow logging
- **Supervisor Agent**: Full multi-agent coordination capabilities
- **Tool Registry**: Complete with LangGraph integration
- **Agent Expert System**: 10 specialized agents with tool binding
- **Service Management**: Comprehensive dependency injection and health monitoring

### ⚠️ **Partially Implemented**
- **Metrics Service**: Basic implementation, needs enhancement
- **Circuit Breakers**: Basic error handling, needs advanced patterns
- **Performance Monitoring**: Basic logging, needs detailed metrics

### 🎯 **Future Improvements**

1. **Advanced Service Registry**
   - Dynamic service registration and discovery
   - Service versioning and compatibility checking
   - Hot reloading capabilities
   - Advanced load balancing

2. **Enhanced Monitoring**
   - Detailed performance metrics with dashboards
   - Real-time alerting systems
   - Resource usage optimization
   - Predictive failure detection

3. **Advanced Features**
   - Horizontal service scaling
   - Advanced circuit breaker patterns
   - Intelligent retry mechanisms with ML
   - Service mesh integration

The enhanced service architecture provides a robust foundation for multi-agent AI workflows with proper tool binding, state management, error recovery, and comprehensive monitoring capabilities. 