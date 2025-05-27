# Code Hero Architecture

## System Overview

```mermaid
graph TB
    subgraph API["API Layer"]
        A[FastAPI Application]
        A --> B[Chat Routes]
        A --> C[Multi-Agent Routes]
        A --> D[AstraDB Routes]
        A --> E[Health Check]
    end

    subgraph Core["Core System"]
        F[State Manager]
        G[Agent System]
        H[LangGraph Workflow Engine]
        I[Enhanced Tools Registry]
        J[Supervisor Coordinator]
        
        F --> G
        F --> H
        F --> I
        F --> J
    end

    subgraph Agents["Agent Layer"]
        K[Supervisor]
        L[Strategic Expert]
        M[LangChain Expert]
        N[FastAPI Expert]
        O[Next.js Expert]
        P[Research Agent]
        Q[Implementation Agent]
        R[Code Reviewer]
        S[Documentation Agent]
        T[Prompt Engineer]
        
        K --> L
        K --> M
        K --> N
        K --> O
        K --> P
        K --> Q
        K --> R
        K --> S
        K --> T
    end

    subgraph Tools["Enhanced Tools"]
        U[Document Search]
        V[Web Search]
        W[Content Fetching]
        X[Code Generation]
        Y[Code Validation]
        Z[Code Analysis]
    end

    subgraph Storage["Storage Layer"]
        AA[AstraDB]
        BB[11 Vector Collections]
        CC[State Persistence]
        DD[Checkpointing]
        
        AA --> BB
        AA --> CC
        AA --> DD
    end

    subgraph External["External Services"]
        EE[OpenAI]
        FF[LangChain/LangGraph]
        GG[Tavily API]
        HH[Other LLM Providers]
    end

    API --> Core
    Core --> Agents
    Core --> Tools
    Core --> Storage
    Agents --> Tools
    Agents --> External
    Tools --> Storage
```

## Enhanced Component Interactions

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI
    participant S as Supervisor
    participant WF as LangGraph Workflow
    participant AG as Specialized Agent
    participant T as Enhanced Tools
    participant V as Vector Store
    
    C->>API: Multi-Agent Request
    API->>S: Coordinate Task
    S->>WF: Initialize Workflow
    
    loop Multi-Agent Execution
        WF->>AG: Route to Specialized Agent
        AG->>T: Bind & Execute Tools
        T->>V: Query/Update Data
        V-->>T: Semantic Results
        T-->>AG: Tool Response
        AG-->>WF: Agent Result
        WF->>S: Update State
    end
    
    S-->>API: Final Result
    API-->>C: Coordinated Response
```

## LangGraph Workflow States

```mermaid
stateDiagram-v2
    [*] --> INITIALIZING
    INITIALIZING --> SUPERVISOR_ROUTING
    SUPERVISOR_ROUTING --> RESEARCH_AGENT
    SUPERVISOR_ROUTING --> WRITER_AGENT
    SUPERVISOR_ROUTING --> CODING_AGENT
    SUPERVISOR_ROUTING --> REVIEW_AGENT
    
    RESEARCH_AGENT --> TOOL_EXECUTION
    WRITER_AGENT --> TOOL_EXECUTION
    CODING_AGENT --> TOOL_EXECUTION
    REVIEW_AGENT --> TOOL_EXECUTION
    
    TOOL_EXECUTION --> AGENT_COMPLETE
    AGENT_COMPLETE --> NEEDS_REVIEW
    AGENT_COMPLETE --> NEEDS_FEEDBACK
    AGENT_COMPLETE --> WORKFLOW_COMPLETE
    
    NEEDS_REVIEW --> REVIEW_AGENT
    NEEDS_FEEDBACK --> HUMAN_FEEDBACK
    HUMAN_FEEDBACK --> SUPERVISOR_ROUTING
    
    WORKFLOW_COMPLETE --> [*]
```

## System Components

### API Layer
- **FastAPI Application**: Modern async web framework with lifespan handlers
- **Multi-Agent Routes**: Coordination endpoints for complex tasks
- **Chat Routes**: Interactive agent communication
- **AstraDB Routes**: Vector database operations
- **Health Check**: System monitoring and status

### Core System
- **State Manager**: Persistent state with checkpointing support
- **Agent System**: 10 specialized agents with role-based tool binding
- **LangGraph Workflow Engine**: Multi-agent orchestration with proper invoke patterns
- **Enhanced Tools Registry**: 6 core tools with category-based assignment
- **Supervisor Coordinator**: Task routing and workflow management

### Agent Layer (10 Specialized Agents)
- **Supervisor**: Orchestrates workflows, manages task routing
- **Strategic Expert**: Strategic planning, decision-making, framework analysis
- **LangChain Expert**: LangChain development, chains, agents, workflows
- **FastAPI Expert**: FastAPI development, REST APIs, backend systems
- **Next.js Expert**: Next.js development, React components, frontend
- **Research Agent**: Information gathering, analysis, documentation search
- **Implementation Agent**: Code generation, implementation, development
- **Code Reviewer**: Code review, quality assurance, validation
- **Documentation Agent**: Documentation generation, technical writing
- **Prompt Engineer**: Enhanced prompt engineering using industry techniques

### Enhanced Tools System
- **Document Search**: Semantic search across 11 AstraDB collections
- **Web Search**: Real-time web search using Tavily API
- **Content Fetching**: Asynchronous web content retrieval
- **Code Generation**: Context-aware templates for FastAPI, Next.js, Python
- **Code Validation**: Comprehensive syntax and quality checking
- **Code Analysis**: Pattern detection, complexity analysis, quality metrics

### Storage Layer
- **AstraDB**: Vector database with OpenAI embeddings
- **11 Vector Collections**: Specialized knowledge domains
- **State Persistence**: Project, agent, and chat state management
- **Checkpointing**: LangGraph workflow state preservation

### External Services
- **OpenAI**: Primary LLM provider with embeddings
- **LangChain/LangGraph**: AI framework and workflow orchestration
- **Tavily API**: Real-time web search capabilities
- **Other LLM Providers**: Deepseek, Groq for diversity

## Enhanced Data Flow

```mermaid
flowchart LR
    subgraph Input
        A[User Request]
        B[Multi-Agent Task]
        C[API Call]
    end

    subgraph Processing
        D[Supervisor Routing]
        E[Agent Selection]
        F[Tool Binding]
        G[LangGraph Execution]
        H[State Management]
        
        D --> E
        E --> F
        F --> G
        G --> H
        H --> D
    end

    subgraph Tools
        I[Enhanced Tools]
        J[Category Assignment]
        K[Content Optimization]
        L[Error Handling]
    end

    subgraph Storage
        M[AstraDB Collections]
        N[Vector Search]
        O[State Persistence]
        P[Checkpointing]
    end

    subgraph Output
        Q[Coordinated Response]
        R[Generated Code]
        S[Analysis Results]
        T[Documentation]
    end

    Input --> Processing
    Processing --> Tools
    Tools --> Storage
    Storage --> Tools
    Processing --> Output
```

## Tool Binding Architecture

```mermaid
graph TD
    subgraph ToolRegistry["Enhanced Tool Registry"]
        A[Tool Categories]
        B[Research Tools]
        C[Coding Tools]
        D[Content Tools]
        E[Validation Tools]
        F[Web Tools]
        
        A --> B
        A --> C
        A --> D
        A --> E
        A --> F
    end

    subgraph AgentBinding["Agent Tool Binding"]
        G[Supervisor - All Tools]
        H[Research - Search Tools]
        I[Coding - Development Tools]
        J[Review - Validation Tools]
        K[Strategic - Analysis Tools]
    end

    subgraph LangGraphIntegration["LangGraph Integration"]
        L[ToolNode Creation]
        M[LLM Tool Binding]
        N[Invoke Method]
        O[Error Handling]
    end

    ToolRegistry --> AgentBinding
    AgentBinding --> LangGraphIntegration
```

## Vector Store Collections Architecture

```mermaid
graph LR
    subgraph AstraDB["AstraDB Collections (11 Total)"]
        A[strategy_book]
        B[langchain_docs]
        C[langgraph_docs]
        D[llamaindex_docs]
        E[nextjs_docs]
        F[crewai_docs]
        G[fastapi_docs]
        H[langsmith_docs]
        I[pydantic_docs]
        J[agno_phidata_docs]
        K[framework_docs]
    end

    subgraph Usage["Collection Usage Patterns"]
        L[Strategic Planning]
        M[Framework Development]
        N[Code Generation]
        O[Documentation]
        
        A --> L
        B & C & D & F & H --> M
        E & G & I --> N
        A & K --> O
    end

    subgraph Search["Enhanced Search"]
        P[Semantic Similarity]
        Q[Content Optimization]
        R[Relevance Scoring]
        S[Result Formatting]
    end

    AstraDB --> Search
```

## Deployment Architecture

```mermaid
flowchart TB
    subgraph Client
        A[CLI Interface]
        B[API Client]
        C[Web Browser]
    end

    subgraph LoadBalancer
        D[NGINX/Reverse Proxy]
    end

    subgraph Application
        E[FastAPI Server 1]
        F[FastAPI Server 2]
        G[FastAPI Server N]
    end

    subgraph Services
        H[AstraDB Cluster]
        I[OpenAI API]
        J[Tavily API]
        K[Other LLM APIs]
    end

    subgraph Monitoring
        L[Health Checks]
        M[Structured Logging]
        N[Performance Metrics]
        O[Error Tracking]
    end

    Client --> LoadBalancer
    LoadBalancer --> Application
    Application --> Services
    Application --> Monitoring
```

## Security Architecture

```mermaid
graph TD
    subgraph Authentication
        A[API Key Management]
        B[Environment Variables]
        C[Service Authentication]
    end

    subgraph Authorization
        D[Role-Based Access]
        E[Agent Permissions]
        F[Tool Access Control]
    end

    subgraph DataSecurity
        G[Vector Encryption]
        H[State Encryption]
        I[Secure Transmission]
    end

    subgraph Monitoring
        J[Access Logging]
        K[Error Tracking]
        L[Security Alerts]
    end

    Authentication --> Authorization
    Authorization --> DataSecurity
    DataSecurity --> Monitoring
```

## Performance Characteristics

### Scalability Metrics
- **Concurrent Agents**: 10 specialized agents
- **Tool Execution**: Parallel tool binding and execution
- **State Management**: Efficient persistence with checkpointing
- **Vector Search**: Optimized semantic similarity queries
- **Content Processing**: Automatic truncation for LLM efficiency

### Response Times
- **Agent Query**: < 2 seconds for simple queries
- **Multi-Agent Coordination**: 5-15 seconds for complex tasks
- **Vector Search**: < 1 second for semantic queries
- **Code Generation**: 2-5 seconds for template-based generation
- **Tool Execution**: Variable based on tool complexity

### Resource Usage
- **Memory**: Efficient state management with minimal overhead
- **CPU**: Optimized for concurrent agent execution
- **Network**: Batched API calls and content optimization
- **Storage**: Vector embeddings with efficient indexing

## Error Handling & Recovery

```mermaid
graph TD
    subgraph ErrorDetection
        A[Tool Execution Errors]
        B[LangGraph Invoke Errors]
        C[State Management Errors]
        D[External API Errors]
    end

    subgraph ErrorHandling
        E[Structured Error Responses]
        F[Fallback Mechanisms]
        G[Retry Logic]
        H[Circuit Breakers]
    end

    subgraph Recovery
        I[State Rollback]
        J[Workflow Resumption]
        K[Alternative Tool Selection]
        L[Human Intervention]
    end

    ErrorDetection --> ErrorHandling
    ErrorHandling --> Recovery
```

## Current Implementation Status

### ✅ Fully Implemented
- **Enhanced Tools System**: 6 tools with LangGraph integration
- **Multi-Agent System**: 10 specialized agents
- **State Management**: Full persistence and tracking
- **FastAPI Server**: Modern lifespan handlers
- **AstraDB Integration**: 11 collections with semantic search
- **Tool Binding**: Category-based assignment
- **Content Optimization**: Automatic truncation
- **Error Handling**: Structured responses

### ⚠️ Partially Implemented
- **LangGraph Workflows**: Implemented with invoke method compatibility issue
- **Streaming Responses**: Basic implementation, needs LangGraph fix
- **Checkpointing**: MemorySaver integration, depends on LangGraph

### 🎯 Future Enhancements
- **Advanced Streaming**: Real-time workflow updates
- **Enhanced Error Recovery**: Circuit breakers and retry mechanisms
- **Performance Optimization**: Caching and query optimization
- **Advanced Analytics**: Detailed performance metrics
- **Horizontal Scaling**: Multi-instance deployment

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary development language
- **FastAPI**: Modern async web framework
- **LangChain/LangGraph**: AI framework and workflow orchestration
- **Pydantic**: Data validation and serialization
- **AstraDB**: Vector database with DataStax

### AI/ML Technologies
- **OpenAI**: GPT models and embeddings
- **LangChain**: AI application framework
- **LangGraph**: Workflow orchestration
- **Vector Embeddings**: Semantic search capabilities

### Infrastructure
- **Uvicorn**: ASGI server
- **Async/Await**: Concurrent execution
- **Environment Variables**: Configuration management
- **Structured Logging**: JSON-based logging

This architecture provides a robust, scalable foundation for multi-agent AI workflows with proper tool binding, state management, and error recovery mechanisms. 