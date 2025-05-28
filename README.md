# Code Hero 🦸‍♂️

A production-ready **hierarchical multi-agent AI system** built with LangGraph that provides specialized assistance across 19 expert domains with **industry-standard prompts**, **dynamic tool binding**, and **comprehensive infrastructure integration**.

## 🎯 Latest Major Updates

### ✅ **Devstral Model Integration** (Just Implemented)
- **Added Mistral's Devstral 24B** - Specialized coding model for Python development
- **Integrated as secondary model** for coding agents in backend configuration
- **46.8% SWE-Bench Verified score** - Multi-file editing and codebase exploration
- **Available via API, HuggingFace, and Ollama** - Flexible deployment options

### ✅ **Code Quality & Build System** (Just Implemented)
- **Black formatting** applied to all Python files (88 char line length)
- **isort import organization** with Black-compatible profile
- **Frontend build fixes** - All TypeScript errors resolved
- **Successful production build** - Ready for deployment

### ✅ **File Structure Consolidation** (Just Implemented)
- **Moved frontend utilities** from backend to proper frontend location
- **Enhanced document management** - Comprehensive FileUpload component
- **API client consolidation** - Full document lifecycle support
- **Cleaned up duplicates** - Removed temporary and redundant files

### ✅ **Industry-Standard Prompt System** (Previously Implemented)
- **Replaced all hardcoded prompts** with dynamic, context-aware system
- **Integrated with existing prompt infrastructure** (`prompts.py`, `config.py`)
- **Scalable for any request type** - no more limitations
- **Context-aware prompt building** using `build_enhanced_prompt()`

### ✅ **Enhanced Tool Binding** (Previously Implemented)
- **Dynamic tool binding** based on agent roles and context
- **11 tools available** including web search, document search, code generation
- **Category-based assignment** (development, research, documentation)
- **Fallback to all tools** when category-specific binding fails

### ✅ **All Critical Issues Fixed** (Just Implemented)
- **AgentRole enum references** - all corrected
- **Tool binding system** - fully functional
- **Error handling** - comprehensive with fallbacks
- **LLM integration** - primary + fallback models
- **Syntax validation** - all files pass compilation

## 🚀 Quick Start

### Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the backend server
python -m uvicorn src.code_hero.main:app --reload --port 8000
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 🏗️ Architecture

### Hierarchical Multi-Agent System
- **19 Specialized Agents**: Each expert in specific domains
- **Hierarchical Team Structure**: Development, Research, and Documentation teams
- **LLM-Based Routing**: Intelligent task routing with structured output
- **Comprehensive Infrastructure**: Full integration with all Code Hero components
- **Template-Based Prompts**: Zero hardcoded responses
- **Model-Aware Selection**: OpenAI, DeepSeek, Groq support

### Hierarchical Team Structure

#### **Development Team**
- **FastAPI Expert** - Backend development and API design
- **Next.js Expert** - Frontend development and React components  
- **Code Expert** - General programming and implementation

#### **Research Team**
- **Research Expert** - Information gathering and analysis
- **Strategic Expert** - Strategic planning and architecture
- **Analysis Expert** - Data analysis and insights

#### **Documentation Team**
- **Documentation Expert** - Technical writing and guides
- **Implementation Expert** - Implementation documentation and tutorials

#### **Main Orchestrator**
- **Supervisor** - Routes between teams and manages workflow

### Agent Domains
1. **SUPERVISOR** - Multi-agent coordination and team routing
2. **RESEARCH** - Research and analysis  
3. **IMPLEMENTATION** - Software development
4. **DOCUMENTATION** - Technical documentation
5. **TRD_CONVERTER** - Requirements conversion
6. **CODE_GENERATOR** - Code generation
7. **CODE_REVIEWER** - Code quality assurance
8. **STANDARDS_ENFORCER** - Compliance enforcement
9. **STRATEGIC_EXPERT** - Strategic planning
10. **LANGCHAIN_EXPERT** - LangChain development
11. **LANGGRAPH_EXPERT** - LangGraph workflows
12. **LLAMAINDEX_EXPERT** - RAG systems
13. **FASTAPI_EXPERT** - Backend development
14. **NEXTJS_EXPERT** - Frontend development
15. **PYDANTIC_EXPERT** - Data validation
16. **AGNO_EXPERT** - Agno framework
17. **CREWAI_EXPERT** - Multi-agent systems
18. **DOCUMENT_ANALYZER** - Document processing
19. **PROMPT_ENGINEER** - Prompt optimization

## 🎨 Code Quality

### Formatting & Standards
- **Black**: Python code formatting (88 char line length) - ✅ Applied to all files
- **isort**: Import organization with Black-compatible profile - ✅ Applied
- **TypeScript**: Frontend build errors resolved - ✅ Production ready
- **Pre-commit hooks**: Automatic formatting on commits
- **Flake8**: Code linting (230 minor issues remaining)

### Quick Format
```bash
# Run formatting script
./format_code.sh

# Or manually
black src/ tests/ --line-length 88 --target-version py39
isort src/ tests/ --profile black
```

### Build Commands
```bash
# Backend build
python setup.py build

# Frontend build (production ready)
npm run build

# Full build verification
npm run build && python setup.py build
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Quick test
python -m pytest tests/ -q

# Test coverage
python -m pytest tests/ --cov=src/code_hero
```

## 📊 System Status

- ✅ **Backend**: Running on port 8000
- ✅ **Frontend**: Running on port 3000 (Next.js 15) - Production build successful
- ✅ **LangGraph**: Running on port 2024 with hierarchical system loaded
- ✅ **Tests**: 20/20 passing
- ✅ **Agents**: 19 experts operational with industry-standard prompts
- ✅ **Hierarchical System**: LLM-based routing with team structure
- ✅ **Tool Binding**: Dynamic binding with 11 tools available
- ✅ **Infrastructure**: Comprehensive integration with all components
- ✅ **Code Quality**: Black + isort formatted (88 char line length)
- ✅ **Build System**: Frontend TypeScript errors resolved, production ready
- ✅ **Devstral Integration**: Mistral's coding model available for Python tasks
- ✅ **File Structure**: Consolidated frontend/backend separation
- ✅ **Document Management**: Enhanced upload/processing capabilities
- ✅ **AgentRole Enums**: All references corrected and validated
- ✅ **Error Handling**: Comprehensive fallbacks and recovery

## 🔧 Configuration

### Environment Variables
```bash
# Required API Keys
OPENAI_API_KEY=your_openai_key
MISTRAL_API_KEY=your_mistral_key    # For Devstral model access
DEEPSEEK_API_KEY=your_deepseek_key  # Optional
GROQ_API_KEY=your_groq_key          # Optional

# Database (Optional)
ASTRA_DB_TOKEN=your_astra_token
ASTRA_DB_ENDPOINT=your_astra_endpoint

# LangSmith Tracing (Optional)
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=code-hero
```

### Model Configuration
The system intelligently selects models based on task requirements:
- **OpenAI GPT-4**: Complex reasoning and analysis
- **Mistral Devstral 24B**: Specialized Python coding and software engineering
- **DeepSeek**: Cost-effective coding tasks
- **Groq**: Fast inference for simple tasks

#### Devstral Integration
- **Specialized for coding**: 46.8% score on SWE-Bench Verified
- **Multi-file editing**: Advanced codebase exploration capabilities
- **Python optimization**: Specifically tuned for Python development
- **Secondary model**: Available as alternative for coding agents

## 🏛️ Infrastructure Components

### Core Infrastructure
- **State Management**: Comprehensive state tracking and persistence
- **Context Management**: Cross-agent context sharing
- **Agent Manager**: Task tracking and completion recording
- **Strategic Agent**: Strategic guidance and planning
- **Service Validation**: Health monitoring and validation
- **Logger**: Structured logging with state tracking
- **Utils**: Tool execution, ID generation, file handling

### Advanced Features
- **Human-in-the-Loop**: Interactive workflow management
- **Workflow Runner**: Advanced workflow execution
- **Performance Metrics**: Comprehensive monitoring
- **Error Handling**: Graceful fallbacks and recovery
- **Service Health**: Real-time component monitoring

## 📁 Project Structure

```
code-hero/
├── src/code_hero/              # Backend source code
│   ├── hierarchical_agents.py  # Hierarchical multi-agent system
│   ├── agent_expert.py         # Expert agent implementations
│   ├── supervisor.py           # LangGraph supervisor
│   ├── state.py               # Agent state management
│   ├── tools.py               # Agent tools
│   ├── prompts.py             # Prompt templates
│   ├── config.py              # Configuration management (includes Devstral)
│   ├── context.py             # Context management
│   ├── logger.py              # Structured logging
│   ├── manager.py             # State management
│   ├── services.py            # Service management
│   ├── utils.py               # Utility functions
│   ├── interfaces.py          # Service interfaces
│   ├── types.py               # Type definitions
│   ├── agents_api.py          # Agent API endpoints
│   ├── documents_api.py       # Document processing endpoints
│   └── main.py                # FastAPI application
├── frontend/                  # Next.js frontend (consolidated)
│   ├── src/app/              # App router pages
│   ├── src/components/       # React components
│   │   ├── documents/        # Document management components
│   │   │   └── FileUpload.tsx # Enhanced file upload with TRD conversion
│   │   ├── agents/           # Agent interface components
│   │   └── ui/               # Reusable UI components
│   ├── src/hooks/            # React hooks (moved from backend)
│   │   └── useDocuments.ts   # Document management hook
│   ├── src/utils/            # Frontend utilities (consolidated)
│   │   └── api.ts            # API client with full document support
│   └── public/               # Static assets
├── tests/                    # Test suite
├── docs/                     # Documentation
├── .pre-commit-config.yaml   # Code formatting hooks
├── format_code.sh            # Formatting script
├── setup.py                  # Python package configuration
└── package.json              # Node.js dependencies and scripts
```

## 🚀 Deployment

### Vercel (Recommended)
```bash
# Deploy frontend
cd frontend
vercel deploy

# Deploy backend (serverless functions)
vercel deploy --prod
```

### Docker
```bash
# Build and run
docker-compose up --build
```

## 📚 Documentation

- **System Status**: `SYSTEM_STATUS.md`
- **Architecture**: `ARCHITECTURE.md`
- **API Documentation**: `API_DOCUMENTATION.md`
- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **Quick Start**: `QUICK_START.md`
- **Frontend Integration**: `frontend/FRONTEND_INTEGRATION.md`
- **API Documentation**: Available at `/docs` when backend is running

## 🤖 Hierarchical Agent Usage

### Simple Requests
```bash
# Greeting - handled directly
curl -X POST "http://localhost:8000/api/chat/?message=hello"

# Development request - routed to Development Team
curl -X POST "http://localhost:8000/api/chat/?message=create%20a%20FastAPI%20endpoint"

# Research request - routed to Research Team  
curl -X POST "http://localhost:8000/api/chat/?message=research%20best%20practices%20for%20API%20design"

# Documentation request - routed to Documentation Team
curl -X POST "http://localhost:8000/api/chat/?message=write%20documentation%20for%20this%20API"
```

### Infrastructure Status
```bash
# Check infrastructure health
python -c "
import asyncio
from src.code_hero.hierarchical_agents import validate_full_infrastructure
async def check():
    status = await validate_full_infrastructure()
    print(f'Status: {status[\"overall_status\"]}')
    print(f'Healthy: {status[\"healthy_components\"]}')
asyncio.run(check())
"
```

## 🤝 Contributing

1. **Code Formatting**: Run `./format_code.sh` before committing
2. **Tests**: Ensure all tests pass with `python -m pytest tests/`
3. **Documentation**: Update relevant docs for new features
4. **Pre-commit**: Install hooks with `pre-commit install`

## 📄 License

MIT License - see LICENSE file for details.

---

**Status**: 🟢 **FULLY OPERATIONAL WITH HIERARCHICAL AGENTS**  
**Last Updated**: January 2025 - Devstral Integration & Build System  
**Version**: 3.1 - Enhanced Multi-Agent System with Specialized Coding Models