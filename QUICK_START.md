# Code Hero Quick Start Guide 🚀

Get up and running with Code Hero's 22-agent system in minutes.

## Prerequisites ✅

- Python 3.8+
- Node.js 18+
- At least one LLM API key (OpenAI, DeepSeek, or Groq)

## 5-Minute Setup

### 1. Clone & Setup Environment
```bash
git clone <repository-url>
cd code-hero

# Create Python environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Create environment file
cp .env.example .env

# Edit .env with your API keys (at minimum one of these):
OPENAI_API_KEY=your_openai_key
DEEPSEEK_API_KEY=your_deepseek_key  
GROQ_API_KEY=your_groq_key
```

### 3. Start the System
```bash
# Terminal 1: Start backend
python -m src.code_hero.main

# Terminal 2: Start frontend
cd frontend
npm install
npm run dev
```

### 4. Verify Installation
```bash
# Check system health
curl http://localhost:8000/health

# Test agent response
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, test the system"}'
```

## Access Points 🌐

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Available Agents 🤖

### Core Agents (19)
| Agent | Purpose | Example Use |
|-------|---------|-------------|
| SUPERVISOR | Task routing | "Route this to the best agent" |
| RESEARCH | Information gathering | "Research React best practices" |
| IMPLEMENTATION | Code development | "Build a user authentication system" |
| DOCUMENTATION | Technical docs | "Document this API endpoint" |
| CODE_GENERATOR | Advanced coding | "Generate a complex algorithm" |
| CODE_REVIEWER | Quality assurance | "Review this code for issues" |
| FASTAPI_EXPERT | Backend APIs | "Create FastAPI endpoints" |
| NEXTJS_EXPERT | Frontend development | "Build a Next.js component" |
| LANGCHAIN_EXPERT | LangChain workflows | "Create a RAG system" |
| LANGGRAPH_EXPERT | Multi-agent systems | "Design agent workflows" |

### Specialized Experts (3)
| Expert | Specialization | Best For |
|--------|----------------|----------|
| QwenExpert | Advanced coding | Complex algorithms, multi-language |
| DeepSeekExpert | Cost-effective development | Open-source projects, research |
| CodingOptimizationExpert | Model selection | Choosing optimal models for tasks |

## Quick Examples 💡

### Chat with Specific Agent
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a FastAPI endpoint for user registration",
    "agent_type": "fastapi_expert"
  }'
```

### Research Query
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find the latest trends in AI development",
    "agent_type": "research"
  }'
```

### Code Generation
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Generate a Python class for handling JWT tokens",
    "agent_type": "code_generator"
  }'
```

## Troubleshooting 🔧

### Common Issues

**Backend won't start:**
```bash
# Check Python version
python --version  # Should be 3.8+

# Verify dependencies
pip install -r requirements.txt

# Check API keys
cat .env | grep API_KEY
```

**Frontend won't start:**
```bash
# Check Node version
node --version  # Should be 18+

# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Agents not responding:**
```bash
# Test system health
python test_hardcoded_responses.py

# Check logs
tail -f logs/code_hero.log
```

### Verify System Status
```bash
# Quick system test
python -c "
import sys
sys.path.append('src')
from code_hero.agent_expert import all_experts
print(f'✅ {len(all_experts)} agents loaded')
print('✅ System ready!')
"
```

## Next Steps 🎯

1. **Explore the UI**: Visit http://localhost:3000
2. **Try Different Agents**: Test various agent types
3. **Check API Docs**: Visit http://localhost:8000/docs
4. **Read Architecture**: See ARCHITECTURE.md for details
5. **Deploy**: Follow DEPLOYMENT_GUIDE.md for production

## Need Help? 🆘

- **System Health**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs
- **Test Suite**: `python test_hardcoded_responses.py`
- **Issues**: GitHub Issues page

---

**You're ready to go! 🎉** 