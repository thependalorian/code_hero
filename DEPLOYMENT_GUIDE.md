# Code Hero Deployment Guide

## 🚀 Overview

This guide covers the complete deployment process for the Code Hero AI-powered development platform, including both the FastAPI backend and Next.js frontend.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Database      │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   (AstraDB)     │
│   Vercel        │    │   Railway/      │    │   DataStax      │
│                 │    │   Render/Heroku │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

### Required Accounts
- [Vercel](https://vercel.com) - Frontend deployment
- [Railway](https://railway.app) / [Render](https://render.com) / [Heroku](https://heroku.com) - Backend deployment
- [DataStax Astra](https://astra.datastax.com) - Vector database
- [OpenAI](https://platform.openai.com) - AI models
- [GitHub](https://github.com) - Code repository

### Required Tools
- Node.js 18+
- Python 3.9+
- Git
- Docker (optional)

## 🗄️ Database Setup (AstraDB)

### 1. Create AstraDB Database

1. **Sign up** at [astra.datastax.com](https://astra.datastax.com)
2. **Create a new database**:
   - Database name: `code-hero-db`
   - Keyspace: `code_hero`
   - Cloud provider: AWS/GCP/Azure
   - Region: Choose closest to your users

3. **Get connection details**:
   - Database ID
   - Region
   - Application Token

### 2. Create Collections

```python
# Run this script to create required collections
import os
from astrapy import DataAPIClient

# Initialize client
client = DataAPIClient(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))
database = client.get_database_by_api_endpoint(
    f"https://{os.getenv('ASTRA_DB_ID')}-{os.getenv('ASTRA_DB_REGION')}.apps.astra.datastax.com"
)

# Create collections
collections = [
    "strategy_book",
    "langchain_docs", 
    "fastapi_docs",
    "nextjs_docs",
    "pydantic_docs",
    "python_guides",
    "react_guides",
    "business_docs",
    "llm_guides",
    "chat_history",
    "project_data"
]

for collection_name in collections:
    try:
        collection = database.create_collection(
            collection_name,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine"
        )
        print(f"Created collection: {collection_name}")
    except Exception as e:
        print(f"Collection {collection_name} might already exist: {e}")
```

## 🔧 Backend Deployment

### Option 1: Railway Deployment

1. **Connect Repository**:
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login and connect
   railway login
   railway link
   ```

2. **Configure Environment Variables**:
   ```bash
   # Set environment variables
   railway variables set OPENAI_API_KEY=your_openai_key
   railway variables set ASTRA_DB_ID=your_db_id
   railway variables set ASTRA_DB_REGION=your_region
   railway variables set ASTRA_DB_APPLICATION_TOKEN=your_token
   railway variables set PORT=8000
   railway variables set HOST=0.0.0.0
   ```

3. **Create Railway Configuration**:
   ```toml
   # railway.toml
   [build]
   builder = "NIXPACKS"
   buildCommand = "pip install -r requirements.txt"
   
   [deploy]
   startCommand = "python -m code_hero run-server --host 0.0.0.0 --port $PORT"
   restartPolicyType = "ON_FAILURE"
   restartPolicyMaxRetries = 10
   
   [env]
   PORT = "8000"
   HOST = "0.0.0.0"
   ```

4. **Deploy**:
   ```bash
   railway up
   ```

### Option 2: Render Deployment

1. **Create `render.yaml`**:
   ```yaml
   services:
     - type: web
       name: code-hero-api
       env: python
       buildCommand: "pip install -r requirements.txt"
       startCommand: "python -m code_hero run-server --host 0.0.0.0 --port $PORT"
       envVars:
         - key: OPENAI_API_KEY
           sync: false
         - key: ASTRA_DB_ID
           sync: false
         - key: ASTRA_DB_REGION
           sync: false
         - key: ASTRA_DB_APPLICATION_TOKEN
           sync: false
         - key: PORT
           value: 10000
         - key: HOST
           value: 0.0.0.0
   ```

2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Select "Web Service"
   - Configure environment variables

### Option 3: Heroku Deployment

1. **Create Heroku App**:
   ```bash
   # Install Heroku CLI
   heroku create code-hero-api
   
   # Set environment variables
   heroku config:set OPENAI_API_KEY=your_key
   heroku config:set ASTRA_DB_ID=your_db_id
   heroku config:set ASTRA_DB_REGION=your_region
   heroku config:set ASTRA_DB_APPLICATION_TOKEN=your_token
   ```

2. **Create `Procfile`**:
   ```
   web: python -m code_hero run-server --host 0.0.0.0 --port $PORT
   ```

3. **Deploy**:
   ```bash
   git push heroku main
   ```

### Option 4: Docker Deployment

1. **Create `Dockerfile`**:
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       curl \
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application code
   COPY . .
   
   # Install the package
   RUN pip install -e .
   
   # Expose port
   EXPOSE 8000
   
   # Health check
   HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
     CMD curl -f http://localhost:8000/health || exit 1
   
   # Run the application
   CMD ["python", "-m", "code_hero", "run-server", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Create `docker-compose.yml`**:
   ```yaml
   version: '3.8'
   
   services:
     api:
       build: .
       ports:
         - "8000:8000"
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - ASTRA_DB_ID=${ASTRA_DB_ID}
         - ASTRA_DB_REGION=${ASTRA_DB_REGION}
         - ASTRA_DB_APPLICATION_TOKEN=${ASTRA_DB_APPLICATION_TOKEN}
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
   ```

3. **Deploy**:
   ```bash
   docker-compose up -d
   ```

## 🌐 Frontend Deployment

### Vercel Deployment (Recommended)

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Configure Environment Variables**:
   ```bash
   # .env.local (for local development)
   NEXT_PUBLIC_API_URL=https://your-api-domain.com
   NEXT_PUBLIC_WS_URL=wss://your-api-domain.com/ws
   ```

3. **Create `vercel.json`**:
   ```json
   {
     "framework": "nextjs",
     "buildCommand": "npm run build",
     "devCommand": "npm run dev",
     "installCommand": "npm install",
     "env": {
       "NEXT_PUBLIC_API_URL": "@api-url",
       "NEXT_PUBLIC_WS_URL": "@ws-url"
     },
     "build": {
       "env": {
         "NEXT_PUBLIC_API_URL": "@api-url",
         "NEXT_PUBLIC_WS_URL": "@ws-url"
       }
     }
   }
   ```

4. **Deploy**:
   ```bash
   cd frontend
   vercel --prod
   ```

5. **Set Environment Variables in Vercel Dashboard**:
   - Go to your project settings
   - Add environment variables:
     - `NEXT_PUBLIC_API_URL`: Your backend URL
     - `NEXT_PUBLIC_WS_URL`: Your WebSocket URL

### Alternative: Netlify Deployment

1. **Create `netlify.toml`**:
   ```toml
   [build]
     command = "npm run build"
     publish = ".next"
   
   [build.environment]
     NEXT_PUBLIC_API_URL = "https://your-api-domain.com"
     NEXT_PUBLIC_WS_URL = "wss://your-api-domain.com/ws"
   
   [[redirects]]
     from = "/api/*"
     to = "https://your-api-domain.com/api/:splat"
     status = 200
   ```

2. **Deploy**:
   - Connect your repository to Netlify
   - Configure build settings
   - Set environment variables

## 🔐 Environment Variables

### Backend Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
ASTRA_DB_ID=your-database-id
ASTRA_DB_REGION=us-east-1
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...

# Optional
DEEPSEEK_API_KEY=your-deepseek-key
GROQ_API_KEY=your-groq-key
TAVILY_API_KEY=your-tavily-key
LANGCHAIN_API_KEY=your-langchain-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=code-hero

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=INFO
```

### Frontend Environment Variables

```bash
# Required
NEXT_PUBLIC_API_URL=https://your-backend-domain.com
NEXT_PUBLIC_WS_URL=wss://your-backend-domain.com/ws

# Optional
NEXT_PUBLIC_ANALYTICS_ID=your-analytics-id
NEXT_PUBLIC_SENTRY_DSN=your-sentry-dsn
```

## 🔍 Health Checks & Monitoring

### Backend Health Checks

```python
# Add to your deployment configuration
HEALTHCHECK_ENDPOINTS = [
    "/health",
    "/api/astra/health"
]

# Example health check script
import requests
import sys

def check_health():
    try:
        response = requests.get("https://your-api-domain.com/health", timeout=10)
        if response.status_code == 200:
            print("✅ Backend is healthy")
            return True
        else:
            print(f"❌ Backend unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    if not check_health():
        sys.exit(1)
```

### Frontend Health Checks

```typescript
// utils/healthCheck.ts
export async function checkFrontendHealth() {
  try {
    // Check if API is reachable
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error('Frontend health check failed:', error);
    return false;
  }
}
```

## 📊 Monitoring & Logging

### Backend Monitoring

1. **Add Sentry for Error Tracking**:
   ```python
   # Add to requirements.txt
   sentry-sdk[fastapi]==1.40.0
   
   # Add to main.py
   import sentry_sdk
   from sentry_sdk.integrations.fastapi import FastApiIntegration
   
   sentry_sdk.init(
       dsn=os.getenv("SENTRY_DSN"),
       integrations=[FastApiIntegration()],
       traces_sample_rate=1.0,
   )
   ```

2. **Add Structured Logging**:
   ```python
   import structlog
   
   structlog.configure(
       processors=[
           structlog.stdlib.filter_by_level,
           structlog.stdlib.add_logger_name,
           structlog.stdlib.add_log_level,
           structlog.stdlib.PositionalArgumentsFormatter(),
           structlog.processors.TimeStamper(fmt="iso"),
           structlog.processors.StackInfoRenderer(),
           structlog.processors.format_exc_info,
           structlog.processors.UnicodeDecoder(),
           structlog.processors.JSONRenderer()
       ],
       context_class=dict,
       logger_factory=structlog.stdlib.LoggerFactory(),
       wrapper_class=structlog.stdlib.BoundLogger,
       cache_logger_on_first_use=True,
   )
   ```

### Frontend Monitoring

1. **Add Analytics**:
   ```typescript
   // utils/analytics.ts
   export function trackEvent(event: string, properties?: Record<string, any>) {
     if (typeof window !== 'undefined' && window.gtag) {
       window.gtag('event', event, properties);
     }
   }
   ```

2. **Add Error Boundary with Sentry**:
   ```typescript
   import * as Sentry from '@sentry/nextjs';
   
   Sentry.init({
     dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
     tracesSampleRate: 1.0,
   });
   ```

## 🚀 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy Code Hero

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      - name: Run tests
        run: pytest
      - name: Check configuration
        run: python -m code_hero check-config
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ASTRA_DB_ID: ${{ secrets.ASTRA_DB_ID }}
          ASTRA_DB_REGION: ${{ secrets.ASTRA_DB_REGION }}
          ASTRA_DB_APPLICATION_TOKEN: ${{ secrets.ASTRA_DB_APPLICATION_TOKEN }}

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      - name: Install dependencies
        run: |
          cd frontend
          npm ci
      - name: Run tests
        run: |
          cd frontend
          npm test
      - name: Build
        run: |
          cd frontend
          npm run build
        env:
          NEXT_PUBLIC_API_URL: https://api.example.com

  deploy-backend:
    needs: test-backend
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Railway
        run: |
          npm install -g @railway/cli
          railway login --token ${{ secrets.RAILWAY_TOKEN }}
          railway up --service backend
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}

  deploy-frontend:
    needs: test-frontend
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          working-directory: frontend
```

## 🔧 Post-Deployment Configuration

### 1. Domain Configuration

```bash
# Backend (Railway/Render/Heroku)
# Configure custom domain in platform dashboard

# Frontend (Vercel)
vercel domains add yourdomain.com
vercel domains add api.yourdomain.com
```

### 2. SSL/TLS Configuration

Most platforms provide automatic SSL certificates. Ensure:
- HTTPS is enforced
- HTTP redirects to HTTPS
- CORS is configured for your domain

### 3. Database Optimization

```python
# Optimize AstraDB collections
async def optimize_collections():
    collections = await astra_client.list_collections()
    for collection in collections:
        # Add indexes for better performance
        await collection.create_index("metadata.timestamp")
        await collection.create_index("metadata.source")
```

## 🧪 Testing Deployment

### Backend Testing

```bash
# Test health endpoint
curl https://your-api-domain.com/health

# Test chat endpoint
curl -X POST https://your-api-domain.com/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "conversation_id": "test"}'

# Test search endpoint
curl -X POST https://your-api-domain.com/api/astra/search \
  -H "Content-Type: application/json" \
  -d '{"query": "FastAPI", "collection": "fastapi_docs", "limit": 3}'
```

### Frontend Testing

```bash
# Test frontend accessibility
curl https://your-frontend-domain.com

# Test API integration
# Open browser developer tools and check network requests
```

## 🔄 Rollback Strategy

### Backend Rollback

```bash
# Railway
railway rollback

# Heroku
heroku rollback

# Docker
docker-compose down
docker-compose up -d --scale api=0
docker-compose up -d
```

### Frontend Rollback

```bash
# Vercel
vercel rollback

# Manual rollback
git revert HEAD
git push origin main
```

## 📈 Scaling Considerations

### Backend Scaling

1. **Horizontal Scaling**:
   - Use load balancers
   - Deploy multiple instances
   - Implement session affinity if needed

2. **Database Scaling**:
   - Monitor AstraDB usage
   - Optimize queries
   - Consider read replicas

3. **Caching**:
   - Implement Redis for session storage
   - Cache frequent API responses
   - Use CDN for static assets

### Frontend Scaling

1. **CDN Configuration**:
   - Vercel automatically provides CDN
   - Configure cache headers
   - Optimize images and assets

2. **Performance Optimization**:
   - Enable Next.js Image optimization
   - Implement code splitting
   - Use dynamic imports

## 🆘 Troubleshooting

### Common Issues

1. **CORS Errors**:
   ```python
   # Ensure CORS is configured correctly
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://your-frontend-domain.com"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Environment Variable Issues**:
   ```bash
   # Check if variables are set
   echo $OPENAI_API_KEY
   
   # Verify in application
   python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
   ```

3. **Database Connection Issues**:
   ```python
   # Test AstraDB connection
   python -c "
   from src.code_hero.tools import astra_retriever
   import asyncio
   asyncio.run(astra_retriever.list_collections())
   "
   ```

### Monitoring Commands

```bash
# Check backend logs
railway logs --service backend

# Check frontend logs
vercel logs

# Monitor system resources
htop
df -h
```

---

This deployment guide provides a comprehensive approach to deploying the Code Hero system in production. Follow the steps carefully and ensure all environment variables are properly configured for a successful deployment. 