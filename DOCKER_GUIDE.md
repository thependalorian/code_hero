# Docker Guide for Code Hero

This guide covers Docker deployment for the **Code Hero Hierarchical Multi-Agent System**.

## 🐳 Quick Start

### Development Mode
```bash
# Start all services in development mode
docker-compose up --build

# Start in background
docker-compose up -d --build

# View logs
docker-compose logs -f code-hero-api
```

### Production Mode
```bash
# Start with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build

# Start in background
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

## 📁 Docker Files Overview

### Core Files
- **`Dockerfile`** - Multi-stage build for optimized production images
- **`docker-compose.yml`** - Base configuration for all services
- **`docker-compose.override.yml`** - Development overrides (auto-loaded)
- **`docker-compose.prod.yml`** - Production-specific configuration
- **`nginx.conf`** - Nginx reverse proxy configuration
- **`.dockerignore`** - Optimized build context exclusions

### Configuration Files
- **`requirements.txt`** - Python dependencies for Docker builds
- **`setup.py`** - Updated package configuration

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │   Frontend      │    │   Backend API   │
│  (Port 80/443)  │────│   (Port 3000)   │────│   (Port 8000)   │
│  Reverse Proxy  │    │    Next.js      │    │    FastAPI      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Redis       │
                    │   (Port 6379)   │
                    │    Caching      │
                    └─────────────────┘
```

## 🔧 Services

### 1. Code Hero API (Backend)
- **Image**: Custom Python 3.11 slim
- **Port**: 8000
- **Features**: 
  - Hierarchical multi-agent system
  - LLM-based routing
  - 19 expert agents
  - Infrastructure integration

### 2. Code Hero Frontend
- **Image**: Custom Node.js
- **Port**: 3000
- **Features**:
  - Next.js 15 application
  - Real-time agent monitoring
  - Interactive chat interface

### 3. Nginx (Production)
- **Image**: nginx:alpine
- **Ports**: 80, 443
- **Features**:
  - Reverse proxy
  - Load balancing
  - SSL termination
  - Rate limiting
  - Static file caching

### 4. Redis (Optional)
- **Image**: redis:7-alpine
- **Port**: 6379
- **Features**:
  - Session management
  - Caching layer
  - Data persistence

## 🌍 Environment Variables

### Required Variables
```bash
# OpenAI API
OPENAI_API_KEY=your_openai_key

# AstraDB Configuration
ASTRA_DB_ID=your_astra_db_id
ASTRA_DB_REGION=your_region
ASTRA_DB_APPLICATION_TOKEN=your_token
```

### Optional Variables
```bash
# Additional LLM Providers
DEEPSEEK_API_KEY=your_deepseek_key
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key

# LangSmith Tracing
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=code-hero

# Hierarchical System Configuration
HIERARCHICAL_ROUTING_ENABLED=true
HIERARCHICAL_DEBUG_MODE=false
HIERARCHICAL_RECURSION_LIMIT=10
```

### Environment File Setup
```bash
# Create .env file
cp .env.example .env

# Edit with your values
nano .env
```

## 🚀 Deployment Commands

### Development
```bash
# Start development environment
docker-compose up --build

# Rebuild specific service
docker-compose build code-hero-api
docker-compose up code-hero-api

# View service logs
docker-compose logs -f code-hero-api
docker-compose logs -f code-hero-frontend

# Execute commands in container
docker-compose exec code-hero-api bash
docker-compose exec code-hero-api python -m code_hero list-agents
```

### Production
```bash
# Deploy to production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Scale services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale code-hero-api=3

# Update single service
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-deps code-hero-api

# Check service health
docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

### Maintenance
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Clean up unused images
docker system prune -a

# View resource usage
docker stats
```

## 📊 Monitoring

### Health Checks
```bash
# Check API health
curl http://localhost:8000/health

# Check frontend
curl http://localhost:3000

# Check through nginx
curl http://localhost/health
```

### Logs
```bash
# View all logs
docker-compose logs

# Follow specific service
docker-compose logs -f code-hero-api

# View last 100 lines
docker-compose logs --tail=100 code-hero-api

# Filter by timestamp
docker-compose logs --since="2024-01-01T00:00:00" code-hero-api
```

### Performance Monitoring
```bash
# Container resource usage
docker stats

# Service-specific stats
docker stats code-hero-api code-hero-frontend

# Disk usage
docker system df
```

## 🔒 Security

### Production Security Features
- **Non-root user**: All containers run as non-root
- **Security headers**: Nginx adds security headers
- **Rate limiting**: API and frontend rate limits
- **Network isolation**: Services communicate via internal network
- **Resource limits**: CPU and memory constraints

### SSL Configuration
```bash
# Generate self-signed certificates (development)
mkdir ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem

# Update nginx.conf to enable HTTPS
# Uncomment HTTPS server block in nginx.conf
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Check port usage
lsof -i :8000
lsof -i :3000

# Use different ports
docker-compose up --build -p 8001:8000
```

#### 2. Environment Variables
```bash
# Check environment in container
docker-compose exec code-hero-api env | grep OPENAI

# Debug configuration
docker-compose exec code-hero-api python -c "
import os
print('OPENAI_API_KEY:', bool(os.getenv('OPENAI_API_KEY')))
"
```

#### 3. Build Issues
```bash
# Clean build
docker-compose build --no-cache code-hero-api

# Check build logs
docker-compose build code-hero-api 2>&1 | tee build.log
```

#### 4. Network Issues
```bash
# Check network
docker network ls
docker network inspect code-hero_code-hero-network

# Test connectivity
docker-compose exec code-hero-frontend ping code-hero-api
```

### Debug Mode
```bash
# Enable debug mode
export HIERARCHICAL_DEBUG_MODE=true
docker-compose up --build

# View detailed logs
docker-compose logs -f code-hero-api | grep DEBUG
```

## 📈 Performance Optimization

### Production Optimizations
- **Multi-stage builds**: Smaller production images
- **Layer caching**: Optimized Dockerfile layer order
- **Resource limits**: Prevent resource exhaustion
- **Health checks**: Automatic service recovery
- **Nginx caching**: Static file caching and compression

### Scaling
```bash
# Scale API service
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale code-hero-api=3

# Load balancing handled by nginx upstream
```

## 🔄 Updates and Maintenance

### Updating Services
```bash
# Pull latest images
docker-compose pull

# Rebuild and restart
docker-compose up --build -d

# Rolling update (zero downtime)
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-deps code-hero-api
```

### Backup and Restore
```bash
# Backup volumes
docker run --rm -v code-hero_redis-data:/data -v $(pwd):/backup alpine tar czf /backup/redis-backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v code-hero_redis-data:/data -v $(pwd):/backup alpine tar xzf /backup/redis-backup.tar.gz -C /data
```

## 📚 Additional Resources

- **Docker Compose Documentation**: https://docs.docker.com/compose/
- **Nginx Configuration**: https://nginx.org/en/docs/
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/
- **Next.js Docker**: https://nextjs.org/docs/deployment

---

**Status**: ✅ **Docker Configuration Complete**  
**Last Updated**: January 2025  
**Version**: 3.0 - Hierarchical Multi-Agent System 