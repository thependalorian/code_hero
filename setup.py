from setuptools import setup, find_packages

setup(
    name="code_hero",
    version="3.0.0",
    description="Hierarchical Multi-Agent AI Development Assistant",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        # Core FastAPI and web framework
        "fastapi==0.115.9",
        "uvicorn[standard]>=0.27.1",
        "pydantic>=2.11.4",
        "python-dotenv>=1.0.0",
        "httpx>=0.27.0",
        
        # CLI and UI
        "typer>=0.9.0",
        "rich>=13.7.0",
        
        # LangChain and AI framework
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
        "langchain-openai>=0.2.0",
        "langchain-community>=0.3.0",
        "langchain-groq>=0.1.0",
        "langgraph>=0.2.0",
        "langsmith>=0.1.0",
        
        # OpenAI and LLM providers
        "openai>=1.12.0",
        "groq>=0.4.0",
        
        # Web search and tools
        "tavily-python>=0.3.0",
        
        # Database and storage
        "astrapy>=1.0.0",
        "cassio>=0.1.3",
        "aiosqlite>=0.19.0",
        
        # Async and networking
        "aiohttp>=3.9.0",
        "aiofiles>=23.0.0",
        "requests>=2.31.0",
        "websockets>=11.0.0",
        
        # Document processing and search
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "pypdf>=3.0.0",
        "python-multipart>=0.0.6",
        
        # Utilities
        "tenacity>=8.2.0",
        "tiktoken>=0.5.0",
        "numpy>=1.26.0,<2.0.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0.0",
        "jinja2>=3.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "deployment": [
            "gunicorn>=21.0.0",
            "docker>=6.1.0",
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "structlog>=23.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "code-hero=code_hero.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="ai, agents, langchain, langgraph, fastapi, development, assistant",
    project_urls={
        "Homepage": "https://github.com/pendalorian/code-hero",
        "Bug Reports": "https://github.com/pendalorian/code-hero/issues",
        "Source": "https://github.com/pendalorian/code-hero",
        "Documentation": "https://github.com/pendalorian/code-hero/docs",
    },
) 