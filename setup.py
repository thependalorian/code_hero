from setuptools import setup, find_packages

setup(
    name="code_hero",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi>=0.115.9",
        "pydantic>=2.11.4",
        "typer>=0.9.0",
        "rich>=13.7.0",
        "uvicorn>=0.27.1",
        "python-dotenv>=1.0.0",
        "httpx>=0.27.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.10",
        "langgraph>=0.0.15",
        "aiohttp>=3.9.0",
        "aiosqlite>=0.19.0",
        "cassio>=0.1.3",
        "openai>=1.12.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "code-hero=code_hero.main:cli",
        ],
    },
) 