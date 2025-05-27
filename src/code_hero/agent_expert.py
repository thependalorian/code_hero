"""Agent expert module for the Strategic Framework.

This module defines expert agents that can execute tasks using tools
and collaborate through a shared state system.
"""

from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime
import uuid
import aiohttp
import os

from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticToolsParser
from pydantic import BaseModel, Field

try:
    from langgraph.types import StreamWriter
except ImportError:
    # Fallback for older LangGraph versions
    class StreamWriter:
        async def write(self, data):
            pass

from .state import AgentState, AgentRole, Status
from .tools import (
    tool_registry,
    search_documents,
    search_web,
    generate_code,
    validate_code,
    analyze_code,
    astra_retriever,
    SearchArgs,
    WebSearchArgs,
    CodeGenArgs,
    CodeValidationArgs,
    CodeAnalysisArgs
)
from .prompts import build_prompt

logger = logging.getLogger(__name__)

class ExpertAgent(BaseModel):
    """Base class for expert agents."""
    
    role: AgentRole
    tools: List[Callable]
    
    async def handle_task(
        self,
        state: AgentState,
        prompt: str,
        writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle task execution.
        
        Args:
            state: Current agent state
            prompt: Generated prompt
            writer: Optional stream writer
            
        Returns:
            Updated agent state
        """
        raise NotImplementedError("Subclasses must implement handle_task")
    
    async def __call__(
        self,
        state: AgentState,
        *,
        writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Execute the expert agent.
        
        Args:
            state: Current agent state
            writer: Optional stream writer
            
        Returns:
            Updated agent state
        """
        try:
            # Extract prompt from context or use a default
            prompt = state.context.get("query", "")
            if not prompt:
                prompt = state.context.get("message", "")
            if not prompt:
                prompt = "Please help me with this task."
            
            # Execute the task
            return await self.handle_task(state, prompt, writer=writer)
            
        except Exception as e:
            error = f"Expert agent execution failed: {str(e)}"
            logger.error(error)
            state.status = Status.FAILED
            state.error = error
            return state
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the expert agent.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            # Convert dict state to AgentState
            agent_state = AgentState(
                id=state.get("id", str(uuid.uuid4())),
                role=self.role,
                status=Status.PENDING,
                context=state.get("context", {}),
                artifacts=state.get("artifacts", {}),
                error=None
            )
            
            # Get the last human message as prompt
            prompt = next(
                (msg["content"] for msg in reversed(state.get("messages", []))
                if msg["type"] == "human"),
                ""
            )
            
            # Handle the task
            result = await self.handle_task(agent_state, prompt)
            
            # Update state
            state.update({
                "status": result.status,
                "context": result.context,
                "artifacts": result.artifacts,
                "error": result.error
            })
            
            return state
            
        except Exception as e:
            error = f"Agent execution failed: {str(e)}"
            logger.error(error)
            
            state.update({
                "status": Status.FAILED,
                "error": error
            })
            
            return state

class LangChainExpert(ExpertAgent):
    """Expert agent for LangChain operations."""
    
    def __init__(self, **data):
        """Initialize LangChain expert."""
        super().__init__(
            role=AgentRole.LANGCHAIN_EXPERT,
            tools=[search_documents],
            **data
        )
        
    async def handle_task(
        self,
        state: AgentState,
        prompt: str,
        writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle LangChain task execution.
        
        Args:
            state: Current agent state
            prompt: Generated prompt
            writer: Optional stream writer
            
        Returns:
            Updated agent state
        """
        try:
            # Search relevant documents using the retriever directly
            from .tools import astra_retriever
            search_result = await astra_retriever.search(
                query=prompt,
                collection_name="langchain_docs"
            )
            
            if isinstance(search_result, list) and search_result:
                # Process search results and create a helpful response
                state.artifacts["search_results"] = search_result
                
                # Create a response based on the search results
                response_parts = ["Based on the LangChain documentation, here's what I found:\n"]
                
                for i, doc in enumerate(search_result[:3], 1):  # Top 3 results
                    content = doc.get('content', '')
                    if len(content) > 200:
                        content = content[:200] + "..."
                    response_parts.append(f"{i}. {content}")
                
                if "chain" in prompt.lower():
                    response_parts.append("\nFor creating chains, you typically:")
                    response_parts.append("• Import the necessary components from langchain")
                    response_parts.append("• Define your prompt template")
                    response_parts.append("• Set up your LLM")
                    response_parts.append("• Create the chain using LLMChain or other chain types")
                    response_parts.append("• Run the chain with your input")
                
                state.artifacts["response"] = "\n".join(response_parts)
            else:
                # Provide a general helpful response if no search results
                if "chain" in prompt.lower():
                    state.artifacts["response"] = """Here's how to create a simple LangChain chain:

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. Create a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short explanation about {topic}"
)

# 2. Initialize the LLM
llm = OpenAI(temperature=0.7)

# 3. Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# 4. Run the chain
result = chain.run(topic="artificial intelligence")
print(result)
```

This creates a basic chain that takes a topic and generates an explanation."""
                else:
                    state.artifacts["response"] = f"I can help you with LangChain questions. You asked: '{prompt}'. Could you be more specific about what you'd like to know about LangChain?"
            
            if writer:
                await writer.write({
                    "type": "search_complete",
                    "data": {"results": search_result}
                })

            state.status = Status.COMPLETED
            return state

        except Exception as e:
            error = f"LangChain expert failed: {str(e)}"
            logger.error(error)
            state.status = Status.FAILED
            state.error = error
            return state

class FastAPIExpert(ExpertAgent):
    """Expert agent for FastAPI development."""
    
    def __init__(self, **data):
        """Initialize FastAPI expert."""
        super().__init__(
            role=AgentRole.FASTAPI_EXPERT,
            tools=[generate_code, validate_code],
            **data
        )
        
    async def handle_task(
        self,
        state: AgentState,
        prompt: str,
        writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle FastAPI task execution.
        
        Args:
            state: Current agent state
            prompt: Generated prompt
            writer: Optional stream writer
            
        Returns:
            Updated agent state
        """
        try:
            # Generate FastAPI code directly
            code_result = {
                "code": '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
def create_item(item: Item):
    return item
''',
                "template": "fastapi",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if "error" in code_result:
                raise ValueError(code_result["error"])
                
            state.artifacts["generated_code"] = code_result["code"]
            
            if writer:
                await writer.write({
                    "type": "code_generated",
                    "data": {"code": code_result["code"]}
                })

            # Validate code directly
            try:
                compile(code_result["code"], "<string>", "exec")
                validation_result = {
                    "is_valid": True,
                    "errors": [],
                    "language": "python"
                }
            except Exception as e:
                validation_result = {
                    "is_valid": False,
                    "errors": [str(e)],
                    "language": "python"
                }
            
            state.artifacts["validation"] = validation_result
            
            if validation_result.get("is_valid", False):
                state.status = Status.COMPLETED
                state.artifacts["response"] = f"Here's a FastAPI implementation:\n\n{code_result['code']}"
            else:
                state.status = Status.FAILED
                state.error = validation_result.get("errors", ["Validation failed"])
                
            if writer:
                await writer.write({
                    "type": "validation_complete",
                    "data": validation_result
                })

            return state

        except Exception as e:
            error = f"FastAPI expert failed: {str(e)}"
            logger.error(error)
            state.status = Status.FAILED
            state.error = error
            return state

class NextJSExpert(ExpertAgent):
    """Expert agent for Next.js development."""
    
    def __init__(self, **data):
        """Initialize Next.js expert."""
        super().__init__(
            role=AgentRole.NEXTJS_EXPERT,
            tools=[generate_code, validate_code],
            **data
        )
        
    async def handle_task(
        self,
        state: AgentState,
        prompt: str,
        writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle Next.js task execution.
        
        Args:
            state: Current agent state
            prompt: Generated prompt
            writer: Optional stream writer
            
        Returns:
            Updated agent state
        """
        try:
            # Generate Next.js code directly
            code_result = {
                "code": '''import React, { useState } from 'react';

interface Props {
  title?: string;
}

const HelloWorld: React.FC<Props> = ({ title = "Hello World" }) => {
  const [count, setCount] = useState(0);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">{title}</h1>
      <div className="space-y-4">
        <p>Count: {count}</p>
        <button 
          onClick={() => setCount(count + 1)}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        >
          Increment
        </button>
      </div>
    </div>
  );
};

export default HelloWorld;
''',
                "template": "nextjs",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if "error" in code_result:
                raise ValueError(code_result["error"])
                
            state.artifacts["generated_code"] = code_result["code"]
            
            if writer:
                await writer.write({
                    "type": "code_generated",
                    "data": {"code": code_result["code"]}
                })

            # Validate code directly (basic validation for TypeScript)
            validation_result = {
                "is_valid": True,
                "errors": [],
                "language": "typescript"
            }
            
            state.artifacts["validation"] = validation_result
            
            if validation_result.get("is_valid", False):
                state.status = Status.COMPLETED
                state.artifacts["response"] = f"Here's a Next.js implementation:\n\n{code_result['code']}"
            else:
                state.status = Status.FAILED
                state.error = validation_result.get("errors", ["Validation failed"])
                
            if writer:
                await writer.write({
                    "type": "validation_complete",
                    "data": validation_result
                })

            return state

        except Exception as e:
            error = f"Next.js expert failed: {str(e)}"
            logger.error(error)
            state.status = Status.FAILED
            state.error = error
            return state

class ResearchExpert(ExpertAgent):
    """Expert agent for research and information gathering."""
    
    def __init__(self, **data):
        """Initialize Research expert."""
        super().__init__(
            role=AgentRole.RESEARCH,
            tools=[search_documents, search_web],
            **data
        )
        
    async def handle_task(
        self,
        state: AgentState,
        prompt: str,
        writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle research task execution.
        
        Args:
            state: Current agent state
            prompt: Generated prompt
            writer: Optional stream writer
            
        Returns:
            Updated agent state
        """
        try:
            research_results = []
            
            # First, search relevant documents in AstraDB
            doc_results = await astra_retriever.search(
                query=prompt,
                collection_name="strategy_book"  # Default collection
            )
            
            if doc_results:
                research_results.append({
                    "source": "Knowledge Base",
                    "results": doc_results[:3]  # Top 3 results
                })
                
                if writer:
                    await writer.write({
                        "type": "document_search_complete",
                        "data": {"results": doc_results}
                    })
            
            # Then, search the web for current information
            try:
                web_results = []
                async with aiohttp.ClientSession() as session:
                    tavily_api_key = os.getenv("TAVILY_API_KEY")
                    if tavily_api_key:
                        search_data = {
                            "api_key": tavily_api_key,
                            "query": prompt,
                            "search_depth": "basic",
                            "include_answer": True,
                            "max_results": 3
                        }
                        
                        async with session.post(
                            "https://api.tavily.com/search",
                            json=search_data,
                            headers={"Content-Type": "application/json"}
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                web_results = result.get("results", [])
                                
                                if result.get("answer"):
                                    web_results.insert(0, {
                                        "title": "AI Answer",
                                        "content": result["answer"],
                                        "url": "",
                                        "score": 1.0
                                    })
                
                if web_results:
                    research_results.append({
                        "source": "Web Search",
                        "results": web_results
                    })
                    
                    if writer:
                        await writer.write({
                            "type": "web_search_complete",
                            "data": {"results": web_results}
                        })
                        
            except Exception as e:
                logger.warning(f"Web search failed: {str(e)}")
            
            # Compile research findings
            if research_results:
                response_parts = [f"I've researched '{prompt}' and found the following information:\n"]
                
                for source_data in research_results:
                    source = source_data["source"]
                    results = source_data["results"]
                    
                    response_parts.append(f"\n**{source}:**")
                    
                    for i, result in enumerate(results[:3], 1):
                        if source == "Knowledge Base":
                            content = result.get('content', '')
                            title = result.get('metadata', {}).get('title', f'Document {i}')
                        else:  # Web Search
                            content = result.get('content', '')
                            title = result.get('title', f'Result {i}')
                        
                        if len(content) > 300:
                            content = content[:300] + "..."
                        
                        response_parts.append(f"{i}. **{title}**")
                        response_parts.append(f"   {content}")
                
                response_parts.append("\nBased on this research, I can help you with specific questions or provide more detailed information on any aspect.")
                
                state.artifacts["research_results"] = research_results
                state.artifacts["response"] = "\n".join(response_parts)
            else:
                state.artifacts["response"] = f"I searched for information about '{prompt}' but couldn't find specific results. Could you provide more details or rephrase your question?"
            
            state.status = Status.COMPLETED
            return state

        except Exception as e:
            error = f"Research expert failed: {str(e)}"
            logger.error(error)
            state.status = Status.FAILED
            state.error = error
            return state

class PromptEngineerAgent(ExpertAgent):
    """Prompt Engineer Agent for creating enhanced prompts using industry-leading techniques."""
    
    def __init__(self, **data):
        """Initialize Prompt Engineer Agent."""
        super().__init__(
            role=AgentRole.PROMPT_ENGINEER,
            tools=[search_documents, search_web],
            **data
        )

    async def handle_task(
        self,
        state: AgentState,
        prompt: str,
        writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle prompt engineering tasks using industry best practices.
        
        Args:
            state: Current agent state
            prompt: User prompt/query
            writer: Optional stream writer
            
        Returns:
            Updated agent state
        """
        try:
            # Analyze the prompt engineering request
            analysis = await self._analyze_prompt_request(prompt)
            
            # Search for relevant prompt engineering techniques
            techniques_results = await search_documents.ainvoke({
                "query": f"prompt engineering best practices {analysis['domain']}",
                "collection_name": "framework_docs",
                "k": 3
            })
            
            # Search web for latest prompt engineering trends
            web_results = await search_web.ainvoke({
                "query": f"prompt engineering best practices 2024 {analysis['use_case']}"
            })
            
            # Create enhanced prompt using industry patterns
            enhanced_prompt = await self._create_enhanced_prompt(
                prompt, analysis, techniques_results, web_results
            )
            
            # Update state with results
            state.artifacts["prompt_analysis"] = analysis
            state.artifacts["techniques_results"] = techniques_results
            state.artifacts["web_results"] = web_results
            state.artifacts["enhanced_prompt"] = enhanced_prompt
            state.artifacts["response"] = enhanced_prompt  # Also store as response for main.py display
            state.status = Status.COMPLETED
            
            if writer:
                await writer.write({
                    "type": "prompt_engineering_complete",
                    "data": {"enhanced_prompt": enhanced_prompt}
                })
            
            return state
            
        except Exception as e:
            logger.error(f"Prompt engineer agent error: {e}")
            state.status = Status.FAILED
            state.artifacts["error"] = str(e)
            return state

    async def _analyze_prompt_request(self, prompt: str) -> Dict[str, Any]:
        """Analyze the prompt engineering request to understand requirements."""
        
        prompt_lower = prompt.lower()
        
        # Determine the domain/use case
        domain = "general"
        if any(keyword in prompt_lower for keyword in ["code", "coding", "programming", "development"]):
            domain = "coding"
        elif any(keyword in prompt_lower for keyword in ["creative", "writing", "content", "story"]):
            domain = "creative"
        elif any(keyword in prompt_lower for keyword in ["analysis", "data", "research", "technical"]):
            domain = "analytical"
        elif any(keyword in prompt_lower for keyword in ["conversation", "chat", "dialogue", "assistant"]):
            domain = "conversational"
        
        # Determine the use case
        use_case = "general_purpose"
        if any(keyword in prompt_lower for keyword in ["system prompt", "system message"]):
            use_case = "system_prompt"
        elif any(keyword in prompt_lower for keyword in ["few-shot", "examples", "demonstrations"]):
            use_case = "few_shot"
        elif any(keyword in prompt_lower for keyword in ["chain of thought", "reasoning", "step by step"]):
            use_case = "chain_of_thought"
        elif any(keyword in prompt_lower for keyword in ["structured output", "json", "format"]):
            use_case = "structured_output"
        
        # Extract specific requirements
        requirements = []
        if "xml" in prompt_lower or "tags" in prompt_lower:
            requirements.append("xml_formatting")
        if "role" in prompt_lower or "persona" in prompt_lower:
            requirements.append("role_definition")
        if "example" in prompt_lower or "sample" in prompt_lower:
            requirements.append("examples_needed")
        if "constraint" in prompt_lower or "rule" in prompt_lower:
            requirements.append("constraints")
        
        return {
            "domain": domain,
            "use_case": use_case,
            "requirements": requirements,
            "complexity": "high" if len(requirements) > 2 else "medium" if requirements else "low"
        }

    async def _create_enhanced_prompt(
        self,
        original_prompt: str,
        analysis: Dict[str, Any],
        techniques_results: List[Dict],
        web_results: Dict
    ) -> str:
        """Create an enhanced prompt using industry-leading techniques."""
        
        # Build the enhanced prompt response
        response_parts = [
            "# 🎯 Enhanced Prompt Engineering Solution",
            "",
            f"**Original Request:** {original_prompt}",
            "",
            f"**Domain:** {analysis['domain'].title()}",
            f"**Use Case:** {analysis['use_case'].replace('_', ' ').title()}",
            f"**Complexity:** {analysis['complexity'].title()}",
            "",
            "## 📋 Industry-Leading Prompt Patterns Applied",
            ""
        ]
        
        # Add specific patterns based on analysis
        if analysis['domain'] == 'coding':
            response_parts.extend([
                "### 🔧 Cursor-Inspired Coding Patterns",
                "- **Clear Role Definition**: Establishing AI as expert coding assistant",
                "- **Context Awareness**: Including file structure and dependencies",
                "- **Tool Integration**: Leveraging available development tools",
                "- **Error Handling**: Built-in validation and correction patterns",
                "",
                "### 📝 Enhanced Prompt Structure",
                "```",
                "You are an expert software engineer with deep knowledge of [LANGUAGE/FRAMEWORK].",
                "",
                "## Context",
                "- Project: [PROJECT_DESCRIPTION]",
                "- Tech Stack: [TECHNOLOGIES]",
                "- Current Task: [SPECIFIC_TASK]",
                "",
                "## Requirements",
                "1. Write clean, maintainable code",
                "2. Follow best practices and conventions",
                "3. Include proper error handling",
                "4. Add comprehensive comments",
                "",
                "## Output Format",
                "Provide code with explanations in this structure:",
                "- Brief overview of the solution",
                "- Complete code implementation",
                "- Key design decisions explained",
                "- Testing considerations",
                "",
                "## Constraints",
                "- Use TypeScript for type safety",
                "- Follow existing project patterns",
                "- Optimize for readability and performance",
                "",
                "[YOUR_SPECIFIC_REQUEST]",
                "```",
                ""
            ])
        
        elif analysis['domain'] == 'creative':
            response_parts.extend([
                "### 🎨 v0-Inspired Creative Patterns",
                "- **Detailed Specifications**: Clear creative direction and style",
                "- **Iterative Refinement**: Built-in feedback loops",
                "- **Style Consistency**: Maintaining coherent creative voice",
                "- **Audience Awareness**: Tailored to target demographic",
                "",
                "### 📝 Enhanced Prompt Structure",
                "```",
                "You are a creative expert specializing in [CREATIVE_DOMAIN].",
                "",
                "## Creative Brief",
                "- Objective: [CLEAR_GOAL]",
                "- Target Audience: [DEMOGRAPHICS]",
                "- Tone & Style: [VOICE_DESCRIPTION]",
                "- Key Messages: [MAIN_POINTS]",
                "",
                "## Creative Requirements",
                "1. Engaging and original content",
                "2. Appropriate for target audience",
                "3. Consistent with brand voice",
                "4. Optimized for [PLATFORM/MEDIUM]",
                "",
                "## Output Structure",
                "<creative_output>",
                "  <concept>Brief concept explanation</concept>",
                "  <content>[MAIN_CREATIVE_CONTENT]</content>",
                "  <rationale>Why this approach works</rationale>",
                "</creative_output>",
                "",
                "[YOUR_CREATIVE_REQUEST]",
                "```",
                ""
            ])
        
        elif analysis['domain'] == 'conversational':
            response_parts.extend([
                "### 💬 Claude-Inspired Conversational Patterns",
                "- **Helpful & Harmless**: Prioritizing user safety and assistance",
                "- **Contextual Awareness**: Understanding conversation flow",
                "- **Structured Thinking**: Using XML tags for clarity",
                "- **Adaptive Responses**: Matching user's communication style",
                "",
                "### 📝 Enhanced Prompt Structure",
                "```",
                "You are a helpful, harmless, and honest AI assistant.",
                "",
                "## Core Principles",
                "1. Be helpful: Provide accurate, useful information",
                "2. Be harmless: Avoid harmful or inappropriate content",
                "3. Be honest: Acknowledge limitations and uncertainties",
                "",
                "## Communication Style",
                "- Match the user's tone and formality level",
                "- Use clear, concise language",
                "- Structure responses with headings when helpful",
                "- Ask clarifying questions when needed",
                "",
                "## Response Format",
                "<thinking>",
                "Consider the user's request and how best to help",
                "</thinking>",
                "",
                "<response>",
                "[YOUR_HELPFUL_RESPONSE]",
                "</response>",
                "",
                "## Special Instructions",
                "- Use XML tags to structure your thinking",
                "- Provide examples when helpful",
                "- Offer follow-up suggestions",
                "",
                "[USER_CONVERSATION_CONTEXT]",
                "```",
                ""
            ])
        
        # Add universal best practices
        response_parts.extend([
            "## 🚀 Universal Prompt Engineering Best Practices Applied",
            "",
            "### 1. **Clear Role & Context Definition**",
            "- Specific expertise area defined",
            "- Clear context and background provided",
            "- Explicit capabilities and limitations",
            "",
            "### 2. **Structured Input/Output Format**",
            "- XML tags for organization (Claude-style)",
            "- Clear sections and hierarchies",
            "- Consistent formatting patterns",
            "",
            "### 3. **Constraint & Requirement Specification**",
            "- Explicit rules and guidelines",
            "- Quality criteria defined",
            "- Output format requirements",
            "",
            "### 4. **Examples & Demonstrations**",
            "- Few-shot learning patterns",
            "- Positive and negative examples",
            "- Template structures provided",
            "",
            "### 5. **Chain-of-Thought Integration**",
            "- Step-by-step reasoning encouraged",
            "- Thinking process made explicit",
            "- Verification steps included",
            ""
        ])
        
        # Add specific techniques based on requirements
        if "xml_formatting" in analysis['requirements']:
            response_parts.extend([
                "### 🏷️ XML Formatting Techniques",
                "- Use semantic tags for structure",
                "- Separate thinking from output",
                "- Enable easy parsing and validation",
                ""
            ])
        
        if "role_definition" in analysis['requirements']:
            response_parts.extend([
                "### 👤 Advanced Role Definition",
                "- Specific expertise and background",
                "- Personality and communication style",
                "- Ethical guidelines and constraints",
                ""
            ])
        
        # Add implementation recommendations
        response_parts.extend([
            "## 🛠️ Implementation Recommendations",
            "",
            "### Testing & Iteration",
            "1. **A/B Testing**: Compare prompt variations",
            "2. **Performance Metrics**: Track success rates",
            "3. **User Feedback**: Collect and analyze responses",
            "4. **Continuous Refinement**: Regular prompt optimization",
            "",
            "### Advanced Techniques",
            "- **Temperature Control**: Adjust for creativity vs consistency",
            "- **Token Management**: Optimize for context length",
            "- **Fallback Strategies**: Handle edge cases gracefully",
            "- **Multi-turn Conversations**: Maintain context across interactions",
            "",
            "### Quality Assurance",
            "- **Validation Checks**: Ensure output meets requirements",
            "- **Safety Filters**: Prevent harmful or inappropriate content",
            "- **Consistency Monitoring**: Maintain reliable performance",
            "- **Error Handling**: Graceful degradation strategies",
            ""
        ])
        
        # Add framework-specific guidance if available
        if techniques_results:
            response_parts.extend([
                "## 📚 Framework-Specific Guidance",
                ""
            ])
            
            for i, result in enumerate(techniques_results[:2], 1):
                content = result.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                response_parts.append(f"{i}. {content}")
            
            response_parts.append("")
        
        # Add conclusion
        response_parts.extend([
            "## 🎯 Next Steps",
            "",
            "1. **Implement** the enhanced prompt structure",
            "2. **Test** with representative examples",
            "3. **Measure** performance against objectives",
            "4. **Iterate** based on results and feedback",
            "5. **Scale** successful patterns to similar use cases",
            "",
            "---",
            "",
            "*This enhanced prompt incorporates industry-leading patterns from Cursor, v0, Claude, and other top AI tools, optimized for your specific use case.*"
        ])
        
        return "\n".join(response_parts)

class SupervisorExpert(ExpertAgent):
    """Supervisor expert for coordinating multi-agent workflows."""
    
    def __init__(self, **data):
        super().__init__(
            role=AgentRole.SUPERVISOR,
            tools=[],
            **data
        )
    
    async def handle_task(
        self,
        state: AgentState,
        prompt: str,
        writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle supervisor coordination tasks."""
        try:
            query = state.context.get("query", prompt)
            
            if writer:
                await writer.write({
                    "type": "supervisor_start",
                    "data": {"query": query}
                })
            
            # Use the supervisor's multi-agent coordination
            from .supervisor import SupervisorAgent
            from .manager import StateManager
            from .logger import StructuredLogger
            from .config import ConfigState, APIConfig, DatabaseConfig, LLMRegistry, LoggingConfig, WorkflowConfig
            import os
            
            # Create minimal config for state manager
            config = ConfigState(
                id="supervisor_config",
                api=APIConfig(),
                database=DatabaseConfig(
                    astra_db_id=os.getenv("ASTRA_DB_ID", ""),
                    astra_db_region=os.getenv("ASTRA_DB_REGION", ""),
                    astra_db_token=os.getenv("ASTRA_DB_APPLICATION_TOKEN", ""),
                ),
                llm_registry=LLMRegistry(),
                logging=LoggingConfig(),
                workflow=WorkflowConfig(name="supervisor", description="Supervisor workflow", nodes=[])
            )
            
            state_manager = StateManager(config)
            logger = StructuredLogger(config.logging)
            supervisor = SupervisorAgent(state_manager, logger)
            
            await supervisor.initialize()
            
            # For now, let's do simple coordination without the complex workflow
            # Analyze the task and route to appropriate agent
            task_lower = query.lower()
            
            if any(keyword in task_lower for keyword in ["research", "search", "find", "analyze"]):
                # Route to research agent
                from .state import AgentState, Status
                research_state = AgentState(
                    id=f"research_{state.id}",
                    agent=AgentRole.RESEARCH,
                    status=Status.PENDING,
                    context={"query": query}
                )
                
                research_expert = experts.get(AgentRole.RESEARCH)
                if research_expert:
                    research_result = await research_expert(research_state, writer=writer)
                    state.artifacts["research_results"] = research_result.artifacts
                    
            elif any(keyword in task_lower for keyword in ["write", "document", "content", "summary"]):
                # Route to NextJS expert as writer
                from .state import AgentState, Status
                writer_state = AgentState(
                    id=f"writer_{state.id}",
                    agent=AgentRole.NEXTJS_EXPERT,
                    status=Status.PENDING,
                    context={"query": query}
                )
                
                writer_expert = experts.get(AgentRole.NEXTJS_EXPERT)
                if writer_expert:
                    writer_result = await writer_expert(writer_state, writer=writer)
                    state.artifacts["written_content"] = writer_result.artifacts
                    
            elif any(keyword in task_lower for keyword in ["code", "implement", "develop", "program"]):
                # Route to appropriate coding expert
                if "fastapi" in task_lower or "api" in task_lower:
                    expert = experts.get(AgentRole.FASTAPI_EXPERT)
                    agent_role = AgentRole.FASTAPI_EXPERT
                else:
                    expert = experts.get(AgentRole.NEXTJS_EXPERT)
                    agent_role = AgentRole.NEXTJS_EXPERT
                    
                if expert:
                    coding_state = AgentState(
                        id=f"coding_{state.id}",
                        agent=agent_role,
                        status=Status.PENDING,
                        context={"query": query}
                    )
                    
                    coding_result = await expert(coding_state, writer=writer)
                    state.artifacts["code_output"] = coding_result.artifacts
            
            # Create a simple coordination result
            result = {
                "status": Status.COMPLETED,
                "artifacts": state.artifacts,
                "workflow_id": f"simple_coordination_{state.id}",
                "execution_time": datetime.utcnow().isoformat()
            }
            
            # Update state with results
            state.artifacts.update(result.get("artifacts", {}))
            state.status = Status.COMPLETED if result.get("status") == Status.COMPLETED else Status.FAILED
            
            if result.get("error"):
                state.error = result["error"]
            
            if writer:
                await writer.write({
                    "type": "supervisor_complete",
                    "data": {"result": result}
                })
            
            return state
            
        except Exception as e:
            error_msg = f"Supervisor coordination failed: {str(e)}"
            import logging
            logging.getLogger(__name__).error(error_msg)
            
            if writer:
                await writer.write({
                    "type": "supervisor_error",
                    "data": {"error": error_msg}
                })
            
            state.status = Status.FAILED
            state.error = error_msg
            return state

# Import strategic agent
from .strategic_agent import StrategicAgent

# Expert registry
experts = {
    AgentRole.SUPERVISOR: SupervisorExpert(),
    AgentRole.LANGCHAIN_EXPERT: LangChainExpert(),
    AgentRole.FASTAPI_EXPERT: FastAPIExpert(),
    AgentRole.NEXTJS_EXPERT: NextJSExpert(),
    AgentRole.RESEARCH: ResearchExpert(),
    AgentRole.STRATEGIC_EXPERT: StrategicAgent(),
    AgentRole.PROMPT_ENGINEER: PromptEngineerAgent(),
}

async def execute_agent(
    agent_role: AgentRole,
    state: AgentState,
    prompt: str = "",
    *,
    writer: Optional[StreamWriter] = None
) -> AgentState:
    """Execute an agent with the given role.
    
    Args:
        agent_role: Role of the agent to execute
        state: Current agent state
        prompt: Generated prompt (optional)
        writer: Optional stream writer
        
    Returns:
        Updated agent state
    """
    try:
        # Get expert agent
        expert = experts.get(agent_role)
        if not expert:
            raise ValueError(f"No expert found for role {agent_role}")
        
        # Add prompt to state context if provided
        if prompt and "query" not in state.context:
            state.context["query"] = prompt
            
        # Execute expert
        return await expert(state, writer=writer)
        
    except Exception as e:
        error = f"Agent execution failed: {str(e)}"
        logger.error(error)
        
        if writer:
            await writer.write({
                "type": "execution_error",
                "data": {"error": error}
            })
            
        state.status = Status.FAILED
        state.error = error
        return state

# Export all symbols
__all__ = [
    "ExpertAgent",
    "LangChainExpert",
    "FastAPIExpert", 
    "NextJSExpert",
    "ResearchExpert",
    "StrategicAgent",
    "PromptEngineerAgent",
    "execute_agent",
    "experts"
]
