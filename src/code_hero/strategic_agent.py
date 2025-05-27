"""Strategic Agent for AI Agent and Workflow Building.

This module implements a Strategic Agent that uses the "Playing to Win" framework
to guide the development of AI agents and workflows. It provides strategic
planning, decision-making, and execution guidance.
"""

from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime
from enum import Enum

from .agent_expert import ExpertAgent
from .state import AgentState, AgentRole, Status
from .tools import search_documents, search_web, astra_retriever
from .prompts import build_enhanced_prompt

try:
    from langgraph.types import StreamWriter
except ImportError:
    class StreamWriter:
        async def write(self, data):
            pass

logger = logging.getLogger(__name__)

class StrategicContext(Enum):
    """Strategic contexts for AI development."""
    AI_AGENT_DEVELOPMENT = "ai_agent_development"
    WORKFLOW_AUTOMATION = "workflow_automation"
    CODE_GENERATION = "code_generation"
    DOCUMENTATION_SYSTEM = "documentation_system"
    RESEARCH_ANALYSIS = "research_analysis"
    GENERAL_STRATEGY = "general_strategy"

class StrategicAgent(ExpertAgent):
    """Strategic Agent for AI agent and workflow building."""
    
    def __init__(self, **data):
        """Initialize Strategic Agent."""
        super().__init__(
            role=AgentRole.STRATEGIC_EXPERT,
            tools=[search_documents, search_web],
            **data
        )

    async def handle_task(
        self,
        state: AgentState,
        prompt: str,
        writer: Optional[StreamWriter] = None
    ) -> AgentState:
        """Handle strategic planning and guidance tasks.
        
        Args:
            state: Current agent state
            prompt: User prompt/query
            writer: Optional stream writer
            
        Returns:
            Updated agent state
        """
        try:
            # Determine strategic context from prompt
            context = self._determine_context(prompt)
            
            # Search for strategic guidance from strategy_book collection
            strategy_results = await search_documents.ainvoke({
                "query": f"strategic planning {prompt}",
                "collection_name": "strategy_book",
                "k": 3
            })
            
            # Get framework guidance
            framework_guidance = self._get_framework_guidance(context, prompt)
            
            # Create strategic response using industry patterns
            response = await self._create_strategic_response(
                prompt, context, strategy_results, framework_guidance
            )
            
            # Update state with results
            state.artifacts["strategic_context"] = context.value if hasattr(context, 'value') else str(context)
            state.artifacts["strategy_results"] = strategy_results
            state.artifacts["framework_guidance"] = framework_guidance
            state.artifacts["response"] = response
            state.status = Status.COMPLETED
            
            if writer:
                await writer.write({
                    "type": "strategic_analysis_complete",
                    "data": {"context": context.value if hasattr(context, 'value') else str(context), "response": response}
                })
            
            return state
            
        except Exception as e:
            logger.error(f"Strategic agent error: {e}")
            state.status = Status.FAILED
            state.artifacts["error"] = str(e)
            return state

    def _determine_context(self, prompt: str) -> str:
        """Determine the strategic context from the prompt."""
        
        prompt_lower = prompt.lower()
        
        if any(keyword in prompt_lower for keyword in ["agent", "ai agent", "intelligent agent"]):
            return "ai_agent_development"
        elif any(keyword in prompt_lower for keyword in ["workflow", "automation", "process"]):
            return "workflow_automation"
        elif any(keyword in prompt_lower for keyword in ["documentation", "docs", "knowledge"]):
            return "documentation_system"
        elif any(keyword in prompt_lower for keyword in ["code", "generation", "programming"]):
            return "code_generation"
        elif any(keyword in prompt_lower for keyword in ["research", "analysis", "investigation"]):
            return "research_analysis"
        else:
            return "general_strategy"

    def _get_framework_guidance(self, context: str, prompt: str) -> Dict[str, List[str]]:
        """Get framework-specific guidance based on context."""
        
        base_guidance = {
            "playing_to_win": [
                "Define your winning aspiration",
                "Choose where to play",
                "Determine how to win",
                "Build required capabilities",
                "Design management systems"
            ]
        }
        
        context_specific_guidance = {
            "ai_agent_development": [
                "Focus on specific use cases and user needs",
                "Design for scalability and maintainability", 
                "Implement robust error handling and monitoring",
                "Use industry-leading frameworks like LangGraph and Pydantic",
                "Apply strategic prompt engineering patterns"
            ],
            "workflow_automation": [
                "Map current processes and identify bottlenecks",
                "Design for parallel execution where possible",
                "Implement proper state management",
                "Use conditional logic for dynamic routing",
                "Build in human-in-the-loop capabilities"
            ],
            "code_generation": [
                "Follow industry coding standards and patterns",
                "Implement comprehensive validation and error handling",
                "Use type hints and proper documentation",
                "Design modular and reusable components",
                "Apply test-driven development practices"
            ]
        }
        
        # Get context-specific guidance, defaulting to general if not found
        specific_guidance = context_specific_guidance.get(
            context, 
            ["Apply strategic thinking to the problem", "Consider long-term implications"]
        )
        
        return {
            **base_guidance,
            "context_specific": specific_guidance
        }

    async def _create_strategic_response(
        self,
        prompt: str,
        context: str,
        strategy_results: List[Dict],
        framework_guidance: Dict[str, List[str]]
    ) -> str:
        """Create a strategic response based on the analysis."""
        
        # Build enhanced context for the strategic prompt
        enhanced_context = {
            "query": prompt,
            "strategic_context": context,
            "business_objectives": "strategic AI system development",
            "technical_constraints": "existing Code Hero architecture",
            "timeline": "iterative development",
            "available_resources": "AstraDB, multiple AI frameworks"
        }
        
        # Create a temporary state for prompt building
        from .state import AgentState, Status
        temp_state = AgentState(
            agent=AgentRole.STRATEGIC_EXPERT,
            status=Status.RUNNING,
            context=enhanced_context
        )
        
        # Use enhanced prompts with industry patterns
        enhanced_prompt = build_enhanced_prompt(
            agent_role=AgentRole.STRATEGIC_EXPERT,
            state=temp_state
        )
        
        # Create structured response using industry patterns
        context_name = context.value if hasattr(context, 'value') else str(context)
        response_parts = [
            f"# 🎯 Strategic Analysis: {context_name.replace('_', ' ').title()}",
            "",
            f"**Query:** {prompt}",
            "",
            "## 📋 Strategic Framework Analysis",
            ""
        ]
        
        # Add strategic principles from strategy book
        if strategy_results:
            response_parts.extend([
                "### 🎲 Strategic Principles",
                ""
            ])
            
            for i, result in enumerate(strategy_results[:3], 1):
                content = result.get("content", "")
                if len(content) > 300:
                    content = content[:300] + "..."
                response_parts.append(f"{i}. {content}")
            
            response_parts.append("")
        
        # Add framework guidance
        response_parts.extend([
            "### 🏗️ Framework Guidance",
            "",
            "**Playing to Win Framework:**"
        ])
        
        for principle in framework_guidance.get("playing_to_win", []):
            response_parts.append(f"- {principle}")
        
        response_parts.extend([
            "",
            "**Context-Specific Recommendations:**"
        ])
        
        for recommendation in framework_guidance.get("context_specific", []):
            response_parts.append(f"- {recommendation}")
        
        # Add industry best practices section
        response_parts.extend([
            "",
            "## 🚀 Industry Best Practices",
            "",
            "### Pydantic & Data Validation",
            "- Use proper type hints and validation patterns",
            "- Implement custom validators for complex business logic", 
            "- Leverage Field() for advanced validation constraints",
            "- Use model_validator for cross-field validation",
            "",
            "### LangGraph & Agent Workflows", 
            "- Design workflows as graphs with clear state management",
            "- Implement proper error handling and retry logic",
            "- Use conditional edges for dynamic routing",
            "- Build in human-in-the-loop capabilities",
            "- Apply parallelization where appropriate",
            "",
            "### Strategic Implementation",
            "- Start with simple, focused use cases",
            "- Build modular, reusable components",
            "- Implement comprehensive monitoring and observability",
            "- Design for scalability from the beginning",
            "- Apply iterative development with continuous feedback"
        ])
        
        return "\n".join(response_parts)

    def _get_context_recommendations(self, context: str, prompt: str) -> List[str]:
        """Get context-specific strategic recommendations."""
        
        base_recommendations = [
            "Start with clear strategic objectives",
            "Apply proven strategic frameworks", 
            "Focus on sustainable competitive advantages",
            "Build capabilities that align with strategy"
        ]
        
        context_specific = {
            "ai_agent_development": [
                "Define your winning aspiration for the AI agent",
                "Choose specific use cases to focus on (where to play)",
                "Develop unique AI capabilities for competitive advantage",
                "Implement monitoring and feedback systems"
            ],
            "workflow_automation": [
                "Map current processes and identify improvement opportunities",
                "Design workflows with clear decision points",
                "Implement proper state management and error handling",
                "Build in scalability and maintainability from the start"
            ],
            "code_generation": [
                "Focus on generating clean, maintainable code",
                "Implement proper validation and testing",
                "Use industry-standard patterns and practices",
                "Design for reusability and modularity"
            ]
        }
        
        specific_recs = context_specific.get(context, [])
        return base_recommendations + specific_recs 