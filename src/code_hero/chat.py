"""Chat handling with proper integration to existing modular architecture."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request

from .agent_expert import experts
from .context import get_services
from .hierarchical_agents import process_with_hierarchical_agents
from .langgraph_agents import (
    create_code_expert_workflow,
    create_research_expert_workflow,
    create_supervisor_workflow,
)
from .logger import StructuredLogger
from .manager import StateManager
from .state import AgentRole, AgentState, ChatState, Message, Status
from .supervisor import SupervisorAgent
from .workflow import workflow_runner

router = APIRouter()

# Global workflow instances (initialized in main.py)
supervisor_workflow = None
code_workflow = None
research_workflow = None


def initialize_chat_workflows():
    """Initialize chat workflows - called from main.py"""
    global supervisor_workflow, code_workflow, research_workflow

    try:
        supervisor_workflow = create_supervisor_workflow()
        code_workflow = create_code_expert_workflow()
        research_workflow = create_research_expert_workflow()
    except Exception:
        # Fallback if LangGraph not available
        supervisor_workflow = None
        code_workflow = None
        research_workflow = None


@router.post("/")
async def chat(
    request: Request,
    services: tuple[StateManager, StructuredLogger, SupervisorAgent] = Depends(
        get_services
    ),
):
    """Chat endpoint using existing modular architecture."""
    state_manager, logger, supervisor = services

    try:
        # Parse request body
        body = await request.json()
        message = body.get("message", "")
        conversation_id = body.get("conversation_id")

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Get or create chat state using existing manager
        if conversation_id:
            state = await state_manager.get_chat_state(conversation_id)
            if not state:
                raise HTTPException(404, f"Chat {conversation_id} not found")
        else:
            # Create or get default project
            project_state = await state_manager.create_default_project()
            if not project_state:
                raise HTTPException(500, "Failed to create default project")

            # Create new chat using existing manager
            state = await state_manager.create_chat(project_state.id)
            if not state:
                raise HTTPException(500, "Failed to create chat state")

        # Add user message to state
        user_message = Message(
            role="user",
            content=message,
            timestamp=datetime.now(),
            metadata={"source": "api"},
        )
        state.messages.append(user_message)

        # Route to appropriate workflow based on message content
        ai_response = await route_to_workflow(message, state, logger)

        # Add AI response to state
        assistant_message = Message(
            role="assistant",
            content=ai_response,
            timestamp=datetime.now(),
            metadata={"source": "workflow"},
        )
        state.messages.append(assistant_message)

        state.status = Status.COMPLETED

        # Update state using existing manager
        await state_manager.update_state(state)

        return {
            "response": ai_response,
            "conversation_id": state.conversation_id,
            "messages": [msg.dict() for msg in state.messages],
            "status": str(state.status),
            "active_agent": str(state.active_agent),
        }

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "message": "Failed to process chat request"},
        )


async def route_to_workflow(
    message: str, state: ChatState, logger: StructuredLogger
) -> str:
    """Route message to appropriate workflow or expert."""
    try:
        # Try the new hierarchical agent system first
        logger.info("Routing to hierarchical agent system")
        response = await process_with_hierarchical_agents(
            message, state.conversation_id
        )

        if response and not response.startswith(
            "Hierarchical agent system is not available"
        ):
            return response

        # Fallback to existing workflow system if hierarchical system fails
        logger.warning(
            "Hierarchical system unavailable, falling back to traditional workflow"
        )
        return await execute_traditional_workflow(message, state, logger)

    except Exception as e:
        logger.error(f"Workflow routing error: {str(e)}")
        return await execute_traditional_workflow(message, state, logger)


async def execute_langgraph_workflow(workflow, message: str, state: ChatState) -> str:
    """Execute LangGraph workflow."""
    try:
        from langchain_core.messages import AIMessage, HumanMessage

        # Convert existing messages to LangChain format
        langchain_messages = []
        for msg in state.messages[:-1]:  # Exclude the current message
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))

        # Add current message
        langchain_messages.append(HumanMessage(content=message))

        # Create workflow state
        workflow_state = {
            "messages": langchain_messages,
            "agent_role": "supervisor",
            "task_type": "chat",
            "tools_available": [],
            "context": state.context,
            "status": "pending",
        }

        # Execute workflow
        config = {"configurable": {"thread_id": state.conversation_id}}
        result = workflow.invoke(workflow_state, config=config)

        # Extract response
        final_messages = result.get("messages", [])
        if final_messages:
            last_message = final_messages[-1]
            if hasattr(last_message, "content"):
                return last_message.content

        return "I processed your request but couldn't generate a proper response."

    except Exception as e:
        raise Exception(f"LangGraph workflow execution failed: {str(e)}")


async def execute_traditional_workflow(
    message: str, state: ChatState, logger: StructuredLogger
) -> str:
    """Execute traditional workflow using existing workflow system."""
    try:
        # Use existing workflow runner
        initial_state = {
            "messages": [{"role": "user", "content": message}],
            "conversation_id": state.conversation_id,
            "project_id": state.context.get("project_id", "default"),
            "context": state.context,
        }

        # Execute workflow using existing system
        result = await workflow_runner(initial_state)

        # Extract response
        if isinstance(result, dict):
            if "response" in result:
                return result["response"]
            elif "messages" in result and result["messages"]:
                last_msg = result["messages"][-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    return last_msg["content"]

        # Fallback to direct expert execution
        return await execute_direct_expert(message, state, logger)

    except Exception as e:
        logger.error(f"Traditional workflow error: {str(e)}")
        return await execute_direct_expert(message, state, logger)


async def execute_direct_expert(
    message: str, state: ChatState, logger: StructuredLogger
) -> str:
    """Direct expert execution as final fallback."""
    try:
        # Determine appropriate expert based on message
        message_lower = message.lower()

        if any(
            word in message_lower
            for word in ["code", "function", "python", "javascript", "api"]
        ):
            expert_role = AgentRole.CODE_GENERATOR
        elif any(
            word in message_lower for word in ["research", "search", "find", "learn"]
        ):
            expert_role = AgentRole.RESEARCH
        elif any(
            word in message_lower for word in ["document", "file", "read", "write"]
        ):
            expert_role = AgentRole.DOCUMENTATION
        else:
            expert_role = AgentRole.SUPERVISOR

        # Get expert
        expert = experts.get(expert_role)
        if not expert:
            # Provide a helpful response even when expert is not available
            return f"I understand you're asking about {message}. While the {expert_role.value} expert is currently unavailable, I can still help you. Could you please provide more specific details about what you'd like to accomplish?"

        # Create agent state
        agent_state = AgentState(
            id=f"chat_task_{datetime.now().timestamp()}",
            agent=expert_role,
            status=Status.PENDING,
            context={
                "query": message,
                "conversation_id": state.conversation_id,
                "project_id": state.context.get("project_id", "default"),
            },
        )

        # Execute expert
        result = await expert.handle_task(agent_state, message)

        # Extract response
        if result.status == Status.COMPLETED and result.artifacts:
            response = (
                result.artifacts.get("response")
                or result.artifacts.get("generated_code")
                or result.artifacts.get("search_results")
                or str(result.artifacts)
            )
            return response if response else "I've processed your request successfully."
        elif result.error:
            return f"I encountered an issue while processing your request: {result.error}. Let me try to help you in a different way. Could you rephrase your question or provide more context?"
        else:
            return "I've processed your request. Could you please provide more specific details about what you'd like me to help you with?"

    except Exception as e:
        logger.error(f"Direct expert execution error: {str(e)}")
        # Provide a helpful fallback response
        return f"I'm here to help you with your request: '{message}'. While I encountered a technical issue, I can still assist you. Could you please tell me more about what you're trying to accomplish? For example, are you looking to:\n\n• Write or debug code\n• Research a topic\n• Create documentation\n• Get general assistance\n\nI'll do my best to help once I understand your needs better."


@router.get("/{conversation_id}")
async def get_chat_history(
    request: Request,
    conversation_id: str,
    services: tuple[StateManager, StructuredLogger, SupervisorAgent] = Depends(
        get_services
    ),
):
    """Get chat history using existing state manager."""
    state_manager, logger, _ = services

    try:
        state = await state_manager.get_chat_state(conversation_id)
        if not state:
            raise HTTPException(404, f"Chat {conversation_id} not found")

        return {
            "conversation_id": state.conversation_id,
            "messages": [msg.dict() for msg in state.messages],
            "status": str(state.status),
            "active_agent": str(state.active_agent),
        }

    except Exception as e:
        logger.error(f"Get chat history error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router and initialization function
__all__ = ["router", "initialize_chat_workflows"]
