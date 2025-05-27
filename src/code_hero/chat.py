"""Chat handling for the strategic framework."""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Optional, Dict, Any, List
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticToolsParser

try:
    from langgraph.types import StreamWriter
except ImportError:
    # Fallback for older LangGraph versions
    class StreamWriter:
        async def write(self, data):
            pass

from .state import ChatState, AgentState, AgentRole, Status, Message
from .context import get_services
from .manager import StateManager
from .logger import StructuredLogger
from .supervisor import SupervisorAgent
from .tools import tool_registry

router = APIRouter()

async def process_message(
    state: ChatState,
    message: str,
    supervisor: SupervisorAgent,
    *,
    writer: Optional[StreamWriter] = None
) -> ChatState:
    """Process a chat message.
    
    Args:
        state: Current chat state
        message: Message to process
        supervisor: Supervisor agent instance
        writer: Optional stream writer
        
    Returns:
        Updated chat state
    """
    try:
        # Create and add user message
        user_message = Message(
            role="user",
            content=message,
            timestamp=datetime.utcnow(),
            metadata={"source": "api"}
        )
        state.messages.append(user_message)

        if writer:
            await writer.write({
                "type": "message",
                "data": user_message.dict()
            })

        # Create agent state for processing
        agent_state = AgentState(
            id=f"agent_{datetime.utcnow().timestamp()}",
            agent=state.active_agent,
            status=Status.RUNNING,
            context={
                "message": message,
                "conversation_id": state.conversation_id,
                "chat_history": [
                    HumanMessage(content=msg.content) if msg.role == "user"
                    else AIMessage(content=msg.content) if msg.role == "assistant"
                    else SystemMessage(content=msg.content)
                    for msg in state.messages
                ]
            },
        )

        # Get available tools for the agent
        tools = tool_registry.get_all_tools()
        
        # Execute agent with tools
        try:
            # Create messages for the agent
            messages = [
                SystemMessage(content="You are a helpful AI assistant that can use tools to accomplish tasks."),
                *agent_state.context["chat_history"],
                HumanMessage(content=message)
            ]
            
            # Get response from supervisor with tools
            response = await supervisor.dispatch_task(
                project_id="default",
                task=agent_state,
                tools=tools
            )
            
            # Parse tool calls if present
            if isinstance(response, dict) and "tool_calls" in response:
                tool_calls = response["tool_calls"]
                results = []
                
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Get and execute tool
                    tool = tool_registry.get_tool(tool_name)
                    if tool:
                        result = await tool(**tool_args)
                        results.append(result)
                
                # Format results into a response
                response = f"I used the following tools:\n\n" + "\n".join(
                    f"- {r}" for r in results
                )
            
        except Exception as e:
            response = "I apologize, but I encountered an error processing your request. Please try again."

        # Create and add assistant message
        assistant_message = Message(
            role="assistant",
            content=response if response else "I'm still processing your request.",
            timestamp=datetime.utcnow(),
            metadata={
                "agent": str(state.active_agent),
                "status": str(agent_state.status)
            }
        )
        state.messages.append(assistant_message)

        if writer:
            await writer.write({
                "type": "message",
                "data": assistant_message.dict()
            })

        state.status = Status.COMPLETED
        return state

    except Exception as e:
        if writer:
            await writer.write({
                "type": "error",
                "data": {"error": str(e)}
            })
        state.status = Status.FAILED
        state.error = str(e)
        return state

@router.post("/")
async def chat(
    request: Request,
    message: str,
    conversation_id: Optional[str] = None,
    services: tuple[StateManager, StructuredLogger, SupervisorAgent] = Depends(get_services),
):
    """Chat endpoint using managed state."""
    state_manager, logger, supervisor = services

    try:
        # Get or create chat state
        if conversation_id:
            state = await state_manager.get_chat_state(conversation_id)
            if not state:
                raise HTTPException(404, f"Chat {conversation_id} not found")
        else:
            state = await state_manager.create_chat()
            if not state:
                raise HTTPException(500, "Failed to create chat state")

        # Process message with streaming
        stream_writer = StreamWriter()
        state = await process_message(state, message, supervisor, writer=stream_writer)
        await state_manager.update_state(state)

        return {
            "response": state.messages[-1].content if state.messages else None,
            "conversation_id": state.conversation_id,
            "messages": [msg.dict() for msg in state.messages],
            "status": state.status,
            "active_agent": str(state.active_agent)
        }

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "Failed to process chat request"
            }
        )

@router.get("/{conversation_id}")
async def get_chat_history(
    request: Request,
    conversation_id: str,
    services: tuple[StateManager, StructuredLogger, SupervisorAgent] = Depends(get_services),
):
    """Get chat history using managed state."""
    state_manager, logger, _ = services

    try:
        state = await state_manager.get_chat_state(conversation_id)
        if not state:
            raise HTTPException(404, f"Chat {conversation_id} not found")

        return {
            "conversation_id": state.conversation_id,
            "messages": state.messages,
            "status": state.status,
            "active_agent": state.active_agent,
        }

    except Exception as e:
        logger.log_error(e)
        raise HTTPException(status_code=500, detail=str(e))

# Export router
__all__ = ["router"]
