"""State management for the strategic framework."""

from typing import Dict, Any, Optional, List
import logging
import asyncio
from datetime import datetime
from pydantic import BaseModel

from .state import (
    BaseState,
    ChatState,
    AgentState,
    GraphState,
    ProjectState,
    Status,
    AgentRole
)
from .config import ConfigState

logger = logging.getLogger(__name__)

class StateManager:
    """Manages application state."""

    def __init__(self, config: ConfigState):
        """Initialize state manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self._initialized = False
        self._states: Dict[str, BaseState] = {}
        self.agent_states = {}  # For supervisor compatibility
        self._locks: Dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        """Initialize state manager."""
        if self._initialized:
            return

        try:
            # Initialize state storage
            self._states = {}
            self._locks = {}
            
            # Create root project state
            root_project = ProjectState(
                id="root",
                project_name="Root Project",
                description="Root project for all workflows",
                graph_state=GraphState(
                    id="root_graph",
                    status=Status.PENDING
                ),
                status=Status.RUNNING,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            await self.add_state(root_project)
            
            self._initialized = True
            logger.info("State manager initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize state manager: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear all states
            self._states.clear()
            self._locks.clear()
            
            self._initialized = False
            logger.info("State manager cleaned up successfully")
            
        except Exception as e:
            error_msg = f"Failed to clean up state manager: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def check_health(self) -> Dict[str, Any]:
        """Check state manager health."""
        return {
            "initialized": self._initialized,
            "project_count": len([s for s in self._states.values() if isinstance(s, ProjectState)]),
            "chat_count": len([s for s in self._states.values() if isinstance(s, ChatState)]),
            "agent_count": len([s for s in self._states.values() if isinstance(s, AgentState)]),
            "graph_count": len([s for s in self._states.values() if isinstance(s, GraphState)])
        }

    async def add_state(self, state: BaseState) -> None:
        """Add new state.
        
        Args:
            state: State to add
        """
        try:
            # Create lock if needed
            if state.id not in self._locks:
                self._locks[state.id] = asyncio.Lock()
                
            async with self._locks[state.id]:
                # Add state
                self._states[state.id] = state
                logger.info(f"Added state: {state.id}")
                
        except Exception as e:
            error_msg = f"Failed to add state {state.id}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def get_state(self, state_id: str) -> Optional[BaseState]:
        """Get state by ID.
        
        Args:
            state_id: State ID
            
        Returns:
            State if found, None otherwise
        """
        try:
            return self._states.get(state_id)
        except Exception as e:
            error_msg = f"Failed to get state {state_id}: {str(e)}"
            logger.error(error_msg)
            return None

    async def update_state(self, state_id: str, **updates) -> None:
        """Update state.
        
        Args:
            state_id: State ID
            **updates: State updates
        """
        try:
            if state_id not in self._locks:
                self._locks[state_id] = asyncio.Lock()
                
            async with self._locks[state_id]:
                if state_id in self._states:
                    state = self._states[state_id]
                    for key, value in updates.items():
                        setattr(state, key, value)
                    state.updated_at = datetime.utcnow()
                    logger.info(f"Updated state: {state_id}")
                    
        except Exception as e:
            error_msg = f"Failed to update state {state_id}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def delete_state(self, state_id: str) -> None:
        """Delete state.
        
        Args:
            state_id: State ID
        """
        try:
            if state_id not in self._locks:
                self._locks[state_id] = asyncio.Lock()
                
            async with self._locks[state_id]:
                if state_id in self._states:
                    del self._states[state_id]
                    logger.info(f"Deleted state: {state_id}")
                    
        except Exception as e:
            error_msg = f"Failed to delete state {state_id}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def create_chat(self, project_id: str) -> ChatState:
        """Create new chat state.
        
        Args:
            project_id: Project ID
            
        Returns:
            Created chat state
        """
        try:
            # Validate project exists
            if project_id not in self._states:
                raise ValueError(f"Project {project_id} not found")
                
            # Create chat state
            chat_state = ChatState(
                id=f"chat_{datetime.utcnow().timestamp()}",
                conversation_id=f"conv_{datetime.utcnow().timestamp()}",
                messages=[],
                status=Status.RUNNING,
                context={"project_id": project_id},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Add state
            await self.add_state(chat_state)
            logger.info(f"Created chat: {chat_state.id}")
            
            return chat_state
            
        except Exception as e:
            error_msg = f"Failed to create chat: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def create_agent(self, project_id: str, role: AgentRole) -> AgentState:
        """Create new agent state.
        
        Args:
            project_id: Project ID
            role: Agent role
            
        Returns:
            Created agent state
        """
        try:
            # Validate project exists
            if project_id not in self._states:
                raise ValueError(f"Project {project_id} not found")
                
            # Create agent state
            agent_state = AgentState(
                id=f"agent_{datetime.utcnow().timestamp()}",
                agent=role,
                status=Status.RUNNING,
                context={"project_id": project_id}
            )
            
            # Add state
            await self.add_state(agent_state)
            logger.info(f"Created agent: {agent_state.id}")
            
            return agent_state
            
        except Exception as e:
            error_msg = f"Failed to create agent: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def create_graph(self, project_id: str) -> GraphState:
        """Create new graph state.
        
        Args:
            project_id: Project ID
            
        Returns:
            Created graph state
        """
        try:
            # Validate project exists
            if project_id not in self._states:
                raise ValueError(f"Project {project_id} not found")
                
            # Create graph state
            graph_state = GraphState(
                id=f"graph_{datetime.utcnow().timestamp()}",
                status=Status.RUNNING,
                metadata={"project_id": project_id},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Add state
            await self.add_state(graph_state)
            logger.info(f"Created graph: {graph_state.id}")
            
            return graph_state
            
        except Exception as e:
            error_msg = f"Failed to create graph: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def get_project_chats(self, project_id: str) -> List[ChatState]:
        """Get all chats for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            List of chat states
        """
        try:
            return [
                state for state in self._states.values()
                if isinstance(state, ChatState) and state.context.get("project_id") == project_id
            ]
        except Exception as e:
            error_msg = f"Failed to get project chats: {str(e)}"
            logger.error(error_msg)
            return []

    async def get_project_agents(self, project_id: str) -> List[AgentState]:
        """Get all agents for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            List of agent states
        """
        try:
            return [
                state for state in self._states.values()
                if isinstance(state, AgentState) and state.context.get("project_id") == project_id
            ]
        except Exception as e:
            error_msg = f"Failed to get project agents: {str(e)}"
            logger.error(error_msg)
            return []

    async def get_project_graphs(self, project_id: str) -> List[GraphState]:
        """Get all graphs for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            List of graph states
        """
        try:
            return [
                state for state in self._states.values()
                if isinstance(state, GraphState) and state.metadata.get("project_id") == project_id
            ]
        except Exception as e:
            error_msg = f"Failed to get project graphs: {str(e)}"
            logger.error(error_msg)
            return []

    async def get_project_state(self, project_id: str) -> Optional[ProjectState]:
        """Get project state by ID.
        
        Args:
            project_id: Project ID
            
        Returns:
            Project state if found, None otherwise
        """
        try:
            # Look for existing project state
            for state in self._states.values():
                if isinstance(state, ProjectState) and state.id == project_id:
                    return state
            
            # If not found, create a default project state
            project_state = ProjectState(
                id=project_id,
                project_name=f"Project {project_id}",
                description=f"Auto-created project for {project_id}",
                graph_state=GraphState(
                    id=f"graph_{project_id}",
                    status=Status.PENDING
                ),
                status=Status.RUNNING
            )
            
            await self.add_state(project_state)
            return project_state
            
        except Exception as e:
            error_msg = f"Failed to get project state {project_id}: {str(e)}"
            logger.error(error_msg)
            return None
