"""Memory Management System for Code Hero.

This module implements comprehensive memory management including:
- Short-term memory: Thread-level conversation tracking with checkpointing
- Long-term memory: Cross-thread user and application data storage
- Memory tools: Prebuilt tools for reading/writing memory
- Semantic search: Vector-based memory retrieval
- Message history management: Summarization and trimming

Based on LangGraph memory documentation and best practices.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Updated imports for LangGraph v0.4.7+ following official documentation
from langgraph.checkpoint.memory import MemorySaver as _MemorySaver  # Import with alias
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

try:
    from langgraph.config import get_store
    from langgraph.prebuilt import InjectedState, InjectedToolCallId
    from langmem.short_term import SummarizationNode

    LANGGRAPH_AVAILABLE = True
    LANGMEM_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    LANGMEM_AVAILABLE = False

    # Fallback implementations
    class MemorySaver:
        def __init__(self):
            self.storage = {}

    class InMemoryStore:
        def __init__(self):
            self.storage = {}

        def put(self, namespace: tuple, key: str, value: Any):
            if namespace not in self.storage:
                self.storage[namespace] = {}
            self.storage[namespace][key] = {
                "value": value,
                "created_at": datetime.now(),
            }

        def get(self, namespace: tuple, key: str):
            if namespace in self.storage and key in self.storage[namespace]:
                return self.storage[namespace][key]
            return None

        def search(self, namespace: tuple, query: str = None, limit: int = 10):
            if namespace not in self.storage:
                return []
            items = list(self.storage[namespace].items())
            return [{"key": k, "value": v["value"]} for k, v in items[:limit]]

    def get_store():
        return None

    class InjectedState:
        pass

    class InjectedToolCallId:
        pass

    class SummarizationNode:
        def __init__(self, **kwargs):
            pass


from .config import get_model_for_agent
from .logger import StructuredLogger
from .state import AgentRole
from .utils import generate_id

logger = logging.getLogger(__name__)


# === MEMORY MODELS ===


class UserInfo(TypedDict):
    """User information for long-term memory."""

    name: str
    email: Optional[str]
    preferences: Dict[str, Any]
    language: str
    timezone: Optional[str]
    created_at: str
    updated_at: str


class ConversationSummary(BaseModel):
    """Conversation summary for short-term memory management."""

    summary: str = Field(description="Summary of the conversation")
    key_points: List[str] = Field(description="Key points from the conversation")
    context: Dict[str, Any] = Field(description="Important context to preserve")
    token_count: int = Field(description="Approximate token count of summary")
    created_at: datetime = Field(default_factory=datetime.now)


class MemoryItem(BaseModel):
    """Generic memory item for long-term storage."""

    id: str = Field(default_factory=lambda: generate_id("mem"))
    content: Any = Field(description="Memory content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


class MemorySearchResult(BaseModel):
    """Memory search result."""

    item: MemoryItem
    score: float = Field(description="Relevance score")
    snippet: str = Field(description="Relevant snippet")


# === MEMORY MANAGER ===


class CodeHeroMemoryManager:
    """Comprehensive memory management for Code Hero agents."""

    def __init__(
        self,
        checkpointer: Optional[MemorySaver] = None,
        store: Optional[InMemoryStore] = None,
        embeddings: Optional[OpenAIEmbeddings] = None,
        logger_service: Optional[StructuredLogger] = None,
        enable_summarization: bool = True,
        max_conversation_tokens: int = 4000,
        max_summary_tokens: int = 500,
    ):
        """Initialize memory manager.

        Args:
            checkpointer: LangGraph checkpointer for short-term memory
            store: LangGraph store for long-term memory
            embeddings: Embeddings for semantic search
            logger_service: Structured logger
            enable_summarization: Whether to enable conversation summarization
            max_conversation_tokens: Max tokens before summarization
            max_summary_tokens: Max tokens for summaries
        """
        # Initialize checkpointer with compatibility handling
        if checkpointer:
            self.checkpointer = checkpointer
        else:
            try:
                self.checkpointer = MemorySaver()
            except Exception as e:
                logger.warning(f"Failed to initialize MemorySaver: {e}")
                # Create a minimal fallback checkpointer
                self.checkpointer = None

        self.store = store or InMemoryStore()
        self.embeddings = embeddings
        self.logger = logger_service
        self.enable_summarization = enable_summarization
        self.max_conversation_tokens = max_conversation_tokens
        self.max_summary_tokens = max_summary_tokens

        # Initialize summarization if available
        self.summarization_node = None
        if LANGMEM_AVAILABLE and enable_summarization:
            try:
                model_config = get_model_for_agent(AgentRole.SUPERVISOR)
                model = ChatOpenAI(
                    model=model_config.get("model", "gpt-4o-mini"),
                    temperature=0.1,
                )

                self.summarization_node = SummarizationNode(
                    token_counter=count_tokens_approximately,
                    model=model,
                    max_tokens=max_conversation_tokens,
                    max_summary_tokens=max_summary_tokens,
                    output_messages_key="llm_input_messages",
                )
                logger.info("Conversation summarization enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize summarization: {e}")

        # Memory namespaces
        self.USER_NAMESPACE = ("users",)
        self.CONVERSATION_NAMESPACE = ("conversations",)
        self.AGENT_NAMESPACE = ("agents",)
        self.PROJECT_NAMESPACE = ("projects",)
        self.KNOWLEDGE_NAMESPACE = ("knowledge",)

        logger.info("Memory manager initialized successfully")

    # === SHORT-TERM MEMORY (Thread-level) ===

    def get_checkpointer(self) -> Optional[MemorySaver]:
        """Get the checkpointer for short-term memory.

        Returns:
            Checkpointer instance or None if not available due to compatibility issues
        """
        return self.checkpointer

    def create_thread_config(self, thread_id: str, **kwargs) -> Dict[str, Any]:
        """Create configuration for thread-level memory.

        Args:
            thread_id: Unique thread identifier
            **kwargs: Additional configuration

        Returns:
            Configuration dict for LangGraph
        """
        return {"configurable": {"thread_id": thread_id, **kwargs}}

    def manage_message_history(
        self,
        messages: List[BaseMessage],
        strategy: Literal["trim", "summarize"] = "summarize",
    ) -> List[BaseMessage]:
        """Manage message history to prevent context overflow.

        Args:
            messages: List of messages
            strategy: Management strategy ("trim" or "summarize")

        Returns:
            Managed message list
        """
        try:
            # Count tokens in current messages
            token_count = count_tokens_approximately(messages)

            if token_count <= self.max_conversation_tokens:
                return messages

            if strategy == "trim":
                return self._trim_messages(messages)
            elif strategy == "summarize" and self.summarization_node:
                return self._summarize_messages(messages)
            else:
                # Fallback to trimming
                return self._trim_messages(messages)

        except Exception as e:
            logger.error(f"Message history management failed: {e}")
            return messages

    def _trim_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Trim messages using LangChain's trim_messages."""
        try:
            return trim_messages(
                messages,
                strategy="last",
                token_counter=count_tokens_approximately,
                max_tokens=self.max_conversation_tokens,
                start_on="human",
                end_on=("human", "tool"),
            )
        except Exception as e:
            logger.error(f"Message trimming failed: {e}")
            # Return last few messages as fallback
            return messages[-10:] if len(messages) > 10 else messages

    def _summarize_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Summarize messages using LangMem summarization."""
        try:
            if not self.summarization_node:
                return self._trim_messages(messages)

            # Create state for summarization
            state = {"messages": messages, "context": {}}

            # Apply summarization
            result = self.summarization_node(state)

            # Return summarized messages
            return result.get("llm_input_messages", messages)

        except Exception as e:
            logger.error(f"Message summarization failed: {e}")
            return self._trim_messages(messages)

    # === LONG-TERM MEMORY (Cross-thread) ===

    async def store_user_info(self, user_id: str, user_info: UserInfo) -> bool:
        """Store user information in long-term memory.

        Args:
            user_id: User identifier
            user_info: User information

        Returns:
            Success status
        """
        try:
            user_info["updated_at"] = datetime.now().isoformat()
            self.store.put(self.USER_NAMESPACE, user_id, user_info)

            if self.logger:
                await self.logger.log_info(f"Stored user info for {user_id}")

            return True
        except Exception as e:
            logger.error(f"Failed to store user info: {e}")
            return False

    async def get_user_info(self, user_id: str) -> Optional[UserInfo]:
        """Retrieve user information from long-term memory.

        Args:
            user_id: User identifier

        Returns:
            User information if found
        """
        try:
            result = self.store.get(self.USER_NAMESPACE, user_id)
            return result.value if result else None
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return None

    async def store_conversation_summary(
        self, conversation_id: str, summary: ConversationSummary
    ) -> bool:
        """Store conversation summary in long-term memory.

        Args:
            conversation_id: Conversation identifier
            summary: Conversation summary

        Returns:
            Success status
        """
        try:
            summary_dict = summary.model_dump()
            summary_dict["created_at"] = summary.created_at.isoformat()

            self.store.put(self.CONVERSATION_NAMESPACE, conversation_id, summary_dict)

            if self.logger:
                await self.logger.log_info(
                    f"Stored conversation summary for {conversation_id}"
                )

            return True
        except Exception as e:
            logger.error(f"Failed to store conversation summary: {e}")
            return False

    async def get_conversation_summary(
        self, conversation_id: str
    ) -> Optional[ConversationSummary]:
        """Retrieve conversation summary from long-term memory.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Conversation summary if found
        """
        try:
            result = self.store.get(self.CONVERSATION_NAMESPACE, conversation_id)
            if result:
                summary_dict = result.value
                summary_dict["created_at"] = datetime.fromisoformat(
                    summary_dict["created_at"]
                )
                return ConversationSummary(**summary_dict)
            return None
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return None

    async def store_memory_item(
        self,
        namespace: tuple,
        key: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
    ) -> bool:
        """Store a memory item in long-term memory.

        Args:
            namespace: Memory namespace
            key: Memory key
            content: Memory content
            metadata: Additional metadata
            tags: Memory tags for categorization
            expires_at: Expiration time

        Returns:
            Success status
        """
        try:
            memory_item = MemoryItem(
                content=content,
                metadata=metadata or {},
                tags=tags or [],
                expires_at=expires_at,
            )

            item_dict = memory_item.model_dump()
            item_dict["created_at"] = memory_item.created_at.isoformat()
            item_dict["updated_at"] = memory_item.updated_at.isoformat()
            if expires_at:
                item_dict["expires_at"] = expires_at.isoformat()

            self.store.put(namespace, key, item_dict)

            if self.logger:
                await self.logger.log_info(f"Stored memory item: {namespace}/{key}")

            return True
        except Exception as e:
            logger.error(f"Failed to store memory item: {e}")
            return False

    async def get_memory_item(self, namespace: tuple, key: str) -> Optional[MemoryItem]:
        """Retrieve a memory item from long-term memory.

        Args:
            namespace: Memory namespace
            key: Memory key

        Returns:
            Memory item if found and not expired
        """
        try:
            result = self.store.get(namespace, key)
            if not result:
                return None

            item_dict = result.value

            # Check expiration
            if item_dict.get("expires_at"):
                expires_at = datetime.fromisoformat(item_dict["expires_at"])
                if datetime.now() > expires_at:
                    # Item expired, remove it
                    await self.delete_memory_item(namespace, key)
                    return None

            # Convert datetime strings back to datetime objects
            item_dict["created_at"] = datetime.fromisoformat(item_dict["created_at"])
            item_dict["updated_at"] = datetime.fromisoformat(item_dict["updated_at"])
            if item_dict.get("expires_at"):
                item_dict["expires_at"] = datetime.fromisoformat(
                    item_dict["expires_at"]
                )

            return MemoryItem(**item_dict)
        except Exception as e:
            logger.error(f"Failed to get memory item: {e}")
            return None

    async def delete_memory_item(self, namespace: tuple, key: str) -> bool:
        """Delete a memory item from long-term memory.

        Args:
            namespace: Memory namespace
            key: Memory key

        Returns:
            Success status
        """
        try:
            # Note: InMemoryStore doesn't have a delete method in the fallback
            # In a real implementation, this would delete from the store
            if hasattr(self.store, "delete"):
                self.store.delete(namespace, key)
            elif hasattr(self.store, "storage"):
                # Fallback for our mock implementation
                if (
                    namespace in self.store.storage
                    and key in self.store.storage[namespace]
                ):
                    del self.store.storage[namespace][key]

            if self.logger:
                await self.logger.log_info(f"Deleted memory item: {namespace}/{key}")

            return True
        except Exception as e:
            logger.error(f"Failed to delete memory item: {e}")
            return False

    async def search_memory(
        self,
        namespace: tuple,
        query: str = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[MemorySearchResult]:
        """Search memory items in a namespace.

        Args:
            namespace: Memory namespace
            query: Search query
            tags: Filter by tags
            limit: Maximum results

        Returns:
            List of search results
        """
        try:
            # Basic search implementation
            # In a real implementation, this would use vector search
            results = []

            if hasattr(self.store, "search"):
                items = self.store.search(namespace, query, limit)
            elif hasattr(self.store, "storage"):
                # Fallback for our mock implementation
                if namespace not in self.store.storage:
                    return []

                items = []
                for key, value in self.store.storage[namespace].items():
                    items.append({"key": key, "value": value["value"]})

                items = items[:limit]
            else:
                return []

            for item in items:
                try:
                    memory_item = MemoryItem(**item["value"])

                    # Filter by tags if specified
                    if tags and not any(tag in memory_item.tags for tag in tags):
                        continue

                    # Calculate relevance score (simplified)
                    score = 1.0
                    if query:
                        content_str = str(memory_item.content).lower()
                        query_lower = query.lower()
                        if query_lower in content_str:
                            score = 0.9
                        else:
                            score = 0.1

                    # Create snippet
                    content_str = str(memory_item.content)
                    snippet = (
                        content_str[:200] + "..."
                        if len(content_str) > 200
                        else content_str
                    )

                    results.append(
                        MemorySearchResult(
                            item=memory_item, score=score, snippet=snippet
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to process search result: {e}")
                    continue

            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            return results

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    # === SEMANTIC SEARCH ===

    async def semantic_search_memory(
        self,
        namespace: tuple,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[MemorySearchResult]:
        """Perform semantic search on memory items.

        Args:
            namespace: Memory namespace
            query: Search query
            limit: Maximum results
            similarity_threshold: Minimum similarity score

        Returns:
            List of semantically similar results
        """
        try:
            if not self.embeddings:
                logger.warning("Embeddings not available, falling back to basic search")
                return await self.search_memory(namespace, query, limit=limit)

            # Get query embedding
            await self.embeddings.aembed_query(query)

            # This is a simplified implementation
            # In a real system, you'd use a vector database
            basic_results = await self.search_memory(namespace, query, limit=limit * 2)

            # Filter by similarity threshold and limit
            semantic_results = []
            for result in basic_results:
                if result.score >= similarity_threshold:
                    semantic_results.append(result)

                if len(semantic_results) >= limit:
                    break

            return semantic_results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return await self.search_memory(namespace, query, limit=limit)

    # === MEMORY CLEANUP ===

    async def cleanup_expired_memories(self) -> int:
        """Clean up expired memory items.

        Returns:
            Number of items cleaned up
        """
        try:
            cleaned_count = 0
            current_time = datetime.now()

            # This is a simplified implementation
            # In a real system, you'd iterate through all namespaces
            for namespace_name in [
                "users",
                "conversations",
                "agents",
                "projects",
                "knowledge",
            ]:
                namespace = (namespace_name,)

                if hasattr(self.store, "storage") and namespace in self.store.storage:
                    expired_keys = []

                    for key, value in self.store.storage[namespace].items():
                        try:
                            item_dict = value["value"]
                            if item_dict.get("expires_at"):
                                expires_at = datetime.fromisoformat(
                                    item_dict["expires_at"]
                                )
                                if current_time > expires_at:
                                    expired_keys.append(key)
                        except Exception as e:
                            logger.warning(
                                f"Failed to check expiration for {namespace}/{key}: {e}"
                            )

                    # Delete expired items
                    for key in expired_keys:
                        await self.delete_memory_item(namespace, key)
                        cleaned_count += 1

            if cleaned_count > 0 and self.logger:
                await self.logger.log_info(
                    f"Cleaned up {cleaned_count} expired memory items"
                )

            return cleaned_count

        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return 0

    # === MEMORY STATISTICS ===

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics.

        Returns:
            Memory statistics
        """
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "namespaces": {},
                "total_items": 0,
                "checkpointer_type": type(self.checkpointer).__name__,
                "store_type": type(self.store).__name__,
                "summarization_enabled": self.summarization_node is not None,
                "embeddings_enabled": self.embeddings is not None,
            }

            # Count items by namespace
            if hasattr(self.store, "storage"):
                for namespace, items in self.store.storage.items():
                    namespace_name = namespace[0] if namespace else "unknown"
                    item_count = len(items)
                    stats["namespaces"][namespace_name] = item_count
                    stats["total_items"] += item_count

            return stats

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}


# === MEMORY TOOLS ===


def create_memory_tools(memory_manager: CodeHeroMemoryManager) -> List[Any]:
    """Create memory tools for agents.

    Args:
        memory_manager: Memory manager instance

    Returns:
        List of memory tools
    """
    tools = []

    @tool
    async def get_user_info(config: RunnableConfig) -> str:
        """Look up user information from long-term memory."""
        try:
            user_id = config["configurable"].get("user_id")
            if not user_id:
                return "No user ID provided in configuration"

            user_info = await memory_manager.get_user_info(user_id)
            if user_info:
                return f"User: {user_info.get('name', 'Unknown')}, Language: {user_info.get('language', 'English')}, Preferences: {user_info.get('preferences', {})}"
            else:
                return "User information not found"
        except Exception as e:
            return f"Error retrieving user info: {str(e)}"

    @tool
    async def save_user_info(user_info: UserInfo, config: RunnableConfig) -> str:
        """Save user information to long-term memory."""
        try:
            user_id = config["configurable"].get("user_id")
            if not user_id:
                return "No user ID provided in configuration"

            success = await memory_manager.store_user_info(user_id, user_info)
            return (
                "Successfully saved user info"
                if success
                else "Failed to save user info"
            )
        except Exception as e:
            return f"Error saving user info: {str(e)}"

    @tool
    async def search_memory(
        namespace: str, query: str, tags: Optional[List[str]] = None, limit: int = 5
    ) -> str:
        """Search memory items by query and tags."""
        try:
            namespace_tuple = (namespace,)
            results = await memory_manager.search_memory(
                namespace_tuple, query, tags, limit
            )

            if not results:
                return f"No memory items found for query: {query}"

            response = f"Found {len(results)} memory items:\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. {result.snippet} (Score: {result.score:.2f})\n"

            return response
        except Exception as e:
            return f"Error searching memory: {str(e)}"

    @tool
    async def store_memory(
        namespace: str,
        key: str,
        content: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store information in long-term memory."""
        try:
            namespace_tuple = (namespace,)
            success = await memory_manager.store_memory_item(
                namespace_tuple, key, content, metadata, tags
            )
            return (
                f"Successfully stored memory item: {key}"
                if success
                else f"Failed to store memory item: {key}"
            )
        except Exception as e:
            return f"Error storing memory: {str(e)}"

    @tool
    async def get_conversation_context(conversation_id: str) -> str:
        """Get conversation context from memory."""
        try:
            summary = await memory_manager.get_conversation_summary(conversation_id)
            if summary:
                return f"Conversation Summary: {summary.summary}\nKey Points: {', '.join(summary.key_points)}"
            else:
                return "No conversation context found"
        except Exception as e:
            return f"Error retrieving conversation context: {str(e)}"

    tools.extend(
        [
            get_user_info,
            save_user_info,
            search_memory,
            store_memory,
            get_conversation_context,
        ]
    )

    return tools


# === MEMORY HOOKS ===


def create_memory_hooks(memory_manager: CodeHeroMemoryManager):
    """Create memory management hooks for agents.

    Args:
        memory_manager: Memory manager instance

    Returns:
        Dictionary of memory hooks
    """

    def pre_model_hook(state: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-model hook for message history management."""
        try:
            messages = state.get("messages", [])
            if not messages:
                return state

            # Manage message history
            managed_messages = memory_manager.manage_message_history(messages)

            return {**state, "llm_input_messages": managed_messages}
        except Exception as e:
            logger.error(f"Pre-model hook failed: {e}")
            return state

    def post_model_hook(state: Dict[str, Any]) -> Dict[str, Any]:
        """Post-model hook for storing conversation summaries."""
        try:
            # This would typically run after model completion
            # to store important conversation context
            conversation_id = state.get("conversation_id")
            if conversation_id and len(state.get("messages", [])) > 10:
                # Create and store summary asynchronously
                # In a real implementation, this would be handled properly
                pass

            return state
        except Exception as e:
            logger.error(f"Post-model hook failed: {e}")
            return state

    return {
        "pre_model_hook": pre_model_hook,
        "post_model_hook": post_model_hook,
    }


# === MEMORY INTEGRATION ===


async def initialize_memory_system(
    enable_checkpointing: bool = True,
    enable_long_term_memory: bool = True,
    enable_embeddings: bool = True,
    logger_service: Optional[StructuredLogger] = None,
) -> CodeHeroMemoryManager:
    """Initialize the complete memory system.

    Args:
        enable_checkpointing: Enable short-term memory checkpointing
        enable_long_term_memory: Enable long-term memory store
        enable_embeddings: Enable semantic search with embeddings
        logger_service: Structured logger

    Returns:
        Configured memory manager
    """
    try:
        # Initialize checkpointer
        checkpointer = None
        if enable_checkpointing and LANGGRAPH_AVAILABLE:
            checkpointer = MemorySaver()
            logger.info("Short-term memory checkpointing enabled")

        # Initialize store
        store = None
        if enable_long_term_memory:
            store = InMemoryStore()
            logger.info("Long-term memory store enabled")

        # Initialize embeddings
        embeddings = None
        if enable_embeddings:
            try:
                embeddings = OpenAIEmbeddings()
                logger.info("Semantic search embeddings enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize embeddings: {e}")

        # Create memory manager
        memory_manager = CodeHeroMemoryManager(
            checkpointer=checkpointer,
            store=store,
            embeddings=embeddings,
            logger_service=logger_service,
        )

        logger.info("Memory system initialized successfully")
        return memory_manager

    except Exception as e:
        logger.error(f"Failed to initialize memory system: {e}")
        raise


# === EXPORTS ===

__all__ = [
    # Core classes
    "CodeHeroMemoryManager",
    "UserInfo",
    "ConversationSummary",
    "MemoryItem",
    "MemorySearchResult",
    # Tools and hooks
    "create_memory_tools",
    "create_memory_hooks",
    # Initialization
    "initialize_memory_system",
    # LangGraph components (if available)
    "MemorySaver",
    "InMemoryStore",
    "LANGGRAPH_AVAILABLE",
    "LANGMEM_AVAILABLE",
]


# Compatibility wrapper for MemorySaver
class CompatibleMemorySaver(_MemorySaver):
    """Compatibility wrapper for MemorySaver to handle config_specs attribute."""

    @property
    def config_specs(self):
        """Provide config_specs for backward compatibility."""
        return []  # Safe default for older implementations

    def __init__(self, *args, **kwargs):
        """Initialize with error handling."""
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            # If initialization fails, create a minimal fallback
            self.storage = {}


# Use the compatible version
MemorySaver = CompatibleMemorySaver
