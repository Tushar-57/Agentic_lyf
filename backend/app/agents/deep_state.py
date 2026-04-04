"""
Deep Agent State Management System
=================================

State management for Deep Agents with file-based context offloading,
TODO management, and isolated agent contexts.
"""

from typing import Dict, List, Any, Optional, Union
from typing_extensions import TypedDict
from datetime import datetime
import uuid
from enum import Enum
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage
import json


class TodoStatus(str, Enum):
    """Status of a TODO item."""
    NOT_STARTED = "not-started"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TodoPriority(str, Enum):
    """Priority levels for TODO items."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Todo:
    """A TODO item for task management and planning."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    status: TodoStatus = TodoStatus.NOT_STARTED
    priority: TodoPriority = TodoPriority.MEDIUM
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    estimated_duration: Optional[int] = None  # minutes
    actual_duration: Optional[int] = None
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "assigned_agent": self.assigned_agent,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Todo':
        """Create Todo from dictionary."""
        todo = cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", ""),
            description=data.get("description", ""),
            status=TodoStatus(data.get("status", TodoStatus.NOT_STARTED.value)),
            priority=TodoPriority(data.get("priority", TodoPriority.MEDIUM.value)),
            assigned_agent=data.get("assigned_agent"),
            dependencies=data.get("dependencies", []),
            tags=data.get("tags", []),
            estimated_duration=data.get("estimated_duration"),
            actual_duration=data.get("actual_duration"),
            notes=data.get("notes", [])
        )
        
        # Parse datetime fields
        if data.get("created_at"):
            todo.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            todo.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("due_date"):
            todo.due_date = datetime.fromisoformat(data["due_date"])
            
        return todo


class AgentExecutionStatus(str, Enum):
    """Status of agent execution."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentContext:
    """Context for individual agent execution."""
    agent_id: str
    agent_type: str
    status: AgentExecutionStatus = AgentExecutionStatus.IDLE
    current_task: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    todos_assigned: List[str] = field(default_factory=list)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)


class HumanApprovalRequest:
    """Request for human approval."""
    def __init__(
        self,
        request_id: str,
        agent_id: str,
        action_type: str,
        description: str,
        details: Dict[str, Any],
        priority: str = "medium",
        timeout_minutes: int = 60
    ):
        self.request_id = request_id
        self.agent_id = agent_id
        self.action_type = action_type
        self.description = description
        self.details = details
        self.priority = priority
        self.created_at = datetime.now()
        self.timeout_minutes = timeout_minutes
        self.status = "pending"  # pending, approved, rejected, expired
        self.user_response: Optional[Dict[str, Any]] = None


class DeepAgentState(TypedDict):
    """
    State structure for Deep Agents system.
    
    This extends the basic agent state with deep agent capabilities:
    - File-based context storage
    - TODO management
    - Agent coordination
    - Human-in-the-loop workflows
    """
    # Core messaging
    messages: List[BaseMessage]
    
    # File-based context storage (key: filename, value: content)
    files: Dict[str, str]
    
    # TODO management
    todos: List[Dict[str, Any]]  # Serialized Todo objects
    
    # Agent coordination
    current_agent: Optional[str]
    agent_contexts: Dict[str, Dict[str, Any]]  # agent_id -> AgentContext dict
    
    # Human-in-the-loop
    approval_requests: List[Dict[str, Any]]  # Pending approval requests
    user_preferences: Dict[str, Any]
    
    # Session management
    session_id: str
    conversation_id: str
    
    # Context and metadata
    context: Dict[str, Any]
    metadata: Dict[str, Any]


class DeepAgentStateManager:
    """
    Manager for Deep Agent State operations.
    
    Provides high-level operations for state management including:
    - File storage and retrieval
    - TODO management
    - Agent context tracking
    - Human approval workflows
    """
    
    def __init__(self, initial_state: Optional[DeepAgentState] = None):
        self.state = initial_state or self._create_empty_state()
        self._conversation_managers: Dict[str, 'DeepAgentStateManager'] = {}
    
    def get_or_create_state(self, conversation_id: str) -> 'DeepAgentStateManager':
        """Get or create state manager for a specific conversation."""
        if conversation_id not in self._conversation_managers:
            state = self._create_empty_state()
            state["conversation_id"] = conversation_id
            self._conversation_managers[conversation_id] = DeepAgentStateManager(state)
        return self._conversation_managers[conversation_id]
    
    def _create_empty_state(self) -> DeepAgentState:
        """Create an empty state structure."""
        return DeepAgentState(
            messages=[],
            files={},
            todos=[],
            current_agent=None,
            agent_contexts={},
            approval_requests=[],
            user_preferences={},
            session_id=str(uuid.uuid4()),
            conversation_id=str(uuid.uuid4()),
            context={},
            metadata={}
        )
    
    # Message operations
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the state."""
        from langchain_core.messages import HumanMessage, AIMessage
        
        if role == "user":
            message = HumanMessage(content=content)
        else:
            message = AIMessage(content=content)
        
        self.state["messages"].append(message)
    
    # File operations
    def store_file(self, filename: str, content: str) -> None:
        """Store content in a file."""
        self.state["files"][filename] = content
    
    def get_file(self, filename: str) -> Optional[str]:
        """Retrieve file content."""
        return self.state["files"].get(filename)
    
    def list_files(self) -> List[str]:
        """List all stored files."""
        return list(self.state["files"].keys())
    
    def delete_file(self, filename: str) -> bool:
        """Delete a file."""
        if filename in self.state["files"]:
            del self.state["files"][filename]
            return True
        return False
    
    # TODO operations
    def add_todo(self, todo: Todo) -> None:
        """Add a new TODO item."""
        self.state["todos"].append(todo.to_dict())
    
    def get_todos(self, status: Optional[TodoStatus] = None, 
                  agent: Optional[str] = None) -> List[Todo]:
        """Get TODO items, optionally filtered by status or agent."""
        todos = [Todo.from_dict(todo_dict) for todo_dict in self.state["todos"]]
        
        if status:
            todos = [todo for todo in todos if todo.status == status]
        
        if agent:
            todos = [todo for todo in todos if todo.assigned_agent == agent]
        
        return todos
    
    def update_todo(self, todo_id: str, updates: Dict[str, Any]) -> bool:
        """Update a TODO item."""
        for todo_dict in self.state["todos"]:
            if todo_dict["id"] == todo_id:
                todo_dict.update(updates)
                todo_dict["updated_at"] = datetime.now().isoformat()
                return True
        return False
    
    def complete_todo(self, todo_id: str) -> bool:
        """Mark a TODO as completed."""
        return self.update_todo(todo_id, {"status": TodoStatus.COMPLETED.value})
    
    # Agent context operations
    def set_current_agent(self, agent_id: str) -> None:
        """Set the currently active agent."""
        self.state["current_agent"] = agent_id
        self._ensure_agent_context(agent_id)
    
    def _ensure_agent_context(self, agent_id: str) -> None:
        """Ensure agent context exists."""
        if agent_id not in self.state["agent_contexts"]:
            context = AgentContext(agent_id=agent_id, agent_type="unknown")
            self.state["agent_contexts"][agent_id] = {
                "agent_id": context.agent_id,
                "agent_type": context.agent_type,
                "status": context.status.value,
                "current_task": context.current_task,
                "tools_used": context.tools_used,
                "files_created": context.files_created,
                "todos_assigned": context.todos_assigned,
                "execution_history": context.execution_history,
                "error_count": context.error_count,
                "last_activity": context.last_activity.isoformat()
            }
    
    def update_agent_status(self, agent_id: str, status: AgentExecutionStatus) -> None:
        """Update agent execution status."""
        self._ensure_agent_context(agent_id)
        self.state["agent_contexts"][agent_id]["status"] = status.value
        self.state["agent_contexts"][agent_id]["last_activity"] = datetime.now().isoformat()
    
    # Human approval operations
    def request_approval(self, request: HumanApprovalRequest) -> None:
        """Request human approval for an action."""
        self.state["approval_requests"].append({
            "request_id": request.request_id,
            "agent_id": request.agent_id,
            "action_type": request.action_type,
            "description": request.description,
            "details": request.details,
            "priority": request.priority,
            "created_at": request.created_at.isoformat(),
            "timeout_minutes": request.timeout_minutes,
            "status": request.status
        })
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        return [req for req in self.state["approval_requests"] if req["status"] == "pending"]
    
    def approve_request(self, request_id: str, user_response: Dict[str, Any]) -> bool:
        """Approve a request."""
        for req in self.state["approval_requests"]:
            if req["request_id"] == request_id:
                req["status"] = "approved"
                req["user_response"] = user_response
                return True
        return False
    
    def reject_request(self, request_id: str, reason: str) -> bool:
        """Reject a request."""
        for req in self.state["approval_requests"]:
            if req["request_id"] == request_id:
                req["status"] = "rejected"
                req["user_response"] = {"reason": reason}
                return True
        return False
    
    # Utility methods
    def get_state(self) -> DeepAgentState:
        """Get the current state."""
        return self.state
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "messages": [msg.content if hasattr(msg, 'content') else str(msg) for msg in self.state["messages"]],
            "files": self.state["files"],
            "todos": self.state["todos"],
            "current_agent": self.state["current_agent"],
            "agent_contexts": self.state["agent_contexts"],
            "approval_requests": self.state["approval_requests"],
            "user_preferences": self.state["user_preferences"],
            "session_id": self.state["session_id"],
            "conversation_id": self.state["conversation_id"],
            "context": self.state["context"],
            "metadata": self.state["metadata"]
        }
    
    def update_state(self, conversation_id: str, state_manager: 'DeepAgentStateManager') -> None:
        """Update state for a conversation (no-op since managers are isolated)."""
        # This is a no-op because each conversation already has its own manager instance
        # The state is automatically persisted in the conversation-specific manager
        pass
    
    def update_context(self, key: str, value: Any) -> None:
        """Update context data."""
        self.state["context"][key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context data."""
        return self.state["context"].get(key, default)
    
    def export_state(self) -> str:
        """Export state as JSON string."""
        # Convert messages to serializable format
        serializable_state = self.state.copy()
        serializable_state["messages"] = [
            {"type": msg.__class__.__name__, "content": msg.content}
            for msg in self.state["messages"]
        ]
        return json.dumps(serializable_state, indent=2, default=str)