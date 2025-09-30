"""
TODO Management Tools for Deep Agents
=====================================

Tools for planning, task breakdown, and progress tracking using TODO lists.
Enables agents to manage complex, multi-step workflows systematically.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId
from typing_extensions import Annotated

from .deep_state import DeepAgentState, Todo, TodoStatus, TodoPriority


@tool(parse_docstring=True)
def create_todo(
    title: str,
    description: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    priority: str = "medium",
    assigned_agent: Optional[str] = None,
    due_date: Optional[str] = None,
    estimated_duration: Optional[int] = None,
    tags: Optional[List[str]] = None,
    dependencies: Optional[List[str]] = None
) -> Command:
    """Create a new TODO item for task planning and tracking.
    
    Use this tool to break down complex tasks into manageable items and
    track progress systematically. Essential for planning workflows.
    
    Args:
        title: Short, descriptive title for the task
        description: Detailed description of what needs to be done
        priority: Task priority - "low", "medium", "high", or "urgent"
        assigned_agent: Agent responsible for this task (optional)
        due_date: Due date in ISO format (YYYY-MM-DD) (optional)
        estimated_duration: Estimated time in minutes (optional)
        tags: List of tags for categorization (optional)
        dependencies: List of TODO IDs this task depends on (optional)
    
    Returns:
        Command that adds the TODO and returns confirmation with ID
    """
    # Validate priority
    try:
        priority_enum = TodoPriority(priority.lower())
    except ValueError:
        priority_enum = TodoPriority.MEDIUM
    
    # Parse due date
    parsed_due_date = None
    if due_date:
        try:
            parsed_due_date = datetime.fromisoformat(due_date)
        except ValueError:
            pass
    
    # Create new TODO
    todo = Todo(
        title=title,
        description=description,
        priority=priority_enum,
        assigned_agent=assigned_agent or state.get("current_agent"),
        due_date=parsed_due_date,
        estimated_duration=estimated_duration,
        tags=tags or [],
        dependencies=dependencies or []
    )
    
    # Add to state
    todos = state.get("todos", [])
    todos.append(todo.to_dict())
    
    # Update agent context if assigned
    agent_contexts = state.get("agent_contexts", {})
    if todo.assigned_agent and todo.assigned_agent in agent_contexts:
        agent_contexts[todo.assigned_agent]["todos_assigned"].append(todo.id)
    
    return Command(
        update={
            "todos": todos,
            "agent_contexts": agent_contexts,
            "messages": [
                ToolMessage(
                    content=f"✅ TODO created successfully!\n\n"
                           f"**ID:** {todo.id}\n"
                           f"**Title:** {todo.title}\n"
                           f"**Priority:** {todo.priority.value}\n"
                           f"**Assigned to:** {todo.assigned_agent or 'Unassigned'}\n"
                           f"**Due:** {due_date or 'No due date'}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def list_todos(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    status: Optional[str] = None,
    assigned_agent: Optional[str] = None,
    priority: Optional[str] = None,
    show_completed: bool = False
) -> Command:
    """List TODO items with optional filtering.
    
    Use this tool to review current tasks, check progress, and plan next steps.
    Essential for maintaining overview of work and priorities.
    
    Args:
        status: Filter by status - "not-started", "in-progress", "completed", "blocked"
        assigned_agent: Filter by assigned agent
        priority: Filter by priority - "low", "medium", "high", "urgent"
        show_completed: Whether to include completed tasks (default: False)
    
    Returns:
        Command that returns formatted list of TODO items
    """
    todos = [Todo.from_dict(todo_dict) for todo_dict in state.get("todos", [])]
    
    # Apply filters
    if status:
        try:
            status_enum = TodoStatus(status.lower().replace(' ', '-'))
            todos = [todo for todo in todos if todo.status == status_enum]
        except ValueError:
            pass
    
    if assigned_agent:
        todos = [todo for todo in todos if todo.assigned_agent == assigned_agent]
    
    if priority:
        try:
            priority_enum = TodoPriority(priority.lower())
            todos = [todo for todo in todos if todo.priority == priority_enum]
        except ValueError:
            pass
    
    if not show_completed:
        todos = [todo for todo in todos if todo.status != TodoStatus.COMPLETED]
    
    if not todos:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="📋 No TODOs found matching the specified criteria.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Sort by priority and creation date
    priority_order = {TodoPriority.URGENT: 0, TodoPriority.HIGH: 1, 
                     TodoPriority.MEDIUM: 2, TodoPriority.LOW: 3}
    todos.sort(key=lambda t: (priority_order.get(t.priority, 4), t.created_at))
    
    # Format TODO list
    todo_list = []
    for todo in todos:
        status_emoji = {
            TodoStatus.NOT_STARTED: "⏳",
            TodoStatus.IN_PROGRESS: "🔄", 
            TodoStatus.COMPLETED: "✅",
            TodoStatus.BLOCKED: "🚫",
            TodoStatus.CANCELLED: "❌"
        }
        
        priority_emoji = {
            TodoPriority.LOW: "🔵",
            TodoPriority.MEDIUM: "🟡", 
            TodoPriority.HIGH: "🟠",
            TodoPriority.URGENT: "🔴"
        }
        
        status_icon = status_emoji.get(todo.status, "❓")
        priority_icon = priority_emoji.get(todo.priority, "⚪")
        
        due_text = ""
        if todo.due_date:
            days_until_due = (todo.due_date - datetime.now()).days
            if days_until_due < 0:
                due_text = f" (⚠️ Overdue by {abs(days_until_due)} days)"
            elif days_until_due == 0:
                due_text = " (📅 Due today)"
            elif days_until_due <= 3:
                due_text = f" (⏰ Due in {days_until_due} days)"
        
        todo_list.append(
            f"{status_icon} {priority_icon} **{todo.title}**{due_text}\n"
            f"   ID: `{todo.id[:8]}...` | Agent: {todo.assigned_agent or 'Unassigned'}\n"
            f"   {todo.description[:100]}" + ("..." if len(todo.description) > 100 else "")
        )
    
    formatted_list = "\n\n".join(todo_list)
    
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"📋 **TODO List ({len(todos)} items):**\n\n{formatted_list}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def update_todo_status(
    todo_id: str,
    status: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    notes: Optional[str] = None
) -> Command:
    """Update the status of a TODO item.
    
    Use this tool to track progress on tasks. Mark items as in-progress when
    starting work, completed when finished, or blocked when encountering issues.
    
    Args:
        todo_id: ID of the TODO to update (can be partial ID)
        status: New status - "not-started", "in-progress", "completed", "blocked", "cancelled"
        notes: Optional notes about the status change
    
    Returns:
        Command that updates the TODO status and returns confirmation
    """
    # Find TODO by full or partial ID
    todos = state.get("todos", [])
    matching_todo = None
    
    for todo_dict in todos:
        if todo_dict["id"] == todo_id or todo_dict["id"].startswith(todo_id):
            matching_todo = todo_dict
            break
    
    if not matching_todo:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"❌ TODO with ID '{todo_id}' not found.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Validate status
    try:
        status_enum = TodoStatus(status.lower().replace(' ', '-'))
    except ValueError:
        valid_statuses = [s.value for s in TodoStatus]
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"❌ Invalid status '{status}'. Valid options: {valid_statuses}",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Update TODO
    old_status = matching_todo["status"]
    matching_todo["status"] = status_enum.value
    matching_todo["updated_at"] = datetime.now().isoformat()
    
    # Add notes if provided
    if notes:
        if "notes" not in matching_todo:
            matching_todo["notes"] = []
        matching_todo["notes"].append(f"{datetime.now().isoformat()}: {notes}")
    
    # Track completion time if completed
    if status_enum == TodoStatus.COMPLETED and "completed_at" not in matching_todo:
        matching_todo["completed_at"] = datetime.now().isoformat()
    
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(
                    content=f"✅ TODO status updated!\n\n"
                           f"**Title:** {matching_todo['title']}\n"
                           f"**Status:** {old_status} → {status_enum.value}\n" +
                           (f"**Notes:** {notes}\n" if notes else ""),
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def breakdown_task(
    main_task: str,
    subtasks: List[str],
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    priority: str = "medium",
    assigned_agent: Optional[str] = None
) -> Command:
    """Break down a complex task into smaller, manageable subtasks.
    
    Use this tool for planning complex workflows by creating a main task
    and multiple subtasks. Creates dependencies automatically.
    
    Args:
        main_task: Description of the main task to break down
        subtasks: List of subtask descriptions
        priority: Priority for all tasks - "low", "medium", "high", "urgent"
        assigned_agent: Agent to assign all tasks to (optional)
    
    Returns:
        Command that creates main task and subtasks with proper dependencies
    """
    # Create main task
    try:
        priority_enum = TodoPriority(priority.lower())
    except ValueError:
        priority_enum = TodoPriority.MEDIUM
    
    main_todo = Todo(
        title=main_task,
        description=f"Main task broken down into {len(subtasks)} subtasks",
        priority=priority_enum,
        assigned_agent=assigned_agent or state.get("current_agent"),
        tags=["main-task", "breakdown"]
    )
    
    # Create subtasks
    subtask_todos = []
    for i, subtask_desc in enumerate(subtasks, 1):
        subtask = Todo(
            title=f"Subtask {i}: {subtask_desc}",
            description=subtask_desc,
            priority=priority_enum,
            assigned_agent=assigned_agent or state.get("current_agent"),
            tags=["subtask", "breakdown"],
            dependencies=[]  # Subtasks can be independent unless specified
        )
        subtask_todos.append(subtask)
    
    # Set main task dependencies to all subtasks
    main_todo.dependencies = [todo.id for todo in subtask_todos]
    
    # Add all TODOs to state
    todos = state.get("todos", [])
    all_new_todos = [main_todo] + subtask_todos
    
    for todo in all_new_todos:
        todos.append(todo.to_dict())
    
    # Update agent context
    agent_contexts = state.get("agent_contexts", {})
    current_agent = assigned_agent or state.get("current_agent")
    if current_agent and current_agent in agent_contexts:
        agent_contexts[current_agent]["todos_assigned"].extend([todo.id for todo in all_new_todos])
    
    # Format response
    subtask_list = "\n".join([f"  {i}. {todo.title} (`{todo.id[:8]}...`)" 
                             for i, todo in enumerate(subtask_todos, 1)])
    
    return Command(
        update={
            "todos": todos,
            "agent_contexts": agent_contexts,
            "messages": [
                ToolMessage(
                    content=f"✅ Task breakdown completed!\n\n"
                           f"**Main Task:** {main_todo.title} (`{main_todo.id[:8]}...`)\n\n"
                           f"**Subtasks ({len(subtask_todos)}):**\n{subtask_list}\n\n"
                           f"💡 Complete all subtasks to automatically complete the main task.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def get_todo_details(
    todo_id: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Get detailed information about a specific TODO item.
    
    Use this tool to review the full details of a task including notes,
    dependencies, progress history, and metadata.
    
    Args:
        todo_id: ID of the TODO to get details for (can be partial ID)
    
    Returns:
        Command that returns detailed TODO information
    """
    # Find TODO by full or partial ID
    todos = state.get("todos", [])
    matching_todo = None
    
    for todo_dict in todos:
        if todo_dict["id"] == todo_id or todo_dict["id"].startswith(todo_id):
            matching_todo = Todo.from_dict(todo_dict)
            break
    
    if not matching_todo:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"❌ TODO with ID '{todo_id}' not found.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Format detailed information
    details = f"""📋 **TODO Details**

**ID:** {matching_todo.id}
**Title:** {matching_todo.title}
**Description:** {matching_todo.description}
**Status:** {matching_todo.status.value}
**Priority:** {matching_todo.priority.value}
**Assigned Agent:** {matching_todo.assigned_agent or 'Unassigned'}

**Timing:**
- Created: {matching_todo.created_at.strftime('%Y-%m-%d %H:%M')}
- Updated: {matching_todo.updated_at.strftime('%Y-%m-%d %H:%M')}
- Due Date: {matching_todo.due_date.strftime('%Y-%m-%d') if matching_todo.due_date else 'Not set'}

**Progress:**
- Estimated Duration: {matching_todo.estimated_duration or 'Not estimated'} minutes
- Actual Duration: {matching_todo.actual_duration or 'Not tracked'} minutes

**Organization:**
- Tags: {', '.join(matching_todo.tags) if matching_todo.tags else 'None'}
- Dependencies: {len(matching_todo.dependencies)} TODO(s)
"""
    
    if matching_todo.notes:
        notes_text = "\n".join([f"  • {note}" for note in matching_todo.notes])
        details += f"\n**Notes:**\n{notes_text}"
    
    if matching_todo.dependencies:
        # Show dependency details
        dep_details = []
        for dep_id in matching_todo.dependencies:
            for todo_dict in todos:
                if todo_dict["id"] == dep_id:
                    dep_todo = Todo.from_dict(todo_dict)
                    status_emoji = "✅" if dep_todo.status == TodoStatus.COMPLETED else "⏳"
                    dep_details.append(f"  {status_emoji} {dep_todo.title} (`{dep_todo.id[:8]}...`)")
                    break
        
        if dep_details:
            details += f"\n\n**Dependencies:**\n" + "\n".join(dep_details)
    
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=details,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


def create_todo_tools():
    """Create a list of TODO management tools for agents."""
    return [create_todo, list_todos, update_todo_status, breakdown_task, get_todo_details]


def get_todo_usage_instructions() -> str:
    """Get instructions for using TODO tools effectively."""
    return """
# TODO MANAGEMENT INSTRUCTIONS

Use these tools to plan, track, and manage complex workflows:

## Planning Strategy:
1. **Start with breakdown_task** for complex workflows
2. **Create specific TODOs** for individual actions
3. **Set priorities** based on urgency and importance
4. **Assign agents** to distribute work appropriately

## Progress Tracking:
1. **Mark in-progress** when starting work on a TODO
2. **Add notes** when updating status to track decisions
3. **Mark completed** when work is finished
4. **Use blocked** when encountering dependencies or issues

## Best Practices:
- Use descriptive titles that clearly indicate the action
- Include detailed descriptions with acceptance criteria
- Set realistic due dates and time estimates
- Review TODO list regularly to maintain priorities
- Break down large tasks into smaller, actionable items

## Status Workflow:
not-started → in-progress → completed
           ↘ blocked → in-progress
           ↘ cancelled

## Priority Guidelines:
- **urgent**: Must be done immediately (same day)
- **high**: Important and time-sensitive (within 2-3 days)
- **medium**: Standard priority (within a week)
- **low**: Nice to have, no specific deadline
"""