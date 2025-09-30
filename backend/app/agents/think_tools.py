"""
Think and Reflection Tools for Deep Agents
==========================================

Tools for strategic thinking, reflection, and decision-making that enable
agents to pause, analyze, and plan their next steps systematically.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId
from typing_extensions import Annotated

from .deep_state import DeepAgentState
from .deep_state import Todo, TodoStatus


@tool(parse_docstring=True)
def think_tool(
    reflection: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    category: str = "general",
    save_to_file: bool = True
) -> Command:
    """Tool for strategic reflection and decision-making.
    
    Use this tool to pause and think strategically about:
    - Current progress and findings
    - What information is still needed
    - Next steps and priorities
    - Quality of current approach
    - Alternative strategies
    
    This creates deliberate decision points in the workflow for better outcomes.
    
    Args:
        reflection: Your detailed reflection on current situation and next steps
        category: Type of reflection - "planning", "analysis", "decision", "review", "general"
        save_to_file: Whether to save reflection to a file for later reference
    
    Returns:
        Command that records reflection and provides acknowledgment
    """
    current_agent = state.get("current_agent", "unknown")
    timestamp = datetime.now()
    
    # Format reflection with metadata
    formatted_reflection = f"""# Strategic Reflection - {category.title()}

**Agent:** {current_agent}
**Time:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Category:** {category}

## Reflection

{reflection}

---
*This reflection was captured to support strategic decision-making.*
"""
    
    # Update agent context with thinking activity
    agent_contexts = state.get("agent_contexts", {})
    if current_agent in agent_contexts:
        agent_contexts[current_agent]["last_activity"] = timestamp.isoformat()
        if "execution_history" not in agent_contexts[current_agent]:
            agent_contexts[current_agent]["execution_history"] = []
        
        agent_contexts[current_agent]["execution_history"].append({
            "action": "strategic_thinking",
            "category": category,
            "timestamp": timestamp.isoformat(),
            "summary": reflection[:200] + "..." if len(reflection) > 200 else reflection
        })
    
    updates = {
        "agent_contexts": agent_contexts,
        "messages": [
            ToolMessage(
                content=f"🤔 **Strategic Reflection Recorded**\n\n"
                       f"Category: {category}\n"
                       f"Length: {len(reflection)} characters\n\n"
                       f"💡 Reflection has been processed and will guide next decisions.",
                tool_call_id=tool_call_id
            )
        ]
    }
    
    # Save to file if requested
    if save_to_file:
        files = state.get("files", {})
        filename = f"reflection_{category}_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        files[filename] = formatted_reflection
        updates["files"] = files
        
        # Update agent's files_created
        if current_agent in agent_contexts:
            agent_contexts[current_agent]["files_created"].append(filename)
    
    return Command(update=updates)


@tool(parse_docstring=True)
def analyze_progress(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    focus_area: Optional[str] = None
) -> Command:
    """Analyze current progress across all active work.
    
    Use this tool to get an overview of:
    - Active TODOs and their status
    - Recent agent activities
    - Available context files
    - Pending human approvals
    
    Helps maintain situational awareness and identify next priorities.
    
    Args:
        focus_area: Optional area to focus analysis on - "todos", "files", "agents", "approvals"
    
    Returns:
        Command that returns comprehensive progress analysis
    """
    
    
    analysis = ["# Progress Analysis\n"]
    
    # Analyze TODOs
    if not focus_area or focus_area == "todos":
        todos = [Todo.from_dict(todo_dict) for todo_dict in state.get("todos", [])]
        
        if todos:
            status_counts = {}
            for todo in todos:
                status_counts[todo.status] = status_counts.get(todo.status, 0) + 1
            
            analysis.append("## 📋 TODO Status")
            for status, count in status_counts.items():
                emoji = {"not-started": "⏳", "in-progress": "🔄", "completed": "✅", 
                        "blocked": "🚫", "cancelled": "❌"}.get(status.value, "❓")
                analysis.append(f"- {emoji} {status.value.replace('-', ' ').title()}: {count}")
            
            # Show urgent/overdue items
            urgent_todos = [t for t in todos if t.priority.value == "urgent" and t.status != TodoStatus.COMPLETED]
            if urgent_todos:
                analysis.append(f"\n⚠️ **{len(urgent_todos)} Urgent TODOs require attention**")
            
            overdue_todos = [t for t in todos if t.due_date and t.due_date < datetime.now() and t.status != TodoStatus.COMPLETED]
            if overdue_todos:
                analysis.append(f"\n🔥 **{len(overdue_todos)} Overdue TODOs**")
        else:
            analysis.append("## 📋 TODO Status\nNo TODOs currently active.")
    
    # Analyze files
    if not focus_area or focus_area == "files":
        files = state.get("files", {})
        analysis.append(f"\n## 📁 Context Files: {len(files)} stored")
        
        if files:
            # Categorize files by type
            file_types = {}
            for filename in files.keys():
                if filename.startswith("reflection_"):
                    file_types["Reflections"] = file_types.get("Reflections", 0) + 1
                elif filename.startswith("analysis_"):
                    file_types["Analysis"] = file_types.get("Analysis", 0) + 1
                elif filename.startswith("research_"):
                    file_types["Research"] = file_types.get("Research", 0) + 1
                else:
                    file_types["Other"] = file_types.get("Other", 0) + 1
            
            for file_type, count in file_types.items():
                analysis.append(f"- {file_type}: {count}")
    
    # Analyze agent activity
    if not focus_area or focus_area == "agents":
        agent_contexts = state.get("agent_contexts", {})
        analysis.append(f"\n## 🤖 Agent Activity: {len(agent_contexts)} agents active")
        
        current_agent = state.get("current_agent")
        if current_agent:
            analysis.append(f"- Current agent: **{current_agent}**")
        
        for agent_id, context in agent_contexts.items():
            status = context.get("status", "unknown")
            last_activity = context.get("last_activity", "unknown")
            tools_used = len(context.get("tools_used", []))
            analysis.append(f"- {agent_id}: {status}, {tools_used} tools used")
    
    # Analyze approvals
    if not focus_area or focus_area == "approvals":
        approval_requests = state.get("approval_requests", [])
        pending_approvals = [req for req in approval_requests if req.get("status") == "pending"]
        
        analysis.append(f"\n## ✋ Human Approvals: {len(pending_approvals)} pending")
        
        if pending_approvals:
            for req in pending_approvals[:3]:  # Show first 3
                analysis.append(f"- {req.get('action_type', 'Unknown')}: {req.get('description', 'No description')[:50]}...")
    
    # Overall assessment
    analysis.append("\n## 🎯 Overall Assessment")
    
    total_todos = len(state.get("todos", []))
    completed_todos = len([t for t in state.get("todos", []) if Todo.from_dict(t).status == TodoStatus.COMPLETED])
    
    if total_todos > 0:
        completion_rate = (completed_todos / total_todos) * 100
        analysis.append(f"- Task completion rate: {completion_rate:.1f}% ({completed_todos}/{total_todos})")
    
    files_count = len(state.get("files", {}))
    if files_count > 0:
        analysis.append(f"- Knowledge base: {files_count} files available for context")
    
    pending_count = len(state.get("approval_requests", []))
    if pending_count > 0:
        analysis.append(f"- User interaction: {pending_count} items need attention")
    
    analysis_text = "\n".join(analysis)
    
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=analysis_text,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def plan_next_steps(
    current_situation: str,
    objectives: List[str],
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    constraints: Optional[List[str]] = None,
    time_horizon: str = "immediate"
) -> Command:
    """Plan next steps based on current situation and objectives.
    
    Use this tool to create structured plans for moving forward:
    - Assess current situation
    - Define clear objectives
    - Identify actionable next steps
    - Consider constraints and dependencies
    
    Args:
        current_situation: Description of where things stand now
        objectives: List of what needs to be achieved
        constraints: Optional list of limitations or requirements
        time_horizon: Planning horizon - "immediate", "short-term", "medium-term", "long-term"
    
    Returns:
        Command that creates structured plan and may generate TODOs
    """
    timestamp = datetime.now()
    
    plan = [
        f"# Strategic Plan - {time_horizon.title()}",
        f"**Created:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Agent:** {state.get('current_agent', 'unknown')}",
        "",
        "## Current Situation",
        current_situation,
        "",
        "## Objectives"
    ]
    
    for i, objective in enumerate(objectives, 1):
        plan.append(f"{i}. {objective}")
    
    if constraints:
        plan.append("\n## Constraints & Requirements")
        for constraint in constraints:
            plan.append(f"- {constraint}")
    
    # Analyze context for better planning
    todos = state.get("todos", [])
    files = state.get("files", {})
    
    plan.append("\n## Context Analysis")
    plan.append(f"- Active TODOs: {len(todos)}")
    plan.append(f"- Available context: {len(files)} files")
    
    # Generate recommendations based on time horizon
    plan.append("\n## Recommended Next Steps")
    
    if time_horizon == "immediate":
        plan.append("Focus on actions that can be completed within the next hour:")
        plan.append("1. Review any blocking issues or urgent TODOs")
        plan.append("2. Gather necessary information or context")
        plan.append("3. Take the first concrete action toward the primary objective")
    elif time_horizon == "short-term":
        plan.append("Focus on actions for the next few hours to 1 day:")
        plan.append("1. Break down complex objectives into specific tasks")
        plan.append("2. Identify and resolve dependencies")
        plan.append("3. Make significant progress on primary objectives")
    elif time_horizon == "medium-term":
        plan.append("Focus on actions for the next few days to 1 week:")
        plan.append("1. Establish sustainable workflows and processes")
        plan.append("2. Build comprehensive knowledge base and context")
        plan.append("3. Address multiple objectives systematically")
    else:  # long-term
        plan.append("Focus on strategic initiatives over weeks to months:")
        plan.append("1. Define comprehensive strategy and roadmap")
        plan.append("2. Establish measurement and review processes")
        plan.append("3. Build systems for continuous improvement")
    
    plan.append("\n## Success Metrics")
    plan.append("- Clear progress toward stated objectives")
    plan.append("- Reduced blocking issues and dependencies")
    plan.append("- Improved situational awareness and context")
    plan.append("- Effective use of available resources and time")
    
    plan_text = "\n".join(plan)
    
    # Save plan to file
    files = state.get("files", {})
    filename = f"plan_{time_horizon}_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
    files[filename] = plan_text
    
    # Update agent context
    agent_contexts = state.get("agent_contexts", {})
    current_agent = state.get("current_agent")
    if current_agent and current_agent in agent_contexts:
        agent_contexts[current_agent]["files_created"].append(filename)
        agent_contexts[current_agent]["execution_history"].append({
            "action": "strategic_planning",
            "time_horizon": time_horizon,
            "objectives_count": len(objectives),
            "timestamp": timestamp.isoformat()
        })
    
    return Command(
        update={
            "files": files,
            "agent_contexts": agent_contexts,
            "messages": [
                ToolMessage(
                    content=f"📝 **Strategic Plan Created**\n\n"
                           f"**Time Horizon:** {time_horizon}\n"
                           f"**Objectives:** {len(objectives)}\n"
                           f"**File:** {filename}\n\n"
                           f"💡 Plan provides structured approach to achieving your objectives.\n\n"
                           f"**Immediate Next Steps:**\n"
                           f"1. Review the detailed plan in {filename}\n"
                           f"2. Create specific TODOs for immediate actions\n"
                           f"3. Begin execution of the first priority item",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def decision_matrix(
    decision: str,
    options: List[str],
    criteria: List[str],
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    weights: Optional[List[float]] = None
) -> Command:
    """Create a decision matrix to evaluate options systematically.
    
    Use this tool for complex decisions where multiple options need to be
    evaluated against multiple criteria. Provides structured analysis.
    
    Args:
        decision: Description of the decision to be made
        options: List of possible options/alternatives
        criteria: List of criteria to evaluate options against
        weights: Optional weights for criteria (0.0-1.0, must sum to 1.0)
    
    Returns:
        Command that creates decision matrix analysis and saves to file
    """
    if weights and len(weights) != len(criteria):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Number of weights must match number of criteria.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    if weights and abs(sum(weights) - 1.0) > 0.01:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="❌ Weights must sum to 1.0.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    # Default equal weights if not provided
    if not weights:
        weights = [1.0 / len(criteria)] * len(criteria)
    
    timestamp = datetime.now()
    
    matrix = [
        f"# Decision Matrix Analysis",
        f"**Decision:** {decision}",
        f"**Created:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Agent:** {state.get('current_agent', 'unknown')}",
        "",
        "## Options to Evaluate"
    ]
    
    for i, option in enumerate(options, 1):
        matrix.append(f"{i}. {option}")
    
    matrix.append("\n## Evaluation Criteria")
    
    for i, (criterion, weight) in enumerate(zip(criteria, weights), 1):
        matrix.append(f"{i}. {criterion} (Weight: {weight:.2f})")
    
    matrix.append("\n## Evaluation Matrix")
    matrix.append("\n*Use this framework to score each option (1-5) against each criterion:*")
    matrix.append("\n| Option | " + " | ".join(criteria) + " | Weighted Score |")
    matrix.append("|--------|" + "|".join(["-" * (len(c) + 2) for c in criteria]) + "|----------------|")
    
    for option in options:
        row = f"| {option[:20]}" + "..." if len(option) > 20 else f"| {option}"
        row += " | " + " | ".join(["_/5_"] * len(criteria)) + " | _TBD_ |"
        matrix.append(row)
    
    matrix.extend([
        "",
        "## Scoring Guide",
        "- **5:** Excellent - Fully meets or exceeds criterion",
        "- **4:** Good - Meets criterion with minor gaps",
        "- **3:** Average - Partially meets criterion",
        "- **2:** Poor - Barely meets criterion",
        "- **1:** Inadequate - Does not meet criterion",
        "",
        "## Analysis Template",
        "",
        "### Strengths & Weaknesses",
        "*Fill in after scoring:*",
        ""
    ])
    
    for option in options:
        matrix.append(f"**{option}:**")
        matrix.append("- Strengths: _TBD_")
        matrix.append("- Weaknesses: _TBD_")
        matrix.append("")
    
    matrix.extend([
        "### Recommendation",
        "*Based on the analysis above:*",
        "",
        "**Recommended Option:** _TBD_",
        "",
        "**Rationale:** _TBD_",
        "",
        "**Implementation Considerations:** _TBD_",
        "",
        "**Risk Mitigation:** _TBD_"
    ])
    
    matrix_text = "\n".join(matrix)
    
    # Save to file
    files = state.get("files", {})
    filename = f"decision_matrix_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
    files[filename] = matrix_text
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"📊 **Decision Matrix Created**\n\n"
                           f"**Decision:** {decision}\n"
                           f"**Options:** {len(options)}\n"
                           f"**Criteria:** {len(criteria)}\n"
                           f"**File:** {filename}\n\n"
                           f"💡 Use the matrix to systematically evaluate each option.\n"
                           f"Complete the scoring and analysis sections for best results.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


def create_think_tools():
    """Create a list of thinking and reflection tools for agents."""
    return [think_tool, analyze_progress, plan_next_steps, decision_matrix]


def get_think_usage_instructions() -> str:
    """Get instructions for using thinking tools effectively."""
    return """
# STRATEGIC THINKING INSTRUCTIONS

Use these tools to enhance decision-making and planning:

## When to Use Think Tools:
- Before starting complex tasks (plan_next_steps)
- After gathering information (think_tool with analysis category)
- When facing decisions (decision_matrix)
- Periodically during long workflows (analyze_progress)
- When changing direction or strategy (think_tool with decision category)

## Thinking Categories:
- **planning**: Before starting new initiatives
- **analysis**: After research or data gathering
- **decision**: When choosing between options
- **review**: After completing tasks or milestones
- **general**: For open-ended reflection

## Best Practices:
1. **Regular reflection**: Use think_tool every few actions
2. **Progress reviews**: Check analyze_progress to maintain awareness
3. **Strategic planning**: Use plan_next_steps for complex workflows
4. **Structured decisions**: Use decision_matrix for important choices
5. **Document insights**: Save reflections to files for future reference

## Reflection Quality:
- Be specific about what you've learned
- Identify gaps in knowledge or approach
- Consider alternative strategies
- Plan concrete next steps
- Assess quality of current progress
"""