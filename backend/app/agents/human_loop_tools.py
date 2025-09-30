"""
Human-in-the-Loop Tools for Deep Agents
======================================

Tools that enable agents to request human approval, guidance, and intervention
for complex decisions and high-stakes actions.
"""

from typing import Optional, List
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId
from typing_extensions import Annotated

from .deep_state import DeepAgentState
from .human_loop import ApprovalPriority


@tool(parse_docstring=True)
def request_approval(
    action_description: str,
    action_type: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    priority: str = "medium",
    timeout_minutes: int = 30,
    context_details: Optional[str] = None,
    risk_assessment: Optional[str] = None
) -> Command:
    """Request human approval for a significant action.
    
    Use this tool when you need human approval before proceeding with:
    - Data modifications or deletions
    - Financial transactions or budget changes
    - External API calls or integrations
    - Goal or habit modifications
    - High-impact decisions
    
    Args:
        action_description: Clear description of what you want to do
        action_type: Type of action (e.g., "data_modification", "budget_change", "goal_modification")
        priority: Priority level - "low", "medium", "high", "critical"
        timeout_minutes: How long to wait for approval (default 30 minutes)
        context_details: Additional context to help human make decision
        risk_assessment: Your assessment of risks and alternatives
    
    Returns:
        Command that initiates approval request and waits for human response
    """
    current_agent = state.get("current_agent", "unknown")
    
    # Map priority string to enum
    priority_map = {
        "low": ApprovalPriority.LOW,
        "medium": ApprovalPriority.MEDIUM,
        "high": ApprovalPriority.HIGH,
        "critical": ApprovalPriority.CRITICAL
    }
    # Store priority for future use in approval workflow
    _ = priority_map.get(priority.lower(), ApprovalPriority.MEDIUM)
    
    # Build context for approval
    approval_context = {
        "current_agent": current_agent,
        "conversation_context": state.get("messages", [])[-3:],  # Last 3 messages
        "available_files": list(state.get("files", {}).keys()),
        "current_todos": [todo for todo in state.get("todos", []) if todo.status != "completed"],
        "timestamp": datetime.now().isoformat()
    }
    
    if context_details:
        approval_context["additional_context"] = context_details
    
    if risk_assessment:
        approval_context["risk_assessment"] = risk_assessment
    
    # Store approval request details in state
    approval_requests = state.get("approval_requests", [])
    request_id = f"approval_{len(approval_requests) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    approval_request = {
        "id": request_id,
        "agent_id": current_agent,
        "action_type": action_type,
        "description": action_description,
        "priority": priority,
        "timeout_minutes": timeout_minutes,
        "context": approval_context,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    
    approval_requests.append(approval_request)
    
    # Create detailed approval prompt for human
    additional_context_section = f"## Additional Context\n{context_details}" if context_details else ""
    risk_assessment_section = f"## Risk Assessment\n{risk_assessment}" if risk_assessment else ""
    
    approval_prompt = f"""# Human Approval Required

**Agent:** {current_agent}
**Action Type:** {action_type}
**Priority:** {priority.upper()}
**Timeout:** {timeout_minutes} minutes

## Proposed Action
{action_description}

## Context
- **Current Conversation:** Recent discussion about user's request
- **Available Data:** {len(approval_context.get('available_files', []))} files available
- **Active TODOs:** {len(approval_context.get('current_todos', []))} items

{additional_context_section}

{risk_assessment_section}

## Decision Options
✅ **APPROVE** - Proceed with the action as described
❌ **DENY** - Block this action and use alternative approach
🔄 **MODIFY** - Approve with modifications (please specify changes)

**Please respond with your decision and any feedback or modifications.**
"""
    
    # Store the approval prompt in files for reference
    files = state.get("files", {})
    approval_filename = f"approval_request_{request_id}.md"
    files[approval_filename] = approval_prompt
    
    return Command(
        update={
            "approval_requests": approval_requests,
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"🔒 **Human Approval Requested**\n\n"
                           f"**Action:** {action_description}\n"
                           f"**Type:** {action_type}\n"
                           f"**Priority:** {priority.title()}\n"
                           f"**Timeout:** {timeout_minutes} minutes\n\n"
                           f"📋 Approval request details saved to `{approval_filename}`\n\n"
                           f"⏳ Waiting for human approval before proceeding...\n\n"
                           f"**Request ID:** {request_id}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def request_guidance(
    question: str,
    context: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    urgency: str = "normal",
    options: Optional[List[str]] = None
) -> Command:
    """Request human guidance for complex decisions or unclear situations.
    
    Use this tool when you need human input on:
    - Ambiguous user requests that need clarification
    - Complex decisions with multiple valid approaches
    - Situations where domain expertise is needed
    - Ethical or sensitive considerations
    
    Args:
        question: Clear question you need guidance on
        context: Relevant context and background information
        urgency: Urgency level - "low", "normal", "high"
        options: Optional list of options you're considering
    
    Returns:
        Command that requests guidance and records the intervention
    """
    current_agent = state.get("current_agent", "unknown")
    
    # Create guidance request
    guidance_requests = state.get("guidance_requests", [])
    request_id = f"guidance_{len(guidance_requests) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    guidance_request = {
        "id": request_id,
        "agent_id": current_agent,
        "question": question,
        "context": context,
        "urgency": urgency,
        "options": options or [],
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    
    guidance_requests.append(guidance_request)
    
    # Create guidance prompt
    options_section = ""
    if options:
        options_list = "\n".join(f"{i+1}. {option}" for i, option in enumerate(options))
        options_section = f"## Options Being Considered\n{options_list}"
    
    guidance_prompt = f"""# Human Guidance Requested

**Agent:** {current_agent}
**Urgency:** {urgency.upper()}
**Request ID:** {request_id}

## Question
{question}

## Context
{context}

{options_section}

## Current State
- **Available Files:** {len(state.get('files', {}))} files
- **Active TODOs:** {len([t for t in state.get('todos', []) if t.status != 'completed'])} items
- **Recent Messages:** {len(state.get('messages', []))} in conversation

## Guidance Needed
Please provide your guidance, suggestions, or direction on how to proceed. Your input will help me make the best decision for the user.
"""
    
    # Store guidance request in files
    files = state.get("files", {})
    guidance_filename = f"guidance_request_{request_id}.md"
    files[guidance_filename] = guidance_prompt
    
    return Command(
        update={
            "guidance_requests": guidance_requests,
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"🤔 **Human Guidance Requested**\n\n"
                           f"**Question:** {question}\n"
                           f"**Urgency:** {urgency.title()}\n"
                           f"**Context:** {context[:100]}{'...' if len(context) > 100 else ''}\n\n"
                           f"📋 Guidance request saved to `{guidance_filename}`\n\n"
                           f"💭 Awaiting human input to proceed with best approach...\n\n"
                           f"**Request ID:** {request_id}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def escalate_to_human(
    issue_description: str,
    escalation_reason: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    severity: str = "medium",
    attempted_solutions: Optional[List[str]] = None
) -> Command:
    """Escalate complex issues or errors to human intervention.
    
    Use this tool when:
    - You encounter errors you cannot resolve
    - The user's request is beyond your capabilities
    - There are safety or ethical concerns
    - Multiple approaches have failed
    
    Args:
        issue_description: Clear description of the issue or problem
        escalation_reason: Why human intervention is needed
        severity: Severity level - "low", "medium", "high", "critical"
        attempted_solutions: List of solutions you've already tried
    
    Returns:
        Command that escalates the issue and requests intervention
    """
    current_agent = state.get("current_agent", "unknown")
    
    # Create escalation record
    escalations = state.get("escalations", [])
    escalation_id = f"escalation_{len(escalations) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    escalation = {
        "id": escalation_id,
        "agent_id": current_agent,
        "issue": issue_description,
        "reason": escalation_reason,
        "severity": severity,
        "attempted_solutions": attempted_solutions or [],
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    
    escalations.append(escalation)
    
    # Create escalation report
    attempted_solutions_section = "## Attempted Solutions\nNo solutions attempted yet."
    if attempted_solutions:
        solutions_list = "\n".join(f"- {solution}" for solution in attempted_solutions)
        attempted_solutions_section = f"## Attempted Solutions\n{solutions_list}"
    
    escalation_report = f"""# Issue Escalation Report

**Agent:** {current_agent}
**Severity:** {severity.upper()}
**Escalation ID:** {escalation_id}
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Issue Description
{issue_description}

## Escalation Reason
{escalation_reason}

{attempted_solutions_section}

## Current Context
- **User Request:** {state.get('messages', [])[-1].get('content', 'N/A') if state.get('messages') else 'N/A'}
- **Agent State:** {len(state.get('files', {}))} files, {len([t for t in state.get('todos', []) if t.status != 'completed'])} active TODOs
- **Conversation Length:** {len(state.get('messages', []))} messages

## Impact Assessment
- **User Experience:** Issue is preventing completion of user's request
- **System State:** Agent workflow has been interrupted
- **Urgency:** {severity} priority intervention needed

## Recommended Actions
1. Review the issue and attempted solutions
2. Provide guidance on alternative approaches
3. Consider if user expectations need to be managed
4. Determine if the request requires different resources or capabilities

## Next Steps
Please review this escalation and provide guidance on how to proceed. The agent is currently blocked and awaiting direction.
"""
    
    # Store escalation in files
    files = state.get("files", {})
    escalation_filename = f"escalation_{escalation_id}.md"
    files[escalation_filename] = escalation_report
    
    return Command(
        update={
            "escalations": escalations,
            "files": files,
            "messages": [
                ToolMessage(
                    content=f"🚨 **Issue Escalated to Human**\n\n"
                           f"**Issue:** {issue_description}\n"
                           f"**Severity:** {severity.title()}\n"
                           f"**Reason:** {escalation_reason}\n\n"
                           f"📋 Escalation report saved to `{escalation_filename}`\n\n"
                           f"🔧 **Attempted Solutions:** {len(attempted_solutions or [])} approaches tried\n\n"
                           f"⚠️ Agent workflow is currently blocked pending human intervention.\n\n"
                           f"**Escalation ID:** {escalation_id}",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(parse_docstring=True)
def check_approval_status(
    request_id: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Check the status of a pending approval request.
    
    Use this tool to check if a human has responded to your approval request.
    
    Args:
        request_id: ID of the approval request to check
    
    Returns:
        Command with current status of the approval request
    """
    approval_requests = state.get("approval_requests", [])
    
    # Find the request
    request = None
    for req in approval_requests:
        if req["id"] == request_id:
            request = req
            break
    
    if not request:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"❌ Approval request `{request_id}` not found.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    status = request.get("status", "pending")
    created_at = request.get("created_at", "unknown")
    
    if status == "pending":
        # Check if timeout has passed
        timeout_minutes = request.get("timeout_minutes", 30)
        created_time = datetime.fromisoformat(created_at) if created_at != "unknown" else datetime.now()
        elapsed_minutes = (datetime.now() - created_time).total_seconds() / 60
        
        if elapsed_minutes > timeout_minutes:
            # Update status to timeout
            request["status"] = "timeout"
            request["response_at"] = datetime.now().isoformat()
            status = "timeout"
        
        remaining_minutes = max(0, timeout_minutes - elapsed_minutes)
        
        return Command(
            update={
                "approval_requests": approval_requests,
                "messages": [
                    ToolMessage(
                        content=f"⏳ **Approval Status: PENDING**\n\n"
                               f"**Request ID:** {request_id}\n"
                               f"**Action:** {request.get('description', 'Unknown')}\n"
                               f"**Created:** {created_at}\n"
                               f"**Time Remaining:** {remaining_minutes:.1f} minutes\n\n"
                               f"📝 Still waiting for human response...",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    elif status == "approved":
        feedback = request.get("human_feedback", "")
        modifications = request.get("modifications", {})
        
        feedback_section = f"**Human Feedback:** {feedback}\n\n" if feedback else ""
        modifications_section = f"**Modifications:** {modifications}\n\n" if modifications else ""
        
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"✅ **Approval Status: APPROVED**\n\n"
                               f"**Request ID:** {request_id}\n"
                               f"**Action:** {request.get('description', 'Unknown')}\n"
                               f"**Approved At:** {request.get('response_at', 'Unknown')}\n\n"
                               f"{feedback_section}"
                               f"{modifications_section}"
                               f"🚀 You can now proceed with the approved action.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    elif status == "denied":
        feedback = request.get("human_feedback", "")
        feedback_section = f"**Human Feedback:** {feedback}\n\n" if feedback else ""
        
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"❌ **Approval Status: DENIED**\n\n"
                               f"**Request ID:** {request_id}\n"
                               f"**Action:** {request.get('description', 'Unknown')}\n"
                               f"**Denied At:** {request.get('response_at', 'Unknown')}\n\n"
                               f"{feedback_section}"
                               f"🔄 Please consider alternative approaches or request guidance.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    elif status == "timeout":
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"⏰ **Approval Status: TIMEOUT**\n\n"
                               f"**Request ID:** {request_id}\n"
                               f"**Action:** {request.get('description', 'Unknown')}\n"
                               f"**Timeout At:** {request.get('response_at', 'Unknown')}\n\n"
                               f"🔄 Human did not respond within timeout period. Consider alternative approach or escalate if critical.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )
    
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"❓ **Approval Status: {status.upper()}**\n\n"
                               f"**Request ID:** {request_id}\n"
                               f"Status is unexpected. Please check the approval request details.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )


@tool(parse_docstring=True)
def get_human_feedback(
    context: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    feedback_type: str = "general",
    specific_question: Optional[str] = None
) -> Command:
    """Request general feedback from human about current approach or results.
    
    Use this tool to get human feedback on:
    - Quality of your work or approach
    - User satisfaction with results
    - Suggestions for improvement
    - Validation of your understanding
    
    Args:
        context: Context for the feedback request
        feedback_type: Type of feedback - "general", "quality", "approach", "satisfaction"
        specific_question: Specific question you want feedback on
    
    Returns:
        Command that requests feedback from human
    """
    current_agent = state.get("current_agent", "unknown")
    
    # Create feedback request
    feedback_requests = state.get("feedback_requests", [])
    request_id = f"feedback_{len(feedback_requests) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    feedback_request = {
        "id": request_id,
        "agent_id": current_agent,
        "context": context,
        "feedback_type": feedback_type,
        "specific_question": specific_question,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    
    feedback_requests.append(feedback_request)
    
    # Create feedback prompt
    specific_question_section = f"## Specific Question\n{specific_question}" if specific_question else ""
    
    feedback_prompt = f"""# Feedback Request

**Agent:** {current_agent}
**Feedback Type:** {feedback_type.title()}
**Request ID:** {request_id}

## Context
{context}

{specific_question_section}

## Feedback Areas
Please provide feedback on:
- **Effectiveness:** How well did the agent address your needs?
- **Approach:** Was the method/strategy appropriate?
- **Quality:** Are you satisfied with the results?
- **Improvements:** Any suggestions for better outcomes?

## Current Results
- **Files Created:** {len(state.get('files', {}))} files
- **TODOs Managed:** {len(state.get('todos', []))} items
- **Conversation Flow:** {len(state.get('messages', []))} messages

Your feedback will help improve future interactions and agent performance.
"""
    
    # Store feedback request
    files = state.get("files", {})
    feedback_filename = f"feedback_request_{request_id}.md"
    files[feedback_filename] = feedback_prompt
    
    return Command(
        update={
            "feedback_requests": feedback_requests,
            "files": files,
            "messages": [
                ToolMessage(
                    content=(f"💬 **Feedback Requested**\n\n"
                           f"**Type:** {feedback_type.title()}\n"
                           f"**Context:** {context[:100]}{'...' if len(context) > 100 else ''}\n\n" +
                           (f"**Specific Question:** {specific_question}\n\n" if specific_question else "") +
                           f"📋 Feedback request saved to `{feedback_filename}`\n\n"
                           f"💭 Your feedback will help improve future performance.\n\n"
                           f"**Request ID:** {request_id}"),
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


def create_human_loop_tools():
    """Create a list of human-in-the-loop tools."""
    return [
        request_approval,
        request_guidance,
        escalate_to_human,
        check_approval_status,
        get_human_feedback
    ]