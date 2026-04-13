"""
API endpoints for actionable productivity features.

Exposes:
- Time analysis with categorization
- Actionable suggestion generation
- Workflow execution (scheduling, reminders)
- Multi-agent coordination
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends

from app.auth.user_context import get_current_user
from app.utils.structured_logging import get_logger, LogComponent
from app.services.time_context_analyzer import get_time_analyzer, TimeWindowAnalysis, WorkType
from app.services.knowledge_base import get_knowledge_base_service
from app.agents.multi_agent_workflow import (
    get_workflow_coordinator,
    WorkflowStatus,
    create_productivity_optimization_workflow,
    create_goal_alignment_workflow
)
from app.agents.specialized_agents.productivity_enhanced import (
    get_enhanced_productivity_agent,
    ActionableSuggestion
)

router = APIRouter(prefix="/api/productivity", tags=["productivity"])
logger = get_logger(__name__, LogComponent.API)


# Pydantic models

class TimeAnalysisRequest(BaseModel):
    """Request time analysis for a specific window."""
    window_label: str = "today"
    days_back: int = 7


class TimeAnalysisResponse(BaseModel):
    """Response with categorized time analysis."""
    window_label: str
    total_minutes: float
    breakdown: Dict[str, float]  # work_type -> minutes
    percentages: Dict[str, float]  # work_type -> percentage
    focus_score_avg: float
    productivity_score_avg: float
    optimization_opportunities: List[Dict[str, Any]]
    pattern_insights: List[str]


class ActionableSuggestionResponse(BaseModel):
    """Actionable suggestion for user to execute."""
    id: str
    title: str
    description: str
    action_type: str  # "schedule", "reminder", "workflow", "manual"
    parameters: Dict[str, Any]
    estimated_impact: str
    time_required: Optional[int] = None
    workflow_id: Optional[str] = None


class SuggestionsRequest(BaseModel):
    """Request actionable suggestions."""
    user_input: str
    context: Optional[Dict[str, Any]] = None


class SuggestionsResponse(BaseModel):
    """Response with actionable suggestions."""
    summary: str
    insights: List[str]
    suggestions: List[ActionableSuggestionResponse]
    workflow_triggered: bool


class WorkflowExecuteRequest(BaseModel):
    """Request to execute a workflow."""
    workflow_id: str
    context: Dict[str, Any]


class WorkflowExecuteResponse(BaseModel):
    """Response from workflow execution start."""
    execution_id: str
    workflow_id: str
    status: str
    message: str


class WorkflowStatusResponse(BaseModel):
    """Response with workflow execution status."""
    execution_id: str
    workflow_id: str
    status: str
    current_step: int
    total_steps: int
    requires_approval: bool
    context_summary: Dict[str, Any]


class WorkflowApprovalRequest(BaseModel):
    """Request to approve/reject a workflow step."""
    step_id: str
    approved: bool


# API Endpoints

@router.post("/analyze-time", response_model=TimeAnalysisResponse)
async def analyze_time(
    request: TimeAnalysisRequest,
    user = Depends(get_current_user)
):
    """
    Analyze time entries with smart categorization.
    
    Categorizes 6000 minutes into deep work, meetings, admin, learning, etc.
    """
    try:
        kb = get_knowledge_base_service(user.storage_key)
        time_analyzer = get_time_analyzer()
        
        # Get time entries from knowledge base
        all_entries = await kb.get_all_entries()
        
        # Filter to time entries within window
        time_entries = [
            entry for entry in all_entries
            if entry.category == "time_entry" or 
               (entry.metadata.get("context", {}).get("source_action", "").startswith("time_entry"))
        ]
        
        # Convert to analysis format
        entry_dicts = []
        for entry in time_entries[:100]:  # Limit for performance
            meta = entry.metadata.get("context", {}) if entry.metadata else {}
            entry_dicts.append({
                "entry_id": entry.entry_id,
                "description": meta.get("description", entry.title),
                "project_name": meta.get("project_name", entry.category),
                "duration_minutes": meta.get("duration_minutes", 30),
                "focus_score": meta.get("focus_score", 5),
                "energy_score": meta.get("energy_score", 5),
                "start_time": meta.get("start_time", entry.created_at.isoformat() if entry.created_at else None),
                "end_time": meta.get("end_time"),
            })
        
        # Get user priorities
        prefs = await kb.get_user_preferences()
        priorities = []
        if prefs and hasattr(prefs, 'general') and isinstance(prefs.general, dict):
            priorities = prefs.general.get("priorities", [])
        
        # Perform analysis
        analysis = time_analyzer.analyze_time_window(
            entries=entry_dicts,
            window_label=request.window_label,
            user_priorities=priorities
        )
        
        # Format response
        breakdown = {}
        percentages = {}
        total = analysis.total_minutes
        
        for work_type, minutes in analysis.categorized_breakdown.items():
            breakdown[work_type.value] = minutes
            percentages[work_type.value] = round(minutes / total * 100, 1) if total > 0 else 0
        
        return TimeAnalysisResponse(
            window_label=analysis.window_label,
            total_minutes=analysis.total_minutes,
            breakdown=breakdown,
            percentages=percentages,
            focus_score_avg=analysis.focus_score_avg,
            productivity_score_avg=analysis.productivity_score_avg,
            optimization_opportunities=analysis.optimization_opportunities,
            pattern_insights=analysis.pattern_insights
        )
        
    except Exception as e:
        logger.error("analyze_time_error", f"Failed to analyze time: {e}", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Time analysis failed: {str(e)}")


@router.post("/suggestions", response_model=SuggestionsResponse)
async def get_suggestions(
    request: SuggestionsRequest,
    user = Depends(get_current_user)
):
    """
    Get actionable productivity suggestions based on user query.
    
    Returns specific, executable recommendations with time slots and impact estimates.
    """
    try:
        agent = get_enhanced_productivity_agent()
        
        # Prepare state
        state = {
            "user_input": request.user_input,
            "context": request.context or {}
        }
        
        # Execute agent
        result = await agent.execute(state)
        
        # Extract actionable data
        actionable_data = result.get("actionable_data", {})
        suggestions = actionable_data.get("suggestions", [])
        
        # Format suggestions
        formatted_suggestions = []
        for i, sugg in enumerate(suggestions):
            if isinstance(sugg, ActionableSuggestion):
                formatted_suggestions.append(ActionableSuggestionResponse(
                    id=f"sugg_{i}",
                    title=sugg.title,
                    description=sugg.description,
                    action_type=sugg.action_type,
                    parameters=sugg.parameters,
                    estimated_impact=sugg.estimated_impact,
                    time_required=sugg.time_required,
                    workflow_id=sugg.workflow_id
                ))
        
        return SuggestionsResponse(
            summary=actionable_data.get("summary", "Analysis complete"),
            insights=actionable_data.get("insights", []),
            suggestions=formatted_suggestions,
            workflow_triggered=actionable_data.get("workflow_triggered", False)
        )
        
    except Exception as e:
        logger.error("suggestions_error", f"Failed to get suggestions: {e}", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Suggestions generation failed: {str(e)}")


@router.post("/execute-workflow", response_model=WorkflowExecuteResponse)
async def execute_workflow(
    request: WorkflowExecuteRequest,
    user = Depends(get_current_user)
):
    """
    Start a multi-agent workflow.
    
    Workflows:
    - productivity_optimization: Analyze patterns → Schedule optimization
    - goal_alignment: Review goals → Reallocate time → Adjust schedule
    """
    try:
        coordinator = get_workflow_coordinator()
        
        # Register workflows if not already done
        coordinator.register_workflow(create_productivity_optimization_workflow())
        coordinator.register_workflow(create_goal_alignment_workflow())
        
        # Start workflow
        execution_id = await coordinator.start_workflow(
            workflow_id=request.workflow_id,
            initial_context=request.context,
            user_id=user.storage_key
        )
        
        return WorkflowExecuteResponse(
            execution_id=execution_id,
            workflow_id=request.workflow_id,
            status="started",
            message=f"Workflow {request.workflow_id} started successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("workflow_execute_error", f"Failed to execute workflow: {e}", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


@router.get("/workflow-status/{execution_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    execution_id: str,
    user = Depends(get_current_user)
):
    """Get status of a running workflow execution."""
    try:
        coordinator = get_workflow_coordinator()
        status = coordinator.get_execution_status(execution_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Workflow execution not found")
        
        return WorkflowStatusResponse(
            execution_id=status["execution_id"],
            workflow_id=status["workflow_id"],
            status=status["status"],
            current_step=status["current_step"],
            total_steps=status["total_steps"],
            requires_approval=status["requires_approval"],
            context_summary=status.get("context_summary", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_status_error", f"Failed to get workflow status: {e}", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/workflow-approve/{execution_id}", response_model=Dict[str, Any])
async def approve_workflow_step(
    execution_id: str,
    request: WorkflowApprovalRequest,
    user = Depends(get_current_user)
):
    """Approve or reject a workflow step requiring user confirmation."""
    try:
        coordinator = get_workflow_coordinator()
        success = await coordinator.provide_approval(
            execution_id=execution_id,
            step_id=request.step_id,
            approved=request.approved
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to process approval")
        
        return {
            "success": True,
            "execution_id": execution_id,
            "step_id": request.step_id,
            "approved": request.approved,
            "message": "Step approved, workflow resuming" if request.approved else "Step rejected, workflow cancelled"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_approval_error", f"Failed to process approval: {e}", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Approval processing failed: {str(e)}")


@router.post("/quick-schedule", response_model=Dict[str, Any])
async def quick_schedule(
    request: Dict[str, Any],
    user = Depends(get_current_user)
):
    """
    Quick schedule a time block without full workflow.
    
    Request body:
    {
        "title": "LeetCode Practice",
        "duration_minutes": 30,
        "preferred_time": "18:00",
        "recurrence": "daily"
    }
    """
    try:
        # This would integrate with calendar/scheduling service
        # For now, return success with what would be created
        
        logger.info(
            "quick_schedule",
            f"Scheduling {request.get('title')} for user {user.storage_key}",
            request
        )
        
        return {
            "success": True,
            "scheduled_event": {
                "title": request.get("title"),
                "duration_minutes": request.get("duration_minutes"),
                "scheduled_time": request.get("preferred_time"),
                "recurrence": request.get("recurrence", "none"),
                "user_id": user.storage_key
            },
            "message": f"Scheduled: {request.get('title')} at {request.get('preferred_time')}"
        }
        
    except Exception as e:
        logger.error("quick_schedule_error", f"Failed to schedule: {e}", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Scheduling failed: {str(e)}")
