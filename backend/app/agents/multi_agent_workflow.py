"""
Multi-Agent Workflow Coordinator

Orchestrates complex workflows across multiple specialized agents,
enabling cross-domain productivity improvements like:
- Productivity analysis → Scheduling optimization
- Goal review → Calendar blocking
- Pattern detection → Reminder creation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Awaitable
import asyncio
import uuid

from ..utils.structured_logging import get_logger, LogComponent

logger = get_logger(__name__, LogComponent.AGENT)


class WorkflowStepType(str, Enum):
    """Types of workflow steps."""
    ANALYSIS = "analysis"  # Data analysis and insight generation
    DECISION = "decision"  # Branching based on conditions
    ACTION = "action"  # Execute an action (schedule, notify, etc.)
    NOTIFICATION = "notification"  # Send notification/reminder
    HANDOFF = "handoff"  # Hand off to another agent
    APPROVAL = "approval"  # Wait for user approval


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_APPROVAL = "waiting_approval"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """A single step in a multi-agent workflow."""
    step_id: str
    step_type: WorkflowStepType
    agent_type: str  # Which agent executes this step
    description: str
    input_mapping: Dict[str, str]  # Map workflow context to agent input
    output_mapping: Dict[str, str]  # Map agent output to workflow context
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None  # Conditional execution
    requires_approval: bool = False
    timeout_seconds: int = 60


@dataclass
class WorkflowDefinition:
    """Definition of a multi-agent workflow."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    on_complete: Optional[str] = None  # Summary message template


@dataclass
class WorkflowExecution:
    """State of a running workflow execution."""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    context: Dict[str, Any] = field(default_factory=dict)
    current_step_index: int = 0
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    user_approvals: Dict[str, bool] = field(default_factory=dict)


class MultiAgentWorkflowCoordinator:
    """
    Coordinates complex multi-agent workflows.
    
    Enables scenarios like:
    1. Productivity agent analyzes time patterns
    2. Detects skill development deficit
    3. Hands off to scheduling agent
    4. Scheduling agent proposes calendar blocks
    5. Creates actionable schedule with user approval
    """
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.agent_registry: Dict[str, Any] = {}  # Agent instances
        self.step_handlers: Dict[WorkflowStepType, Callable] = {
            WorkflowStepType.ANALYSIS: self._handle_analysis_step,
            WorkflowStepType.ACTION: self._handle_action_step,
            WorkflowStepType.HANDOFF: self._handle_handoff_step,
            WorkflowStepType.NOTIFICATION: self._handle_notification_step,
            WorkflowStepType.DECISION: self._handle_decision_step,
            WorkflowStepType.APPROVAL: self._handle_approval_step,
        }
    
    def register_agent(self, agent_type: str, agent_instance: Any):
        """Register an agent instance for workflow execution."""
        self.agent_registry[agent_type] = agent_instance
        logger.info("agent_registered", f"Registered {agent_type} agent for workflows")
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a workflow definition."""
        self.workflows[workflow.workflow_id] = workflow
        logger.info("workflow_registered", f"Registered workflow: {workflow.name}")
    
    async def start_workflow(
        self,
        workflow_id: str,
        initial_context: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """
        Start a new workflow execution.
        
        Returns:
            execution_id for tracking
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            context={
                **initial_context,
                "user_id": user_id,
                "execution_id": execution_id,
                "workflow_name": workflow.name,
            },
            started_at=datetime.now()
        )
        
        self.executions[execution_id] = execution
        
        # Start execution asynchronously
        asyncio.create_task(self._run_workflow(execution_id))
        
        logger.info(
            "workflow_started",
            f"Started workflow {workflow.name} ({execution_id})",
            {"workflow_id": workflow_id, "execution_id": execution_id}
        )
        
        return execution_id
    
    async def _run_workflow(self, execution_id: str):
        """Execute workflow steps sequentially."""
        execution = self.executions.get(execution_id)
        if not execution:
            return
        
        workflow = self.workflows.get(execution.workflow_id)
        if not workflow:
            execution.status = WorkflowStatus.FAILED
            return
        
        execution.status = WorkflowStatus.RUNNING
        
        try:
            while execution.current_step_index < len(workflow.steps):
                step = workflow.steps[execution.current_step_index]
                
                # Check condition
                if step.condition and not step.condition(execution.context):
                    execution.current_step_index += 1
                    continue
                
                # Execute step
                logger.info(
                    "workflow_step_start",
                    f"Executing step {step.step_id} ({step.step_type})",
                    {"execution_id": execution_id, "step_id": step.step_id}
                )
                
                handler = self.step_handlers.get(step.step_type)
                if not handler:
                    raise ValueError(f"No handler for step type: {step.step_type}")
                
                result = await handler(step, execution)
                execution.step_results.append({
                    "step_id": step.step_id,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update context with outputs
                for key, value in result.items():
                    execution.context[step.output_mapping.get(key, key)] = value
                
                # Check for approval requirement
                if step.requires_approval and step.step_id not in execution.user_approvals:
                    execution.status = WorkflowStatus.WAITING_APPROVAL
                    logger.info(
                        "workflow_waiting_approval",
                        f"Workflow {execution_id} waiting for approval on step {step.step_id}"
                    )
                    return  # Pause execution
                
                execution.current_step_index += 1
            
            # Complete workflow
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            
            logger.info(
                "workflow_completed",
                f"Workflow {execution_id} completed successfully",
                {"execution_id": execution_id, "steps_completed": len(execution.step_results)}
            )
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.context["error"] = str(e)
            logger.error(
                "workflow_failed",
                f"Workflow {execution_id} failed: {e}",
                {"execution_id": execution_id, "error": str(e)}
            )
    
    async def provide_approval(self, execution_id: str, step_id: str, approved: bool):
        """Provide user approval for a workflow step."""
        execution = self.executions.get(execution_id)
        if not execution:
            return False
        
        if execution.status != WorkflowStatus.WAITING_APPROVAL:
            return False
        
        execution.user_approvals[step_id] = approved
        
        if approved:
            execution.status = WorkflowStatus.RUNNING
            execution.current_step_index += 1
            # Resume workflow
            asyncio.create_task(self._run_workflow(execution_id))
        else:
            execution.status = WorkflowStatus.CANCELLED
        
        return True
    
    async def _handle_analysis_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Handle analysis step by calling the appropriate agent."""
        agent = self.agent_registry.get(step.agent_type)
        if not agent:
            raise ValueError(f"Agent not found: {step.agent_type}")
        
        # Prepare input from context
        agent_input = {}
        for key, context_key in step.input_mapping.items():
            agent_input[key] = execution.context.get(context_key)
        
        # Execute agent
        result = await agent.execute({
            "user_input": step.description,
            "context": agent_input
        })
        
        return result
    
    async def _handle_handoff_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Handle handoff to another agent."""
        target_agent_type = step.input_mapping.get("target_agent", step.agent_type)
        agent = self.agent_registry.get(target_agent_type)
        
        if not agent:
            return {"handoff_status": "failed", "error": f"Agent {target_agent_type} not found"}
        
        # Prepare handoff context
        handoff_context = {}
        for key, context_key in step.input_mapping.items():
            if key != "target_agent":
                handoff_context[key] = execution.context.get(context_key)
        
        # Execute target agent
        result = await agent.execute({
            "user_input": step.description,
            "context": handoff_context
        })
        
        return {
            "handoff_status": "completed",
            "from_agent": step.agent_type,
            "to_agent": target_agent_type,
            "result": result
        }
    
    async def _handle_action_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Handle action step (schedule creation, reminder, etc.)."""
        action_type = step.input_mapping.get("action_type", "unknown")
        
        # Extract action parameters
        params = {}
        for key, context_key in step.input_mapping.items():
            if key != "action_type":
                params[key] = execution.context.get(context_key)
        
        # Execute action (placeholder - integrate with actual services)
        logger.info(
            "workflow_action",
            f"Executing {action_type} action",
            {"execution_id": execution.execution_id, "action_type": action_type}
        )
        
        return {
            "action_status": "ready_for_execution",
            "action_type": action_type,
            "params": params,
            "requires_confirmation": True
        }
    
    async def _handle_notification_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Handle notification step."""
        notification_text = step.input_mapping.get("message_template", step.description)
        
        # Format with context
        try:
            formatted_message = notification_text.format(**execution.context)
        except KeyError:
            formatted_message = notification_text
        
        return {
            "notification_sent": True,
            "message": formatted_message,
            "channel": step.input_mapping.get("channel", "in_app")
        }
    
    async def _handle_decision_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Handle decision/branching step."""
        condition_result = True
        if step.condition:
            condition_result = step.condition(execution.context)
        
        return {
            "decision_made": True,
            "condition_result": condition_result,
            "next_branch": "true_branch" if condition_result else "false_branch"
        }
    
    async def _handle_approval_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Handle user approval step."""
        # This step just marks that approval is needed
        # The actual approval is handled via provide_approval()
        return {
            "approval_required": True,
            "step_id": step.step_id,
            "description": step.description
        }
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow execution."""
        execution = self.executions.get(execution_id)
        if not execution:
            return None
        
        return {
            "execution_id": execution.execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "current_step": execution.current_step_index,
            "total_steps": len(self.workflows[execution.workflow_id].steps) if execution.workflow_id in self.workflows else 0,
            "context_summary": {k: v for k, v in execution.context.items() if k not in ["user_input", "detailed_analysis"]},
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "requires_approval": execution.status == WorkflowStatus.WAITING_APPROVAL,
        }


# Pre-defined workflows

def create_productivity_optimization_workflow() -> WorkflowDefinition:
    """
    Create workflow: Productivity analysis → Scheduling optimization.
    
    Triggered when user asks about productivity or time optimization.
    """
    return WorkflowDefinition(
        workflow_id="productivity_optimization",
        name="Productivity Optimization",
        description="Analyze productivity patterns and create optimized schedule",
        steps=[
            WorkflowStep(
                step_id="analyze_patterns",
                step_type=WorkflowStepType.ANALYSIS,
                agent_type="productivity",
                description="Analyze time patterns and identify optimization opportunities",
                input_mapping={
                    "time_entries": "time_entries",
                    "user_priorities": "priorities",
                    "profile_snapshot": "profile_snapshot"
                },
                output_mapping={
                    "analysis_result": "pattern_analysis",
                    "optimization_opportunities": "opportunities",
                    "recommended_blocks": "recommended_blocks"
                }
            ),
            WorkflowStep(
                step_id="decision_need_scheduling",
                step_type=WorkflowStepType.DECISION,
                agent_type="orchestrator",
                description="Determine if scheduling optimization is needed",
                input_mapping={},
                output_mapping={"proceed": "should_schedule"},
                condition=lambda ctx: len(ctx.get("opportunities", [])) > 0
            ),
            WorkflowStep(
                step_id="handoff_to_scheduling",
                step_type=WorkflowStepType.HANDOFF,
                agent_type="orchestrator",
                description="Hand off to scheduling agent to create optimized schedule",
                input_mapping={
                    "target_agent": "scheduling",
                    "recommended_blocks": "recommended_blocks",
                    "optimization_opportunities": "opportunities",
                    "user_preferences": "profile_snapshot"
                },
                output_mapping={
                    "schedule_proposal": "proposed_schedule",
                    "calendar_events": "calendar_events"
                }
            ),
            WorkflowStep(
                step_id="approval_for_schedule",
                step_type=WorkflowStepType.APPROVAL,
                agent_type="orchestrator",
                description="Request user approval for proposed schedule changes",
                input_mapping={"schedule_summary": "proposed_schedule"},
                output_mapping={"approved": "schedule_approved"},
                requires_approval=True
            ),
            WorkflowStep(
                step_id="create_calendar_events",
                step_type=WorkflowStepType.ACTION,
                agent_type="scheduling",
                description="Create approved calendar events",
                input_mapping={
                    "action_type": "create_calendar_events",
                    "events": "calendar_events"
                },
                output_mapping={"created_events": "created_events"},
                condition=lambda ctx: ctx.get("schedule_approved", False)
            ),
            WorkflowStep(
                step_id="set_reminders",
                step_type=WorkflowStepType.NOTIFICATION,
                agent_type="orchestrator",
                description="Set up reminder notifications",
                input_mapping={
                    "message_template": "Set reminder for {recommended_blocks}",
                    "channel": "in_app"
                },
                output_mapping={"reminders_set": "reminders_status"}
            )
        ],
        on_complete="Productivity optimization workflow completed. Created {created_events} calendar events."
    )


def create_goal_alignment_workflow() -> WorkflowDefinition:
    """
    Create workflow: Goal review → Time reallocation → Schedule adjustment.
    
    Triggered when user asks about goals or missing priorities.
    """
    return WorkflowDefinition(
        workflow_id="goal_alignment",
        name="Goal Alignment & Time Reallocation",
        description="Review goals and reallocate time to align with priorities",
        steps=[
            WorkflowStep(
                step_id="analyze_goal_coverage",
                step_type=WorkflowStepType.ANALYSIS,
                agent_type="productivity",
                description="Analyze goal coverage and identify gaps",
                input_mapping={
                    "goals": "active_goals",
                    "time_entries": "time_entries",
                    "priorities": "priorities"
                },
                output_mapping={
                    "goal_coverage": "goal_coverage",
                    "gaps": "goal_gaps",
                    "recommendations": "goal_recommendations"
                }
            ),
            WorkflowStep(
                step_id="identify_available_slots",
                step_type=WorkflowStepType.ANALYSIS,
                agent_type="scheduling",
                description="Find available time slots for goal activities",
                input_mapping={
                    "calendar": "current_calendar",
                    "idle_gaps": "detected_gaps",
                    "target_duration": "recommended_duration"
                },
                output_mapping={
                    "available_slots": "available_slots",
                    "optimal_slots": "optimal_slots"
                }
            ),
            WorkflowStep(
                step_id="propose_schedule_changes",
                step_type=WorkflowStepType.HANDOFF,
                agent_type="orchestrator",
                description="Generate schedule proposal with goal-aligned blocks",
                input_mapping={
                    "target_agent": "scheduling",
                    "goal_gaps": "goal_gaps",
                    "optimal_slots": "optimal_slots",
                    "recommendations": "goal_recommendations"
                },
                output_mapping={
                    "proposed_changes": "schedule_changes",
                    "impact_assessment": "impact"
                },
                requires_approval=True
            ),
            WorkflowStep(
                step_id="notify_user",
                step_type=WorkflowStepType.NOTIFICATION,
                agent_type="orchestrator",
                description="Notify user of schedule changes",
                input_mapping={
                    "message_template": "Goal alignment complete. {impact}",
                    "channel": "in_app"
                },
                output_mapping={}
            )
        ],
        on_complete="Goal alignment workflow completed. Reallocated time to match priorities."
    )


# Singleton instance
_workflow_coordinator: Optional[MultiAgentWorkflowCoordinator] = None


def get_workflow_coordinator() -> MultiAgentWorkflowCoordinator:
    """Get or create the workflow coordinator singleton."""
    global _workflow_coordinator
    if _workflow_coordinator is None:
        _workflow_coordinator = MultiAgentWorkflowCoordinator()
        # Register default workflows
        _workflow_coordinator.register_workflow(create_productivity_optimization_workflow())
        _workflow_coordinator.register_workflow(create_goal_alignment_workflow())
    return _workflow_coordinator
