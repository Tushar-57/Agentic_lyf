"""
Human-in-the-Loop (HITL) System for Deep Agents
==============================================

Implements human approval workflows, guidance injection, and intervention
capabilities for high-stakes decisions and complex workflows.
"""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio
from dataclasses import dataclass, field

from .deep_state import DeepAgentState


class ApprovalStatus(Enum):
    """Status of human approval requests."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"
    MODIFIED = "modified"


class ApprovalPriority(Enum):
    """Priority levels for approval requests."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InterventionType(Enum):
    """Types of human interventions."""
    APPROVAL = "approval"
    GUIDANCE = "guidance"
    CORRECTION = "correction"
    ESCALATION = "escalation"
    FEEDBACK = "feedback"


@dataclass
class ApprovalRequest:
    """Represents a human approval request."""
    id: str
    agent_id: str
    action_type: str
    description: str
    context: Dict[str, Any]
    priority: ApprovalPriority
    timeout_minutes: int
    created_at: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING
    response: Optional[str] = None
    response_at: Optional[datetime] = None
    human_feedback: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if approval request has expired."""
        if self.timeout_minutes <= 0:
            return False
        return datetime.now() > self.created_at + timedelta(minutes=self.timeout_minutes)
    
    @property
    def time_remaining(self) -> int:
        """Get remaining time in minutes."""
        if self.timeout_minutes <= 0:
            return -1
        remaining = (self.created_at + timedelta(minutes=self.timeout_minutes)) - datetime.now()
        return max(0, int(remaining.total_seconds() / 60))


@dataclass
class HumanIntervention:
    """Represents a human intervention in agent workflow."""
    id: str
    agent_id: str
    workflow_id: str
    intervention_type: InterventionType
    description: str
    human_input: str
    context: Dict[str, Any]
    created_at: datetime
    applied_at: Optional[datetime] = None
    impact_assessment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HumanInTheLoopManager:
    """
    Manages human-in-the-loop interactions for the deep agent system.
    
    Provides:
    - Approval workflow management
    - Human intervention tracking
    - Decision escalation
    - Feedback collection and integration
    """
    
    def __init__(self):
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        self.approval_history: List[ApprovalRequest] = []
        self.interventions: List[HumanIntervention] = []
        self.approval_callbacks: Dict[str, Callable] = {}
        
        # Configuration
        self.default_timeout_minutes = 30
        self.auto_approve_threshold = ApprovalPriority.LOW
        self.escalation_threshold = ApprovalPriority.CRITICAL
        
        # Approval criteria for different action types
        self.approval_criteria = {
            "data_modification": {
                "required": True,
                "priority": ApprovalPriority.HIGH,
                "timeout": 15
            },
            "external_api_call": {
                "required": True,
                "priority": ApprovalPriority.MEDIUM,
                "timeout": 10
            },
            "file_deletion": {
                "required": True,
                "priority": ApprovalPriority.HIGH,
                "timeout": 20
            },
            "budget_change": {
                "required": True,
                "priority": ApprovalPriority.CRITICAL,
                "timeout": 60
            },
            "goal_modification": {
                "required": True,
                "priority": ApprovalPriority.MEDIUM,
                "timeout": 30
            },
            "habit_change": {
                "required": False,
                "priority": ApprovalPriority.LOW,
                "timeout": 5
            },
            "routine_adjustment": {
                "required": False,
                "priority": ApprovalPriority.LOW,
                "timeout": 5
            }
        }
    
    async def request_approval(
        self,
        agent_id: str,
        action_type: str,
        description: str,
        context: Dict[str, Any],
        priority: Optional[ApprovalPriority] = None,
        timeout_minutes: Optional[int] = None
    ) -> ApprovalRequest:
        """
        Request human approval for an agent action.
        
        Args:
            agent_id: ID of the requesting agent
            action_type: Type of action requiring approval
            description: Human-readable description of the action
            context: Relevant context for the approval decision
            priority: Priority level (auto-determined if None)
            timeout_minutes: Timeout in minutes (auto-determined if None)
        
        Returns:
            ApprovalRequest object that can be monitored for status
        """
        # Determine approval requirements
        criteria = self.approval_criteria.get(action_type, {})
        
        if priority is None:
            priority = criteria.get("priority", ApprovalPriority.MEDIUM)
        
        if timeout_minutes is None:
            timeout_minutes = criteria.get("timeout", self.default_timeout_minutes)
        
        # Create approval request
        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            action_type=action_type,
            description=description,
            context=context,
            priority=priority,
            timeout_minutes=timeout_minutes,
            created_at=datetime.now(),
            metadata={
                "requires_approval": criteria.get("required", True),
                "auto_approve_eligible": priority.value == ApprovalPriority.LOW.value
            }
        )
        
        # Store request
        self.pending_approvals[request.id] = request
        
        # Auto-approve low priority items if configured
        if (priority == ApprovalPriority.LOW and 
            not criteria.get("required", True)):
            await self._auto_approve(request)
        
        return request
    
    async def wait_for_approval(
        self,
        request_id: str,
        poll_interval: float = 2.0
    ) -> ApprovalRequest:
        """
        Wait for approval request to be resolved.
        
        Args:
            request_id: ID of the approval request
            poll_interval: How often to check status (seconds)
        
        Returns:
            Resolved ApprovalRequest
        """
        while request_id in self.pending_approvals:
            request = self.pending_approvals[request_id]
            
            # Check for timeout
            if request.is_expired and request.status == ApprovalStatus.PENDING:
                await self._handle_timeout(request)
                break
            
            # Check if resolved
            if request.status != ApprovalStatus.PENDING:
                break
            
            await asyncio.sleep(poll_interval)
        
        # Move to history and return
        if request_id in self.pending_approvals:
            request = self.pending_approvals.pop(request_id)
            self.approval_history.append(request)
            return request
        else:
            # Find in history
            for req in self.approval_history:
                if req.id == request_id:
                    return req
            
            raise ValueError(f"Approval request {request_id} not found")
    
    async def provide_approval(
        self,
        request_id: str,
        approved: bool,
        feedback: Optional[str] = None,
        modifications: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Provide human approval or denial for a request.
        
        Args:
            request_id: ID of the approval request
            approved: Whether the action is approved
            feedback: Optional human feedback
            modifications: Optional modifications to the proposed action
        
        Returns:
            True if approval was successfully recorded
        """
        if request_id not in self.pending_approvals:
            return False
        
        request = self.pending_approvals[request_id]
        
        # Update request
        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
        request.response = "approved" if approved else "denied"
        request.response_at = datetime.now()
        request.human_feedback = feedback
        request.modifications = modifications
        
        if modifications:
            request.status = ApprovalStatus.MODIFIED
        
        # Execute callback if registered
        if request_id in self.approval_callbacks:
            callback = self.approval_callbacks.pop(request_id)
            await callback(request)
        
        return True
    
    async def record_intervention(
        self,
        agent_id: str,
        workflow_id: str,
        intervention_type: InterventionType,
        description: str,
        human_input: str,
        context: Dict[str, Any]
    ) -> HumanIntervention:
        """
        Record a human intervention in an agent workflow.
        
        Args:
            agent_id: ID of the agent being intervened
            workflow_id: ID of the workflow being modified
            intervention_type: Type of intervention
            description: Description of the intervention
            human_input: The human's input or guidance
            context: Relevant context
        
        Returns:
            HumanIntervention record
        """
        intervention = HumanIntervention(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            workflow_id=workflow_id,
            intervention_type=intervention_type,
            description=description,
            human_input=human_input,
            context=context,
            created_at=datetime.now()
        )
        
        self.interventions.append(intervention)
        return intervention
    
    def get_pending_approvals(self, agent_id: Optional[str] = None) -> List[ApprovalRequest]:
        """Get list of pending approval requests."""
        requests = list(self.pending_approvals.values())
        
        if agent_id:
            requests = [r for r in requests if r.agent_id == agent_id]
        
        # Sort by priority and creation time
        priority_order = {
            ApprovalPriority.CRITICAL: 0,
            ApprovalPriority.HIGH: 1,
            ApprovalPriority.MEDIUM: 2,
            ApprovalPriority.LOW: 3
        }
        
        return sorted(requests, key=lambda r: (priority_order[r.priority], r.created_at))
    
    def get_approval_history(
        self,
        agent_id: Optional[str] = None,
        days: int = 7
    ) -> List[ApprovalRequest]:
        """Get approval history for analysis."""
        cutoff = datetime.now() - timedelta(days=days)
        history = [r for r in self.approval_history if r.created_at >= cutoff]
        
        if agent_id:
            history = [r for r in history if r.agent_id == agent_id]
        
        return sorted(history, key=lambda r: r.created_at, reverse=True)
    
    def get_intervention_history(
        self,
        agent_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        days: int = 7
    ) -> List[HumanIntervention]:
        """Get intervention history for analysis."""
        cutoff = datetime.now() - timedelta(days=days)
        interventions = [i for i in self.interventions if i.created_at >= cutoff]
        
        if agent_id:
            interventions = [i for i in interventions if i.agent_id == agent_id]
        
        if workflow_id:
            interventions = [i for i in interventions if i.workflow_id == workflow_id]
        
        return sorted(interventions, key=lambda i: i.created_at, reverse=True)
    
    async def _auto_approve(self, request: ApprovalRequest) -> None:
        """Automatically approve low-risk requests."""
        request.status = ApprovalStatus.APPROVED
        request.response = "auto_approved"
        request.response_at = datetime.now()
        request.metadata["auto_approved"] = True
    
    async def _handle_timeout(self, request: ApprovalRequest) -> None:
        """Handle timeout for approval requests."""
        request.status = ApprovalStatus.TIMEOUT
        request.response = "timeout"
        request.response_at = datetime.now()
        
        # Apply default action based on priority
        if request.priority in [ApprovalPriority.LOW, ApprovalPriority.MEDIUM]:
            # Default to approval for low/medium priority
            request.status = ApprovalStatus.APPROVED
            request.metadata["timeout_action"] = "auto_approved"
        else:
            # Default to denial for high/critical priority
            request.status = ApprovalStatus.DENIED
            request.metadata["timeout_action"] = "auto_denied"
    
    def register_approval_callback(
        self,
        request_id: str,
        callback: Callable[[ApprovalRequest], None]
    ) -> None:
        """Register callback to be executed when approval is received."""
        self.approval_callbacks[request_id] = callback
    
    def get_approval_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get approval statistics for monitoring."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_requests = [r for r in self.approval_history if r.created_at >= cutoff]
        
        if not recent_requests:
            return {"total": 0, "period_days": days}
        
        total = len(recent_requests)
        approved = len([r for r in recent_requests if r.status == ApprovalStatus.APPROVED])
        denied = len([r for r in recent_requests if r.status == ApprovalStatus.DENIED])
        timeout = len([r for r in recent_requests if r.status == ApprovalStatus.TIMEOUT])
        modified = len([r for r in recent_requests if r.status == ApprovalStatus.MODIFIED])
        
        # Average response time (excluding timeouts)
        responded = [r for r in recent_requests if r.response_at and r.status != ApprovalStatus.TIMEOUT]
        avg_response_time = 0
        if responded:
            total_time = sum((r.response_at - r.created_at).total_seconds() for r in responded)
            avg_response_time = total_time / len(responded) / 60  # in minutes
        
        return {
            "total": total,
            "approved": approved,
            "denied": denied,
            "timeout": timeout,
            "modified": modified,
            "approval_rate": (approved / total * 100) if total > 0 else 0,
            "avg_response_time_minutes": round(avg_response_time, 2),
            "period_days": days
        }


# Global HITL manager instance
_hitl_manager = None

def get_hitl_manager() -> HumanInTheLoopManager:
    """Get the global human-in-the-loop manager instance."""
    global _hitl_manager
    if _hitl_manager is None:
        _hitl_manager = HumanInTheLoopManager()
    return _hitl_manager


# Convenience functions for agents
async def request_human_approval(
    agent_id: str,
    action_type: str,
    description: str,
    context: Dict[str, Any],
    priority: Optional[ApprovalPriority] = None,
    timeout_minutes: Optional[int] = None,
    wait_for_response: bool = True
) -> ApprovalRequest:
    """Convenience function for agents to request approval."""
    manager = get_hitl_manager()
    
    request = await manager.request_approval(
        agent_id=agent_id,
        action_type=action_type,
        description=description,
        context=context,
        priority=priority,
        timeout_minutes=timeout_minutes
    )
    
    if wait_for_response:
        return await manager.wait_for_approval(request.id)
    
    return request


async def record_human_intervention(
    agent_id: str,
    workflow_id: str,
    intervention_type: InterventionType,
    description: str,
    human_input: str,
    context: Dict[str, Any]
) -> HumanIntervention:
    """Convenience function for recording interventions."""
    manager = get_hitl_manager()
    
    return await manager.record_intervention(
        agent_id=agent_id,
        workflow_id=workflow_id,
        intervention_type=intervention_type,
        description=description,
        human_input=human_input,
        context=context
    )