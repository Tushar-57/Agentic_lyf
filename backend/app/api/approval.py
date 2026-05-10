"""
API endpoints for interaction approval and feedback collection.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.interaction_recorder import get_interaction_recorder
from app.utils.structured_logging import get_logger, LogComponent

logger = get_logger(__name__, LogComponent.API)

router = APIRouter()

class ApprovalRequest(BaseModel):
    interaction_id: str
    approved: bool
    feedback: Optional[str] = None

class InteractionResponse(BaseModel):
    id: str
    user_input: str
    agent_response: str
    agent_type: str
    timestamp: str
    status: str
    knowledge_sources: Optional[List[Dict[str, Any]]] = []

@router.get("/pending", response_model=List[InteractionResponse])
async def get_pending_interactions():
    """Get all pending interactions waiting for user approval."""
    try:
        logger.info("get_pending", "Getting pending interactions endpoint called")
        recorder = get_interaction_recorder()
        if not recorder:
            logger.error("recorder_none", "Recorder is None")
            raise HTTPException(status_code=500, detail="Interaction recorder not initialized")
        
        logger.info("recorder_instance", f"Recorder instance ID: {id(recorder)}")
        # Bug fix: get_pending_interactions is async — must await. Without await
        # it returns a coroutine; len() raises TypeError and the endpoint 500s,
        # which causes the UI to render "All Caught Up" while the stats endpoint
        # (which uses a sync count path) keeps showing the red dot.
        pending = await recorder.get_pending_interactions()
        logger.info("pending_count", f"Got {len(pending)} pending interactions", {"count": len(pending)})
        return [InteractionResponse(**interaction) for interaction in pending]
        
    except Exception as e:
        logger.error("get_pending_error", "Error getting pending interactions", error=e)
        raise HTTPException(status_code=500, detail="Failed to get pending interactions")

@router.post("/approve")
async def approve_interaction(request: ApprovalRequest):
    """Approve or reject a pending interaction."""
    try:
        recorder = get_interaction_recorder()
        if not recorder:
            raise HTTPException(status_code=500, detail="Interaction recorder not initialized")
        
        if request.approved:
            success = await recorder.approve_interaction(request.interaction_id)
            action = "approved"
        else:
            success = await recorder.reject_interaction(request.interaction_id)
            action = "rejected"
        
        if not success:
            raise HTTPException(status_code=404, detail="Interaction not found")
        
        logger.info("interaction_action", f"User {action} interaction", {"action": action, "interaction_id": request.interaction_id})
        
        return {
            "success": True, 
            "action": action,
            "interaction_id": request.interaction_id,
            "feedback": request.feedback
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("approval_error", "Error processing approval", error=e)
        raise HTTPException(status_code=500, detail="Failed to process approval")

@router.get("/stats")
async def get_approval_stats():
    """Get statistics about pending approvals."""
    try:
        recorder = get_interaction_recorder()
        if not recorder:
            raise HTTPException(status_code=500, detail="Interaction recorder not initialized")
        
        stats = recorder.get_recording_stats()
        return stats
        
    except Exception as e:
        logger.error("stats_error", "Error getting approval stats", error=e)
        raise HTTPException(status_code=500, detail="Failed to get stats")

@router.post("/bulk-approve")
async def bulk_approve_interactions(interaction_ids: List[str], approved: bool = True):
    """Bulk approve or reject multiple interactions."""
    try:
        recorder = get_interaction_recorder()
        if not recorder:
            raise HTTPException(status_code=500, detail="Interaction recorder not initialized")
        
        results = []
        for interaction_id in interaction_ids:
            if approved:
                success = await recorder.approve_interaction(interaction_id)
            else:
                success = await recorder.reject_interaction(interaction_id)
            
            results.append({
                "interaction_id": interaction_id,
                "success": success,
                "action": "approved" if approved else "rejected"
            })
        
        return {"results": results}
        
    except Exception as e:
        logger.error("bulk_approval_error", "Error in bulk approval", error=e)
        raise HTTPException(status_code=500, detail="Failed to process bulk approval")