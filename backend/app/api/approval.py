"""
API endpoints for interaction approval and feedback collection.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from app.services.interaction_recorder import get_interaction_recorder

logger = logging.getLogger(__name__)

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
        logger.info("Getting pending interactions endpoint called")
        recorder = get_interaction_recorder()
        if not recorder:
            logger.error("Recorder is None!")
            raise HTTPException(status_code=500, detail="Interaction recorder not initialized")
        
        logger.info("Recorder instance ID: %s", id(recorder))
        pending = recorder.get_pending_interactions()
        logger.info("Got %d pending interactions from recorder", len(pending))
        return [InteractionResponse(**interaction) for interaction in pending]
        
    except Exception as e:
        logger.error("Error getting pending interactions: %s", str(e))
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
        
        logger.info("User %s interaction %s", action, request.interaction_id)
        
        return {
            "success": True, 
            "action": action,
            "interaction_id": request.interaction_id,
            "feedback": request.feedback
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing approval: %s", str(e))
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
        logger.error("Error getting approval stats: %s", str(e))
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
        logger.error("Error in bulk approval: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to process bulk approval")