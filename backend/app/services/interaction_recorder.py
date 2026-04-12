"""
Enhanced Interaction Recording Service with User Approval

This service requires explicit user approval before recording interactions
to prevent knowledge base pollution with unvalidated assumptions.
"""
import asyncio
import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..utils.structured_logging import get_logger, LogComponent

logger = get_logger(__name__, LogComponent.SERVICE)

class PendingInteraction:
    """Represents an interaction awaiting user approval."""
    
    def __init__(self, user_input: str, agent_response: str, agent_type: str, 
                 context: Optional[Dict] = None, interaction_id: str = None, 
                 knowledge_sources: Optional[List[Dict]] = None):
        self.id = interaction_id or f"pending_{datetime.now().isoformat()}"
        self.user_input = user_input
        self.agent_response = agent_response
        self.agent_type = agent_type
        self.context = context or {}
        self.knowledge_sources = knowledge_sources or []  # Store KB sources used
        self.timestamp = datetime.now()
        self.status = "pending_approval"

class InteractionRecorder:
    """Enhanced service requiring user approval before recording interactions."""
    
    def __init__(self, knowledge_base_service, llm_service):
        self.knowledge_base = knowledge_base_service
        self.llm_service = llm_service
        self.logger = logger
        
        # Store pending interactions waiting for user approval
        self.pending_interactions: List[PendingInteraction] = []
        self._pending_lock = asyncio.Lock()  # Thread-safe access to pending_interactions
        
        # Common trivial patterns to filter out
        self.trivial_patterns = [
            r'^(hi|hello|hey|thanks?|thank you|ok|okay|yes|no|sure)\.?$',
            r'^(what time is it|what\'s the weather|how are you)\??$',
            r'^(test|testing|test message|hello world)\.?$',
            r'^[.]{1,3}$',  # Just dots
            r'^\s*$',       # Just whitespace
            r'^(lol|haha|😂|👍|👌|🙂|😊)$',  # Simple reactions
        ]
        
        # Patterns for valuable content
        self.valuable_patterns = [
            r'(goal|objective|plan|strategy|target)',
            r'(health|medical|symptom|doctor|medication)',
            r'(finance|money|budget|invest|expense|income)',
            r'(project|task|deadline|meeting|schedule)',
            r'(learn|study|research|course|skill)',
            r'(problem|issue|solution|fix|troubleshoot)',
            r'(data|analysis|report|metrics|statistics)',
            r'(preference|setting|configuration|customize)',
        ]

    def _truncate_text(self, value: Any, limit: int = 220) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return f"{text[:limit - 3]}..."

    def _extract_context_sources(self, context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Derive transparent knowledge sources from merged RAG context."""
        if not isinstance(context, dict):
            return []

        sources: List[Dict[str, Any]] = []

        for interaction in context.get("relevant_interactions", [])[:3]:
            if not isinstance(interaction, dict):
                continue
            sources.append({
                "type": "Previous Interaction",
                "content": self._truncate_text(interaction.get("content", "")),
                "similarity": interaction.get("similarity", 0),
                "created_at": interaction.get("created_at"),
                "category": interaction.get("category"),
                "metadata": interaction.get("metadata", {}),
            })

        for preference in context.get("user_preferences", [])[:3]:
            if not isinstance(preference, dict):
                continue
            sources.append({
                "type": "User Preference",
                "content": self._truncate_text(preference.get("content", "")),
                "similarity": preference.get("similarity", 0),
                "category": preference.get("category"),
                "metadata": preference.get("metadata", {}),
            })

        for entry in context.get("recent_time_entries", [])[:3]:
            if not isinstance(entry, dict):
                continue

            description = str(entry.get("description") or "work session").strip()
            project_name = str(entry.get("project_name") or "Unassigned").strip()
            duration_minutes = entry.get("duration_minutes")
            duration_suffix = f" ({duration_minutes}m)" if duration_minutes is not None else ""

            sources.append({
                "type": "Recent Time Entry",
                "content": self._truncate_text(f"{project_name}: {description}{duration_suffix}"),
                "similarity": entry.get("similarity", 0),
                "created_at": entry.get("created_at"),
                "category": "time_entry",
                "metadata": {
                    "project_name": entry.get("project_name"),
                    "description": entry.get("description"),
                    "duration_minutes": entry.get("duration_minutes"),
                    "billable": entry.get("billable"),
                },
            })

        for insight in context.get("patterns_and_insights", [])[:2]:
            if not isinstance(insight, dict):
                continue
            sources.append({
                "type": "Pattern/Insight",
                "content": self._truncate_text(insight.get("content", "")),
                "similarity": insight.get("similarity", 0),
                "metadata": insight.get("metadata", {}),
            })

        return sources

    def _looks_like_time_entry_context(self, context: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(context, dict):
            return False

        source = str(context.get("source", "")).strip().lower()
        source_action = str(context.get("source_action", "")).strip().lower()
        forced_category = str(context.get("category", "")).strip().lower()

        return (
            forced_category == "time_entry"
            or source == "alterego_timetracker"
            or "time_entry" in source_action
            or context.get("time_entry_id") is not None
        )

    def _should_promote_to_insight(self, pending: PendingInteraction) -> bool:
        """Promote high-value approved responses into insight category for discoverability."""
        if self._looks_like_time_entry_context(pending.context):
            return False

        combined_text = f"{pending.user_input} {pending.agent_response}".lower()
        response_text = pending.agent_response.strip()

        structured_signal = response_text.count("\n") >= 4 or any(
            token in combined_text
            for token in [
                "today's rundown",
                "what you completed",
                "what you missed",
                "next actions",
                "insight",
                "pattern",
                "analysis",
                "reflection",
                "summary",
            ]
        )
        depth_signal = len(response_text) >= 280

        return structured_signal and depth_signal

    def _build_approval_context(self, pending: PendingInteraction, approved_as_insight: bool) -> Dict[str, Any]:
        context_payload: Dict[str, Any] = dict(pending.context or {})

        context_payload["approval"] = {
            "approved": True,
            "approved_at": datetime.now().isoformat(),
            "source": "user_approval_queue",
            "approved_as_insight": approved_as_insight,
        }
        context_payload["approved_by_user"] = True
        context_payload["approved_at"] = datetime.now().isoformat()

        if pending.knowledge_sources:
            context_payload["knowledge_sources"] = pending.knowledge_sources

        if approved_as_insight and not self._looks_like_time_entry_context(context_payload):
            context_payload["category"] = "insight"
            context_payload["approved_as_insight"] = True

        return context_payload
    
    async def create_pending_interaction(self, user_input: str, agent_response: str, 
                                       agent_type: str = "general", context: Optional[Dict] = None,
                                       knowledge_sources: Optional[List[Dict]] = None) -> Optional[str]:
        """
        Create a pending interaction that requires user approval.
        
        Args:
            user_input: User's input
            agent_response: Agent's response
            agent_type: Type of agent
            context: Interaction context
            knowledge_sources: List of knowledge base sources used
        
        Returns:
            str: Interaction ID for approval tracking, None if filtered out
        """
        try:
            # Check if interaction should even be considered for recording
            should_consider = await self.should_record_interaction(
                user_input, agent_response, agent_type, context
            )
            
            if not should_consider:
                self.logger.debug("Interaction filtered out - not creating pending approval for %s", agent_type)
                return None
            
            resolved_sources = knowledge_sources or self._extract_context_sources(context)

            # Create pending interaction
            pending = PendingInteraction(
                user_input=user_input,
                agent_response=agent_response,
                agent_type=agent_type,
                context=context,
                knowledge_sources=resolved_sources
            )
            
            async with self._pending_lock:
                self.pending_interactions.append(pending)
            self.logger.info("Created pending interaction %s for user approval (%s)", pending.id, agent_type)
            return pending.id
            
        except Exception as e:
            self.logger.error("Error creating pending interaction: %s", str(e))
            return None
    
    async def approve_interaction(self, interaction_id: str) -> bool:
        """
        Approve a pending interaction and record it to knowledge base.
        
        Returns:
            bool: True if approved and recorded successfully
        """
        try:
            # Find the pending interaction
            pending = None
            for interaction in self.pending_interactions:
                if interaction.id == interaction_id:
                    pending = interaction
                    break
            
            if not pending:
                self.logger.warning("Pending interaction not found: %s", interaction_id)
                return False
            
            approved_as_insight = self._should_promote_to_insight(pending)
            approved_context = self._build_approval_context(pending, approved_as_insight)

            # Record the approved interaction
            saved_entry = await self.knowledge_base.add_interaction_history(
                agent_type=pending.agent_type,
                user_input=pending.user_input,
                agent_response=pending.agent_response,
                context=approved_context
            )
            
            # Remove from pending list
            async with self._pending_lock:
                self.pending_interactions.remove(pending)
            self.logger.info(
                "USER APPROVED and recorded interaction %s (%s) entry_id=%s category=%s insight=%s",
                interaction_id,
                pending.agent_type,
                getattr(saved_entry, "entry_id", None),
                getattr(saved_entry, "category", None),
                approved_as_insight,
            )
            return True
            
        except Exception as e:
            self.logger.error("Error approving interaction: %s", str(e))
            return False
    
    async def reject_interaction(self, interaction_id: str) -> bool:
        """
        Reject a pending interaction without recording.

        Returns:
            bool: True if rejected successfully
        """
        try:
            # Find and remove the pending interaction with lock protection
            async with self._pending_lock:
                for interaction in self.pending_interactions:
                    if interaction.id == interaction_id:
                        self.pending_interactions.remove(interaction)
                        self.logger.info("USER REJECTED interaction %s (%s)", interaction_id, interaction.agent_type)
                        return True

            self.logger.warning("Pending interaction not found for rejection: %s", interaction_id)
            return False

        except Exception as e:
            self.logger.error("Error rejecting interaction: %s", str(e))
            return False
    
    async def get_pending_interactions(self) -> List[Dict[str, Any]]:
        """Get all pending interactions waiting for approval."""
        async with self._pending_lock:
            pending_count = len(self.pending_interactions)
            pending_list = list(self.pending_interactions)  # Copy under lock
        self.logger.info("Getting pending interactions - count: %d, instance id: %s",
                        pending_count, id(self))
        return [
            {
                "id": pending.id,
                "user_input": pending.user_input,
                "agent_response": pending.agent_response,
                "agent_type": pending.agent_type,
                "timestamp": pending.timestamp.isoformat(),
                "status": pending.status,
                "knowledge_sources": pending.knowledge_sources or []
            }
            for pending in pending_list
        ]

    async def record_if_valuable(self, user_input: str, agent_response: str, 
                                agent_type: str = "general", context: Optional[Dict] = None, 
                                user_approved: bool = False) -> bool:
        """
        DEPRECATED: Use create_pending_interaction() instead.
        This method now only works with explicit user approval.
        """
        if not user_approved:
            self.logger.info("Use create_pending_interaction() for new approval workflow")
            return False
        
        return await self._record_approved_interaction(user_input, agent_response, agent_type, context)
    
    async def _record_approved_interaction(self, user_input: str, agent_response: str, 
                                         agent_type: str, context: Optional[Dict] = None) -> bool:
        """Internal method to record already-approved interactions."""
        try:
            should_record = await self.should_record_interaction(
                user_input, agent_response, agent_type, context
            )
            
            if should_record:
                await self.knowledge_base.add_interaction_history(
                    agent_type=agent_type,
                    user_input=user_input,
                    agent_response=agent_response,
                    context=context or {}
                )
                
                self.logger.info("Recorded USER-APPROVED interaction for %s", agent_type)
                return True
            else:
                self.logger.debug("Filtered out trivial interaction for %s", agent_type)
                return False
                
        except Exception as e:
            self.logger.error("Error recording approved interaction: %s", str(e))
            return False
    
    async def should_record_interaction(self, user_input: str, agent_response: str, 
                                      agent_type: str = "general", context: Optional[Dict] = None) -> bool:
        """
        Determine if an interaction is worth recording based on content analysis.
        """
        try:
            # Always consider for certain agent types that handle important data
            important_agents = {'health', 'finance', 'productivity', 'journal', 'scheduling'}
            if agent_type.lower() in important_agents:
                # But still filter obvious greetings/test messages
                if self._is_trivial_interaction(user_input, agent_response):
                    return False
                return True
            
            # Filter out trivial interactions
            if self._is_trivial_interaction(user_input, agent_response):
                return False
            
            # Check for valuable content patterns
            if self._contains_valuable_content(user_input, agent_response):
                return True
            
            # For complex interactions, use LLM analysis
            if len(user_input) > 100 or len(agent_response) > 200:
                return await self._llm_analysis(user_input, agent_response, agent_type)
            
            # Default to not recording for short, unclear interactions
            return False
            
        except Exception as e:
            self.logger.error("Error analyzing interaction value: %s", str(e))
            # On error, default to considering it recordable
            return True
    
    def _is_trivial_interaction(self, user_input: str, agent_response: str) -> bool:
        """Check if interaction is trivial and should be filtered out."""
        user_clean = user_input.strip().lower()
        response_clean = agent_response.strip().lower()
        
        # Check user input against trivial patterns
        for pattern in self.trivial_patterns:
            if re.match(pattern, user_clean, re.IGNORECASE):
                return True
        
        # Check for very short exchanges
        if len(user_clean) < 10 and len(response_clean) < 50:
            return True
        
        # Check for test/placeholder content
        if any(word in user_clean for word in ['test', 'testing', 'placeholder', 'example']):
            if len(user_clean) < 30:  # Only filter short test messages
                return True
        
        # Check for repetitive content
        if user_clean == response_clean.lower():
            return True
            
        return False
    
    def _contains_valuable_content(self, user_input: str, agent_response: str) -> bool:
        """Check if content contains valuable patterns worth recording."""
        combined_text = f"{user_input} {agent_response}".lower()
        
        for pattern in self.valuable_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True
                
        return False
    
    async def _llm_analysis(self, user_input: str, agent_response: str, agent_type: str) -> bool:
        """Use LLM to analyze if interaction is worth recording."""
        try:
            analysis_prompt = f"""
            Analyze if this interaction should be recorded to a knowledge base:
            
            User: {user_input}
            Agent ({agent_type}): {agent_response}
            
            Consider:
            - Does it contain useful information or preferences?
            - Would it help personalize future interactions?
            - Is it more than just casual conversation?
            
            Respond only: YES or NO
            """
            
            response = await self.llm_service.generate_response(analysis_prompt)
            
            return "yes" in response.lower()
            
        except Exception as e:
            self.logger.error("Error in LLM analysis: %s", str(e))
            return False
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """Get statistics about recording activity."""
        try:
            pending_count = len(self.pending_interactions)
            agents_with_pending = len(set(p.agent_type for p in self.pending_interactions))
            
            self.logger.info("Recording stats - pending count: %d, agents: %d, instance id: %s", 
                            pending_count, agents_with_pending, id(self))
            
            return {
                "pending_interactions": pending_count,
                "total_pending": pending_count,
                "agents_with_pending": agents_with_pending
            }
        except Exception as e:
            self.logger.error("Error getting recording stats: %s", str(e))
            return {"error": "Failed to get stats"}

# Per-user recorder instances
_recorders_by_user: Dict[str, InteractionRecorder] = {}


def get_interaction_recorder(knowledge_base_service=None, llm_service=None, user_id: Optional[str] = None):
    """Get or create a user-scoped interaction recorder instance."""
    from app.auth.user_context import get_current_user_id, normalize_user_storage_key
    from app.services.knowledge_base import get_knowledge_base_service

    resolved_user_id = normalize_user_storage_key(user_id or get_current_user_id())
    recorder = _recorders_by_user.get(resolved_user_id)
    resolved_kb_service = knowledge_base_service or get_knowledge_base_service(resolved_user_id)

    logger.info(
        "Getting interaction recorder for user=%s current instance=%s kb_service=%s llm_service=%s",
        resolved_user_id,
        id(recorder) if recorder else None,
        resolved_kb_service is not None,
        llm_service is not None,
    )

    if recorder is None:
        try:
            from app.llm import service as llm_service_module

            llm_svc = llm_service or llm_service_module._llm_service

            recorder = InteractionRecorder(resolved_kb_service, llm_svc)
            _recorders_by_user[resolved_user_id] = recorder
            logger.info("Created new interaction recorder instance for user=%s id=%s", resolved_user_id, id(recorder))
        except Exception as e:
            logger.error("Failed to initialize interaction recorder for user=%s: %s", resolved_user_id, str(e))
            return None
    else:
        # Keep existing pending approvals, but refresh dependencies after KB force-reset.
        if recorder.knowledge_base is not resolved_kb_service:
            logger.info(
                "Rebinding interaction recorder knowledge base for user=%s recorder=%s",
                resolved_user_id,
                id(recorder),
            )
            recorder.knowledge_base = resolved_kb_service

        if llm_service is not None and recorder.llm_service is not llm_service:
            recorder.llm_service = llm_service

    return recorder