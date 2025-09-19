"""
Intelligent Interaction Recording Service

This service intelligently determines whether an interaction should be recorded
to the knowledge base based on content analysis and filters to prevent noise.
"""
import logging
import re
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class InteractionRecorder:
    """Intelligent service for recording valuable interactions to knowledge base."""
    
    def __init__(self, knowledge_base_service, llm_service):
        self.knowledge_base = knowledge_base_service
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)
        
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
        
    async def record_if_valuable(self, user_input: str, agent_response: str, 
                                agent_type: str = "general", context: Optional[Dict] = None) -> bool:
        """
        Analyze interaction and record only if it's valuable.
        
        Returns:
            bool: True if recorded, False if filtered out
        """
        try:
            # Check if interaction should be recorded
            should_record = await self.should_record_interaction(
                user_input, agent_response, agent_type, context
            )
            
            if should_record:
                # Record the interaction
                await self.knowledge_base.add_interaction_history(
                    agent_type=agent_type,
                    user_input=user_input,
                    agent_response=agent_response,
                    context=context or {}
                )
                
                self.logger.info(f"Recorded valuable interaction for {agent_type}")
                return True
            else:
                self.logger.debug(f"Filtered out trivial interaction for {agent_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in interaction recording: {e}")
            # On error, default to recording to be safe
            await self.knowledge_base.add_interaction_history(
                agent_type=agent_type,
                user_input=user_input,
                agent_response=agent_response,
                context=context or {}
            )
            return True
    
    async def should_record_interaction(self, user_input: str, agent_response: str, 
                                      agent_type: str = "general", context: Optional[Dict] = None) -> bool:
        """
        Determine if an interaction is worth recording based on content analysis.
        
        Args:
            user_input: User's message
            agent_response: Agent's response
            agent_type: Type of agent handling the interaction
            context: Additional context about the interaction
            
        Returns:
            bool: True if should record, False otherwise
        """
        try:
            # Always record for certain agent types that handle important data
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
            self.logger.error(f"Error analyzing interaction value: {e}")
            # On error, default to recording
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
        """Check if interaction contains valuable content patterns."""
        combined_text = f"{user_input} {agent_response}".lower()
        
        # Check for valuable content patterns
        for pattern in self.valuable_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                return True
        
        # Check for questions that indicate learning/information seeking
        question_indicators = ['how', 'what', 'why', 'when', 'where', 'which', 'can you', 'could you']
        if any(indicator in user_input.lower() for indicator in question_indicators):
            if len(user_input) > 20:  # Substantial questions
                return True
        
        # Check for actionable content
        action_words = ['create', 'build', 'make', 'develop', 'implement', 'setup', 'configure', 'install']
        if any(word in combined_text for word in action_words):
            return True
        
        return False
    
    async def _llm_analysis(self, user_input: str, agent_response: str, agent_type: str) -> bool:
        """Use LLM to analyze if interaction has long-term value."""
        try:
            analysis_prompt = f"""
            Analyze this interaction to determine if it should be recorded for future reference.
            
            User Input: "{user_input}"
            Agent Response: "{agent_response}"
            Agent Type: {agent_type}
            
            Consider these factors:
            - Does it contain personal preferences, goals, or important information?
            - Could this information be useful for future interactions?
            - Is it more than just casual conversation or troubleshooting?
            - Does it contain domain-specific knowledge or insights?
            - Would losing this interaction impact future personalization?
            
            Respond with only "YES" if it should be recorded, or "NO" if it should be filtered out.
            """
            
            response = await self.llm_service.get_completion(
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            result = response.content.strip().upper()
            return result == "YES"
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {e}")
            # On error, default to recording
            return True
    
    async def get_recording_stats(self) -> Dict[str, Any]:
        """Get statistics about recording behavior."""
        try:
            # This would be implemented based on your tracking needs
            return {
                "total_analyzed": 0,
                "recorded": 0,
                "filtered": 0,
                "filter_rate": 0.0,
                "last_analysis": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting recording stats: {e}")
            return {}


# Global instance
_interaction_recorder = None

def get_interaction_recorder():
    """Get the global interaction recorder instance."""
    global _interaction_recorder
    
    if _interaction_recorder is None:
        from .knowledge_base import get_knowledge_base_service
        from ..llm.service import get_llm_service
        
        knowledge_base = get_knowledge_base_service()
        llm_service = get_llm_service()
        _interaction_recorder = InteractionRecorder(knowledge_base, llm_service)
    
    return _interaction_recorder