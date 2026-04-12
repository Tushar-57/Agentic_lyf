"""
Finance Agent - Specialized agent for financial management, budgeting, and expense tracking.
"""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..base import BaseAgent, AgentType, AgentCapability, AgentState
from ..prompts import get_agent_prompt
from ...llm.service import get_llm_service
from ...llm.base import CompletionRequest, ChatMessage
from ...services.knowledge_base import get_knowledge_base_service
from ...models.knowledge import KnowledgeEntryType
from ...services.interaction_recorder import get_interaction_recorder
from ...utils.structured_logging import get_logger, LogComponent

logger = get_logger(__name__, LogComponent.AGENT)


class FinanceAgent(BaseAgent):
    """Specialized agent for financial management, budgeting, and expense tracking."""
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="budget_management",
                description="Create and manage personal budgets based on income and expenses",
                parameters={"income_tracking": True, "expense_categories": True}
            ),
            AgentCapability(
                name="expense_tracking",
                description="Track and categorize expenses with insights and recommendations",
                parameters={"category_analysis": True, "spending_patterns": True}
            ),
            AgentCapability(
                name="financial_goals",
                description="Set and track financial goals like savings, debt reduction, investments",
                parameters={"goal_tracking": True, "progress_monitoring": True}
            ),
            AgentCapability(
                name="financial_advice",
                description="Provide personalized financial advice and recommendations",
                parameters={"risk_assessment": True, "investment_suggestions": True}
            )
        ]
        
        super().__init__(
            agent_id=f"finance_{uuid.uuid4().hex[:8]}",
            agent_type=AgentType.FINANCE,
            capabilities=capabilities,
            system_prompt=get_agent_prompt(AgentType.FINANCE)
        )
        
        self.knowledge_base = get_knowledge_base_service()
    
    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute finance-related requests with contextual knowledge."""
        try:
            user_input = state.get("user_input", "")
            state_context = state.get("context", {}) if isinstance(state, dict) else {}
            logger.info("finance_processing", "FinanceAgent processing", {"input_preview": user_input[:100]})
            
            # Get contextual knowledge from knowledge base
            contextual_knowledge = await self.knowledge_base.get_contextual_knowledge_for_agent(
                user_input=user_input,
                agent_type="finance",
                max_results=10
            )

            merged_context = self._merge_with_routing_context(contextual_knowledge, state_context)
            
            # Determine finance task type
            if any(keyword in user_input.lower() for keyword in ["budget", "budgeting", "monthly budget"]):
                response = await self._handle_budget_planning(user_input, merged_context)
            elif any(keyword in user_input.lower() for keyword in ["expense", "spending", "track expenses"]):
                response = await self._handle_expense_tracking(user_input, merged_context)
            elif any(keyword in user_input.lower() for keyword in ["save", "savings", "financial goal"]):
                response = await self._handle_financial_goals(user_input, merged_context)
            else:
                response = await self._handle_general_finance(user_input, merged_context)
            
            # Create pending interaction for explicit user approval.
            recorder = get_interaction_recorder()
            if recorder:
                interaction_id = await recorder.create_pending_interaction(
                    user_input=user_input,
                    agent_response=response,
                    agent_type="finance",
                    context=merged_context,
                )
                if interaction_id:
                    logger.info("Created pending interaction %s for finance approval", interaction_id)
            
            return {
                "response": response,
                "reasoning": {
                    "agent_type": "finance",
                    "context_used": len(merged_context.get("relevant_interactions", [])),
                    "profile_role": (merged_context.get("profile_snapshot") or {}).get("role"),
                    "coach_style": (merged_context.get("coach_profile") or {}).get("style"),
                    "specialized_handling": True
                }
            }
            
        except Exception as e:
            logger.error("finance_execution_failed", "FinanceAgent execution failed", error=e)
            return {
                "response": "I'm having trouble with financial analysis right now. Please try again later.",
                "reasoning": {"error": str(e), "agent_type": "finance"}
            }
    
    async def _handle_budget_planning(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle budget planning requests."""
        try:
            budget_context = self._build_finance_context(context, "budget")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "💰 I'd be happy to help with budget planning! Could you share your monthly income and main expense categories?"
            
            prompt = f"""
            As a financial advisor, create a personalized budget plan based on the user's request and financial context.
            
            User Request: {user_input}
            Financial Context: {budget_context}
            
            Provide a detailed budget plan with:
            1. Budget breakdown by categories (housing, food, transportation, etc.)
            2. Savings recommendations (50/30/20 rule or adjusted for their situation)
            3. Specific actionable tips for their financial situation
            4. Monthly tracking suggestions
            
            Use emojis and format nicely with clear sections.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=420
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error("budget_planning_failed", "Budget planning failed", error=e)
            return "💰 I'd be happy to help you create a personalized budget! Could you share your monthly income and main expense categories so I can provide specific recommendations?"
    
    async def _handle_expense_tracking(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle expense tracking requests."""
        try:
            expense_context = self._build_finance_context(context, "expenses")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "📊 I'd be happy to help track your expenses! What specific expenses would you like to track?"
            
            prompt = f"""
            As a financial advisor, help the user with expense tracking and analysis.
            
            User Request: {user_input}
            Expense Context: {expense_context}
            
            Provide:
            1. Expense tracking recommendations and tools
            2. Category suggestions for their lifestyle
            3. Analysis of spending patterns if available
            4. Tips to reduce unnecessary expenses
            5. Monthly review suggestions
            
            Use emojis and format nicely with actionable advice.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=420
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error("expense_tracking_failed", "Expense tracking failed", error=e)
            return "📊 I'd be happy to help you track your expenses effectively! What specific expense categories would you like to focus on?"
    
    async def _handle_financial_goals(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle financial goal setting and tracking."""
        try:
            goals_context = self._build_finance_context(context, "goals")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "🎯 I'd be happy to help with your financial goals! What specific financial objectives do you have in mind?"
            
            prompt = f"""
            As a financial advisor, help the user set and achieve their financial goals.
            
            User Request: {user_input}
            Financial Goals Context: {goals_context}
            
            Provide:
            1. SMART financial goal framework
            2. Specific savings strategies for their goals
            3. Timeline recommendations
            4. Progress tracking methods
            5. Motivation and milestone celebration ideas
            
            Use emojis and format nicely with actionable steps.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=420
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error("financial_goals_failed", "Financial goals failed", error=e)
            return "🎯 I'd be happy to help you set and achieve your financial goals! What specific financial objectives do you have in mind?"
    
    async def _handle_general_finance(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle general financial queries."""
        try:
            finance_context = self._build_finance_context(context, "general")
            
            llm_service = await get_llm_service()
            if not llm_service:
                return "💰 I'm here to help with your financial questions! What specific financial topic would you like assistance with?"
            
            prompt = f"""
            As a financial advisor, provide helpful advice for the user's financial question.
            
            User Request: {user_input}
            Financial Context: {finance_context}
            
            Provide relevant financial advice, tips, and recommendations based on their question.
            Use emojis and format nicely with clear, actionable information.
            """
            
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.3,
                max_tokens=320
            )
            
            response = await llm_service.chat_completion(request)
            return response.content
            
        except Exception as e:
            logger.error("general_finance_failed", "General finance failed", error=e)
            return "💰 I'm here to help with your financial questions! What specific financial topic would you like assistance with?"
    
    def _build_finance_context(self, context: Dict[str, Any], finance_type: str) -> str:
        """Build financial context from available knowledge."""
        context_parts = []

        profile_snapshot = context.get("profile_snapshot", {}) if isinstance(context, dict) else {}
        if isinstance(profile_snapshot, dict):
            role = profile_snapshot.get("role")
            priorities = profile_snapshot.get("priorities", []) if isinstance(profile_snapshot.get("priorities"), list) else []
            if role or priorities:
                context_parts.append(
                    f"Profile: role={role or 'unknown'} | priorities={', '.join(str(item) for item in priorities[:3])}"
                )

        coach_profile = context.get("coach_profile", {}) if isinstance(context, dict) else {}
        if isinstance(coach_profile, dict) and coach_profile:
            context_parts.append(
                f"Coach style: {coach_profile.get('style')} | directive: {coach_profile.get('directive')}"
            )

        intent_blueprint = context.get("intent_blueprint", {}) if isinstance(context, dict) else {}
        if isinstance(intent_blueprint, dict) and intent_blueprint:
            context_parts.append(
                "Intent blueprint: "
                f"intent={intent_blueprint.get('primary_intent')} | "
                f"outcome={intent_blueprint.get('expected_outcome')}"
            )
        
        # Add agent preferences (financial preferences from knowledge base)
        if "agent_preferences" in context and context["agent_preferences"]:
            prefs = context["agent_preferences"]
            if isinstance(prefs, dict):
                finance_prefs = {k: v for k, v in prefs.items() if any(term in k.lower() for term in ["finance", "budget", "income", "expense", "saving"])}
                if finance_prefs:
                    context_parts.append(f"Financial preferences: {finance_prefs}")
        
        # Add context summary
        if "context_summary" in context and context["context_summary"]:
            context_parts.append(f"Previous financial context: {context['context_summary']}")
        
        return " | ".join(context_parts) if context_parts else f"No specific {finance_type} context available"

    def _merge_with_routing_context(self, kb_context: Dict[str, Any], state_context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge orchestrator routing context with finance KB context."""
        merged_context = dict(kb_context or {})
        if not isinstance(state_context, dict):
            return merged_context

        for key in ("profile_snapshot", "coach_profile", "intent_blueprint", "knowledge_context_summary"):
            value = state_context.get(key)
            if value:
                merged_context[key] = value

        return merged_context