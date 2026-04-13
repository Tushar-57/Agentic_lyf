"""
Enhanced Productivity Agent - Actionable productivity with smart time analysis.

Integrates SmartTimeContextAnalyzer to provide:
- Work type categorization (deep work, meetings, admin, learning)
- Goal alignment analysis
- Actionable recommendations with workflow triggers
- Multi-agent coordination for schedule optimization
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ..base import BaseAgent, AgentType, AgentCapability, AgentState
from ..prompts import get_agent_prompt
from ...llm.service import get_llm_service
from ...llm.base import CompletionRequest, ChatMessage
from ...services.knowledge_base import get_knowledge_base_service
from ...services.time_context_analyzer import (
    get_time_analyzer, 
    TimeWindowAnalysis, 
    WorkType,
    EnergyPattern,
    SmartTimeContextAnalyzer
)
from ...services.interaction_recorder import get_interaction_recorder
from ...agents.multi_agent_workflow import (
    get_workflow_coordinator,
    WorkflowStepType,
    create_productivity_optimization_workflow,
    create_goal_alignment_workflow
)
from ...utils.structured_logging import get_logger, LogComponent

logger = get_logger(__name__, LogComponent.AGENT)


class ActionableSuggestion:
    """Represents an actionable suggestion the user can execute."""
    
    def __init__(
        self,
        title: str,
        description: str,
        action_type: str,  # "schedule", "reminder", "workflow", "manual"
        parameters: Dict[str, Any],
        estimated_impact: str,
        time_required: Optional[int] = None,
        workflow_id: Optional[str] = None
    ):
        self.title = title
        self.description = description
        self.action_type = action_type
        self.parameters = parameters
        self.estimated_impact = estimated_impact
        self.time_required = time_required
        self.workflow_id = workflow_id


class EnhancedProductivityAgent(BaseAgent):
    """
    Enhanced productivity agent with smart time analysis and actionable outputs.
    
    Improvements over base ProductivityAgent:
    1. Categorizes 6000 minutes into deep work, meetings, admin, learning
    2. Identifies specific optimization opportunities
    3. Generates actionable suggestions with one-click execution
    4. Triggers multi-agent workflows for complex optimizations
    5. Provides quantified recommendations ("Block 6:00-6:30 PM for LeetCode")
    """
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                name="smart_time_analysis",
                description="Categorize time into work types (deep, meetings, admin, learning)",
                parameters={"categorization": True, "pattern_detection": True}
            ),
            AgentCapability(
                name="actionable_recommendations",
                description="Generate specific, executable recommendations with scheduling",
                parameters={"schedule_integration": True, "quantified_suggestions": True}
            ),
            AgentCapability(
                name="multi_agent_coordination",
                description="Coordinate with scheduling and notification agents",
                parameters={"workflow_triggers": True, "handoff_capability": True}
            ),
            AgentCapability(
                name="goal_gap_analysis",
                description="Identify specific gaps between goals and actual time allocation",
                parameters={"alignment_scoring": True, "deficit_calculation": True}
            ),
        ]
        
        super().__init__(
            agent_id="productivity_enhanced",
            agent_type=AgentType.PRODUCTIVITY,
            capabilities=capabilities,
            system_prompt=self._build_enhanced_system_prompt()
        )
        
        self.knowledge_base = get_knowledge_base_service()
        self.time_analyzer = get_time_analyzer()
        self.workflow_coordinator = get_workflow_coordinator()
    
    def _build_enhanced_system_prompt(self) -> str:
        return """You are an advanced productivity coach with deep time analytics capabilities.

Your analysis goes beyond "6000 minutes of work" to categorize time into:
- Deep Work: Focused, high-value cognitive work (coding, problem-solving)
- Learning: Skill development (LeetCode, courses, reading)
- Meetings: Synchronous communication
- Admin: Shallow work (emails, tickets, quick tasks)
- Context Switching: Fragmented short sessions

Guidelines:
1. Always categorize time by work type, not just report total hours
2. Calculate specific deficits: "You spent 45min on LeetCode vs 30min daily goal = 15min deficit"
3. Identify concrete optimization opportunities with time slots
4. Provide actionable suggestions with estimated impact
5. Trigger multi-agent workflows when scheduling changes are needed
6. Quantify recommendations: "Block 6:00-6:30 PM (previously idle 45min)"

Response format:
- Summary: High-level time breakdown
- Insights: 3-5 pattern observations with data
- Goal Gaps: Specific deficits with numbers
- Actionable Suggestions: Executable actions with impact estimates"""

    async def execute(self, state: AgentState) -> Dict[str, Any]:
        """Execute enhanced productivity analysis with actionable outputs."""
        try:
            user_input = state.get("user_input", "")
            state_context = state.get("context", {}) if isinstance(state, dict) else {}
            
            logger.info(
                "enhanced_productivity_processing",
                "EnhancedProductivityAgent processing",
                {"input_preview": user_input[:100]}
            )
            
            # Get contextual knowledge
            contextual_knowledge = await self.knowledge_base.get_contextual_knowledge_for_agent(
                user_input=user_input,
                agent_type="productivity",
                max_results=10
            )
            
            merged_context = self._merge_with_routing_context(contextual_knowledge, state_context)
            
            # Perform smart time analysis
            time_analysis = await self._analyze_time_context(merged_context)
            
            # Determine request type and handle
            normalized_input = user_input.lower()
            
            if self._is_performance_review_request(normalized_input):
                response_data = await self._handle_enhanced_performance_review(
                    user_input, merged_context, time_analysis
                )
            elif any(kw in normalized_input for kw in ["what should i do", "next steps", "recommendation"]):
                response_data = await self._handle_actionable_recommendations(
                    user_input, merged_context, time_analysis
                )
            elif any(kw in normalized_input for kw in ["goal", "objective", "priority"]):
                response_data = await self._handle_goal_gap_analysis(
                    user_input, merged_context, time_analysis
                )
            else:
                response_data = await self._handle_general_enhanced(
                    user_input, merged_context, time_analysis
                )
            
            # Format response with actionable sections
            formatted_response = self._format_actionable_response(response_data)
            
            # Create pending interaction
            recorder = get_interaction_recorder()
            if recorder:
                await recorder.create_pending_interaction(
                    user_input=user_input,
                    agent_response=formatted_response,
                    agent_type="productivity_enhanced",
                    context={
                        **merged_context,
                        "time_analysis": time_analysis,
                        "suggestions": response_data.get("suggestions", [])
                    }
                )
            
            return {
                "response": formatted_response,
                "reasoning": {
                    "agent_type": "productivity_enhanced",
                    "time_analysis_summary": time_analysis.pattern_insights[:3] if time_analysis else [],
                    "suggestions_count": len(response_data.get("suggestions", [])),
                    "workflow_triggered": response_data.get("workflow_triggered", False),
                },
                "actionable_data": response_data  # For frontend action buttons
            }
            
        except Exception as e:
            logger.error("enhanced_productivity_execution_failed", "Execution failed", error=e)
            return {
                "response": "I encountered an error analyzing your productivity patterns. Please try again.",
                "reasoning": {"error": str(e), "agent_type": "productivity_enhanced"}
            }
    
    async def _analyze_time_context(
        self,
        context: Dict[str, Any]
    ) -> Optional[TimeWindowAnalysis]:
        """Perform smart time analysis on user's time entries."""
        try:
            # Extract time entries from context
            time_entries = context.get("recent_time_entries", [])
            time_window_summary = context.get("time_window_summary", {})
            
            if time_window_summary.get("top_entries"):
                time_entries = time_window_summary.get("top_entries", [])
            
            if not time_entries:
                return None
            
            # Get user priorities for goal alignment
            profile = context.get("profile_snapshot", {})
            priorities = profile.get("priorities", []) if isinstance(profile, dict) else []
            
            # Perform analysis
            window_label = time_window_summary.get("window_label", "Recent period")
            analysis = self.time_analyzer.analyze_time_window(
                entries=time_entries,
                window_label=window_label,
                user_priorities=priorities
            )
            
            logger.info(
                "time_analysis_complete",
                f"Analyzed {len(time_entries)} entries",
                {
                    "total_minutes": analysis.total_minutes,
                    "deep_work_min": analysis.categorized_breakdown.get(WorkType.DEEP_WORK, 0),
                    "learning_min": analysis.categorized_breakdown.get(WorkType.LEARNING, 0),
                    "opportunities": len(analysis.optimization_opportunities)
                }
            )
            
            return analysis
            
        except Exception as e:
            logger.error("time_analysis_error", "Failed to analyze time context", error=e)
            return None
    
    async def _handle_enhanced_performance_review(
        self,
        user_input: str,
        context: Dict[str, Any],
        time_analysis: Optional[TimeWindowAnalysis]
    ) -> Dict[str, Any]:
        """Handle performance review with enhanced time categorization."""
        
        if not time_analysis:
            return {
                "summary": "No time tracking data available for analysis.",
                "insights": [],
                "suggestions": []
            }
        
        # Calculate detailed metrics
        total_minutes = time_analysis.total_minutes
        breakdown = time_analysis.categorized_breakdown
        opportunities = time_analysis.optimization_opportunities
        
        # Calculate percentages
        deep_work_pct = (breakdown.get(WorkType.DEEP_WORK, 0) / total_minutes * 100) if total_minutes > 0 else 0
        learning_pct = (breakdown.get(WorkType.LEARNING, 0) / total_minutes * 100) if total_minutes > 0 else 0
        meeting_pct = (breakdown.get(WorkType.MEETINGS, 0) / total_minutes * 100) if total_minutes > 0 else 0
        admin_pct = (breakdown.get(WorkType.SHALLOW_WORK, 0) / total_minutes * 100) if total_minutes > 0 else 0
        
        # Build insights
        insights = time_analysis.pattern_insights.copy()
        
        # Add specific insights based on analysis
        if deep_work_pct < 20:
            insights.append(f"Deep work at {deep_work_pct:.0f}% - recommend increasing to 30% for high-impact output")
        
        if learning_pct < 5:
            insights.append(f"Learning investment at {learning_pct:.0f}% - consider deliberate practice blocks")
        
        if meeting_pct > 30:
            insights.append(f"Meeting load at {meeting_pct:.0f}% - may be crowding out deep work")
        
        # Generate actionable suggestions
        suggestions = []
        
        # Suggestion 1: Deep work optimization
        if deep_work_pct < 20:
            # Find idle gaps for deep work
            idle_gaps = [g for g in time_analysis.gaps_detected if g.get("duration_minutes", 0) > 45]
            if idle_gaps:
                best_gap = max(idle_gaps, key=lambda g: g.get("duration_minutes", 0))
                suggestions.append(ActionableSuggestion(
                    title="Create Deep Work Block",
                    description=f"Block {best_gap['duration_minutes']:.0f}min idle time for focused work",
                    action_type="workflow",
                    parameters={
                        "duration_minutes": 90,
                        "work_type": "deep_work",
                        "preferred_time": "morning"
                    },
                    estimated_impact="+15% deep work ratio, improved output quality",
                    time_required=90,
                    workflow_id="productivity_optimization"
                ))
        
        # Suggestion 2: Learning time block
        if learning_pct < 10:
            suggestions.append(ActionableSuggestion(
                title="Schedule Daily Learning Block",
                description="Add 30-min daily LeetCode/skill practice during low-energy periods",
                action_type="workflow",
                parameters={
                    "duration_minutes": 30,
                    "work_type": "learning",
                    "frequency": "daily"
                },
                estimated_impact="+5% learning ratio, skill development consistency",
                time_required=30,
                workflow_id="productivity_optimization"
            ))
        
        # Suggestion 3: Batch admin tasks
        if admin_pct > 35:
            suggestions.append(ActionableSuggestion(
                title="Batch Admin Tasks",
                description="Group shallow work into 2x 30min blocks instead of scattered sessions",
                action_type="manual",
                parameters={"strategy": "time_blocking"},
                estimated_impact="-20% context switching, +10% deep work capacity"
            ))
        
        # Trigger workflow if significant optimization needed
        workflow_triggered = len(opportunities) >= 2
        
        return {
            "summary": {
                "total_time": f"{total_minutes/60:.1f} hours",
                "deep_work": f"{breakdown.get(WorkType.DEEP_WORK, 0)/60:.1f}h ({deep_work_pct:.0f}%)",
                "learning": f"{breakdown.get(WorkType.LEARNING, 0)/60:.1f}h ({learning_pct:.0f}%)",
                "meetings": f"{breakdown.get(WorkType.MEETINGS, 0)/60:.1f}h ({meeting_pct:.0f}%)",
                "admin": f"{breakdown.get(WorkType.SHALLOW_WORK, 0)/60:.1f}h ({admin_pct:.0f}%)",
            },
            "insights": insights,
            "optimization_opportunities": opportunities,
            "suggestions": suggestions,
            "workflow_triggered": workflow_triggered,
            "time_analysis": time_analysis
        }
    
    async def _handle_actionable_recommendations(
        self,
        user_input: str,
        context: Dict[str, Any],
        time_analysis: Optional[TimeWindowAnalysis]
    ) -> Dict[str, Any]:
        """Generate specific, actionable recommendations."""
        
        suggestions = []
        
        # Get profile info
        profile = context.get("profile_snapshot", {})
        priorities = profile.get("priorities", []) if isinstance(profile, dict) else []
        
        # Analyze idle gaps for opportunities
        if time_analysis and time_analysis.gaps_detected:
            idle_gaps = sorted(
                [g for g in time_analysis.gaps_detected if g.get("duration_minutes", 0) > 30],
                key=lambda g: g.get("duration_minutes", 0),
                reverse=True
            )
            
            if idle_gaps and priorities:
                # Suggest using largest idle gap for top priority
                top_priority = priorities[0]
                best_gap = idle_gaps[0]
                gap_minutes = best_gap.get("duration_minutes", 0)
                
                suggestions.append(ActionableSuggestion(
                    title=f"Block Time for {top_priority}",
                    description=f"Use {gap_minutes:.0f}min idle window for {top_priority}",
                    action_type="workflow",
                    parameters={
                        "duration_minutes": min(gap_minutes, 60),
                        "activity": top_priority,
                        "use_idle_gap": True
                    },
                    estimated_impact=f"Progress on {top_priority} without sacrificing other commitments",
                    workflow_id="goal_alignment"
                ))
        
        # LeetCode specific suggestion
        if any("leetcode" in p.lower() for p in priorities):
            suggestions.append(ActionableSuggestion(
                title="Daily LeetCode Block",
                description="Schedule 6:00-6:30 PM for LeetCode practice (30min)",
                action_type="schedule",
                parameters={
                    "title": "LeetCode Practice",
                    "duration_minutes": 30,
                    "preferred_time": "18:00",
                    "recurrence": "daily"
                },
                estimated_impact="21 problems/month, consistent skill building",
                time_required=30
            ))
        
        # Skill development suggestion
        if any("skill" in p.lower() for p in priorities):
            suggestions.append(ActionableSuggestion(
                title="Morning Learning Block",
                description="Protect 8:00-8:45 AM for skill development before reactive work",
                action_type="schedule",
                parameters={
                    "title": "Skill Development",
                    "duration_minutes": 45,
                    "preferred_time": "08:00",
                    "recurrence": "weekdays"
                },
                estimated_impact="3.75 hours/week of focused skill building",
                time_required=45
            ))
        
        return {
            "summary": "Based on your patterns and priorities, here are actionable recommendations:",
            "insights": ["Identified optimal time slots from your idle gaps", "Aligned with your stated priorities"],
            "suggestions": suggestions,
            "workflow_triggered": len(suggestions) > 0
        }
    
    async def _handle_goal_gap_analysis(
        self,
        user_input: str,
        context: Dict[str, Any],
        time_analysis: Optional[TimeWindowAnalysis]
    ) -> Dict[str, Any]:
        """Analyze gaps between goals and actual time spent."""
        
        profile = context.get("profile_snapshot", {})
        priorities = profile.get("priorities", []) if isinstance(profile, dict) else []
        active_goals = profile.get("active_goals", []) if isinstance(profile, dict) else []
        
        if not time_analysis or not priorities:
            return {
                "summary": "Insufficient data for goal gap analysis",
                "insights": [],
                "suggestions": []
            }
        
        # Calculate goal coverage
        goal_coverage = time_analysis.goal_coverage
        total_minutes = time_analysis.total_minutes
        
        gaps = []
        suggestions = []
        
        for priority in priorities[:3]:
            priority_key = f"priority:{priority}"
            minutes_spent = goal_coverage.get(priority_key, 0)
            
            # Calculate deficit (assume 30min/day = 3.5hrs/week target)
            target_minutes = 210  # 30min * 7 days
            deficit = max(0, target_minutes - minutes_spent)
            deficit_pct = (deficit / target_minutes * 100) if target_minutes > 0 else 0
            
            if deficit > 60:
                gaps.append({
                    "priority": priority,
                    "minutes_spent": minutes_spent,
                    "target_minutes": target_minutes,
                    "deficit": deficit,
                    "deficit_pct": deficit_pct
                })
                
                # Create specific suggestion
                specific_time = self._suggest_optimal_time(time_analysis, deficit)
                
                suggestions.append(ActionableSuggestion(
                    title=f"Close Gap: {priority}",
                    description=f"You spent {minutes_spent/60:.1f}h on {priority} vs 3.5h target. "
                               f"Deficit: {deficit/60:.1f}h ({deficit_pct:.0f}%)",
                    action_type="schedule",
                    parameters={
                        "title": priority,
                        "duration_minutes": min(deficit / 5, 60),  # Spread over remaining days
                        "specific_time": specific_time
                    },
                    estimated_impact=f"Close {deficit_pct:.0f}% of gap in one week",
                    time_required=int(min(deficit / 5, 60))
                ))
        
        return {
            "summary": f"Analyzed {len(priorities)} priorities against {total_minutes/60:.1f}h tracked time",
            "goal_gaps": gaps,
            "insights": [
                f"{len(gaps)} priorities have significant time deficits" if gaps else "All priorities well-covered",
                f"Goal alignment: {len(goal_coverage)/len(priorities)*100:.0f}% of priorities tracked"
            ],
            "suggestions": suggestions,
            "workflow_triggered": len(gaps) > 0,
            "time_analysis": time_analysis
        }
    
    def _suggest_optimal_time(
        self,
        time_analysis: TimeWindowAnalysis,
        duration_needed: float
    ) -> str:
        """Suggest optimal time slot based on patterns."""
        # Find idle gaps that fit the duration
        suitable_gaps = [
            g for g in time_analysis.gaps_detected
            if g.get("duration_minutes", 0) >= duration_needed
        ]
        
        if suitable_gaps:
            # Return the largest gap's start time (simplified)
            best_gap = max(suitable_gaps, key=lambda g: g.get("duration_minutes", 0))
            return best_gap.get("start", "flexible")
        
        # Default suggestions based on work patterns
        energy_dist = time_analysis.energy_distribution
        if energy_dist.get(EnergyPattern.HIGH_FOCUS, 0) > 0:
            return "during peak focus hours (likely morning)"
        
        return "evening (post-work hours)"
    
    async def _handle_general_enhanced(
        self,
        user_input: str,
        context: Dict[str, Any],
        time_analysis: Optional[TimeWindowAnalysis]
    ) -> Dict[str, Any]:
        """Handle general productivity queries with enhanced context."""
        
        # Use LLM with enhanced context
        llm_service = await get_llm_service()
        if not llm_service:
            return {
                "summary": "Unable to generate recommendations at this time.",
                "insights": [],
                "suggestions": []
            }
        
        # Build rich prompt with time analysis
        analysis_context = ""
        if time_analysis:
            breakdown = time_analysis.categorized_breakdown
            total = time_analysis.total_minutes
            
            analysis_context = f"""
Time Analysis ({time_analysis.window_label}):
- Total: {total/60:.1f} hours
- Deep Work: {breakdown.get(WorkType.DEEP_WORK, 0)/60:.1f}h ({breakdown.get(WorkType.DEEP_WORK, 0)/total*100:.0f}%)
- Learning: {breakdown.get(WorkType.LEARNING, 0)/60:.1f}h ({breakdown.get(WorkType.LEARNING, 0)/total*100:.0f}%)
- Meetings: {breakdown.get(WorkType.MEETINGS, 0)/60:.1f}h ({breakdown.get(WorkType.MEETINGS, 0)/total*100:.0f}%)
- Admin: {breakdown.get(WorkType.SHALLOW_WORK, 0)/60:.1f}h ({breakdown.get(WorkType.SHALLOW_WORK, 0)/total*100:.0f}%)

Optimization Opportunities: {len(time_analysis.optimization_opportunities)}
Idle Gaps Available: {len(time_analysis.gaps_detected)}
"""
        
        prompt = f"""
You are a productivity coach. Provide specific, actionable advice.

User Query: {user_input}

{analysis_context}

Response Requirements:
1. Categorize time by work type (deep/learning/meetings/admin) not just total hours
2. Provide 3 specific insights with numbers
3. Suggest 2-3 concrete actions with time slots
4. Include one multi-agent workflow suggestion if scheduling changes needed

Format:
## Summary
[Brief overview with work type breakdown]

## Key Insights
1. [Observation with specific data]
2. [Pattern identified]
3. [Optimization opportunity]

## Actionable Recommendations
- [Specific action with time slot]
- [Another action]

## Suggested Workflow
[If complex changes needed, describe multi-agent coordination]
"""
        
        request = CompletionRequest(
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.3,
            max_tokens=600
        )
        
        response = await llm_service.chat_completion(request)
        
        # Parse suggestions from response (simplified)
        suggestions = []
        if time_analysis and time_analysis.optimization_opportunities:
            for opp in time_analysis.optimization_opportunities[:2]:
                suggestions.append(ActionableSuggestion(
                    title=opp.get("type", "Optimization"),
                    description=opp.get("suggestion", ""),
                    action_type="workflow",
                    parameters={},
                    estimated_impact=opp.get("action", ""),
                    workflow_id="productivity_optimization"
                ))
        
        return {
            "summary": response.content or "Analysis complete",
            "insights": time_analysis.pattern_insights if time_analysis else [],
            "suggestions": suggestions,
            "workflow_triggered": len(suggestions) > 0,
            "full_response": response.content
        }
    
    def _format_actionable_response(self, response_data: Dict[str, Any]) -> str:
        """Format response with actionable sections."""
        lines = []
        
        # Summary
        summary = response_data.get("summary", {})
        if isinstance(summary, dict):
            lines.append("## Time Breakdown (Categorized)")
            for key, value in summary.items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        else:
            lines.append(f"## Summary\n{summary}")
        
        lines.append("")
        
        # Insights
        insights = response_data.get("insights", [])
        if insights:
            lines.append("## Key Insights")
            for i, insight in enumerate(insights[:5], 1):
                lines.append(f"{i}. {insight}")
            lines.append("")
        
        # Goal Gaps
        gaps = response_data.get("goal_gaps", [])
        if gaps:
            lines.append("## Goal Gaps Identified")
            for gap in gaps:
                lines.append(f"- **{gap['priority']}**: {gap['minutes_spent']/60:.1f}h spent vs {gap['target_minutes']/60:.1f}h target")
                lines.append(f"  - Deficit: {gap['deficit']/60:.1f}h ({gap['deficit_pct']:.0f}%)")
            lines.append("")
        
        # Optimization Opportunities
        opps = response_data.get("optimization_opportunities", [])
        if opps:
            lines.append("## Optimization Opportunities")
            for opp in opps[:3]:
                lines.append(f"- **{opp.get('type', 'Opportunity')}** ({opp.get('severity', 'medium')})")
                lines.append(f"  - {opp.get('suggestion', '')}")
            lines.append("")
        
        # Actionable Suggestions
        suggestions = response_data.get("suggestions", [])
        if suggestions:
            lines.append("## 🎯 Actionable Suggestions")
            lines.append("")
            for i, suggestion in enumerate(suggestions[:3], 1):
                lines.append(f"### {i}. {suggestion.title}")
                lines.append(f"{suggestion.description}")
                if suggestion.time_required:
                    lines.append(f"⏱️ **Time Required**: {suggestion.time_required} minutes")
                lines.append(f"📈 **Estimated Impact**: {suggestion.estimated_impact}")
                if suggestion.action_type == "workflow":
                    lines.append(f"⚡ **One-Click Action**: Trigger workflow to implement")
                lines.append("")
        
        # Full response if available
        full_response = response_data.get("full_response")
        if full_response and not lines:
            return full_response
        
        return "\n".join(lines)
    
    def _is_performance_review_request(self, normalized_input: str) -> bool:
        """Check if input is requesting performance review."""
        return bool(re.search(
            r"\b(how did i do|review|performance|analyze|breakdown|summary|time analysis)\b",
            normalized_input
        ))
    
    def _merge_with_routing_context(
        self,
        kb_context: Dict[str, Any],
        state_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge orchestrator routing context with KB context."""
        merged = dict(kb_context or {})
        if not isinstance(state_context, dict):
            return merged
        
        for key in ("profile_snapshot", "coach_profile", "intent_blueprint", 
                    "knowledge_context_summary", "time_window_summary"):
            if key in state_context:
                merged[key] = state_context[key]
        
        return merged


# Singleton
_enhanced_agent: Optional[EnhancedProductivityAgent] = None


def get_enhanced_productivity_agent() -> EnhancedProductivityAgent:
    """Get or create the enhanced productivity agent."""
    global _enhanced_agent
    if _enhanced_agent is None:
        _enhanced_agent = EnhancedProductivityAgent()
    return _enhanced_agent
