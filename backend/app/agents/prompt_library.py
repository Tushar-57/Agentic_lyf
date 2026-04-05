"""Dedicated prompt templates for orchestration and routing quality."""

from __future__ import annotations

import json
from typing import Any, Dict


class OrchestratorPromptLibrary:
    """Prompt factory for data-grounded orchestrator prompts."""

    @staticmethod
    def build_intent_classification_prompt(
        *,
        user_input: str,
        complexity: str,
        agent_descriptions: Dict[str, str],
        context: Dict[str, Any] | None = None,
        profile_snapshot: Dict[str, Any] | None = None,
    ) -> str:
        """Build a strict classification prompt grounded in user profile and context."""
        context = context or {}
        profile_snapshot = profile_snapshot or {}

        compact_context: Dict[str, Any] = {}
        for key, value in context.items():
            if key in {"conversation_history", "state_manager"}:
                continue
            compact_context[key] = value

        profile_block = {
            "role": profile_snapshot.get("role"),
            "priorities": profile_snapshot.get("priorities", []),
            "preferred_tone": profile_snapshot.get("preferred_tone"),
            "mentor": profile_snapshot.get("mentor"),
            "active_goals": profile_snapshot.get("active_goals", []),
        }

        return f"""
    Route this user request to the best specialist agent.

    Your goal is not just domain matching. Choose routing that maximizes user improvement and immediate usefulness.

User request:
{user_input}

Task complexity signal: {complexity}

Available agents and domains:
{json.dumps(agent_descriptions, indent=2)}

User profile snapshot:
{json.dumps(profile_block, indent=2)}

Runtime context:
{json.dumps(compact_context, indent=2)}

Routing rules:
1. Prioritize semantic intent first, then user profile priorities.
2. If request mixes domains, choose the domain most likely to produce an immediate useful action.
3. Use GENERAL only when no specialist clearly fits.
4. Confidence should reflect certainty, not optimism.
5. For ambiguous "what should I do/work on now" prompts, favor actionable planning over generic conversation.
6. For "how did I do today" style prompts, favor review-oriented routes that can leverage history/time-entry data.

Return ONLY JSON with this exact schema:
{{
    "agent_type": "PRODUCTIVITY",
    "confidence": 0.86,
    "reason": "short reason",
    "primary_intent": "next_best_action",
    "expected_outcome": "prioritized_plan",
    "time_horizon": "now",
    "coach_tone_hint": "direct"
}}

Allowed agent_type values:
PRODUCTIVITY, HEALTH, FINANCE, SCHEDULING, JOURNAL, GENERAL

Allowed primary_intent values:
next_best_action, daily_review, planning, execution_help, reflection, tracking, question_answering, general_guidance

Allowed time_horizon values:
now, today, this_week, long_term, unspecified
""".strip()
