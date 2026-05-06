"""
BrainService — thin façade over the knowledge base providing three semantic
memory types for the agent layer:

  • Episodic — past conversation turns (user side only). What we talked about.
  • Semantic — computed INSIGHT / PATTERN entries (intelligence). What we know.
  • Working — recent classified behavioral data (last 30 days) across
    time_entry_v2, habit_entry, morning_intent, evening_reflection. What's
    happening right now.

Working memory in the agent state itself (DeepAgentState) stays separate;
this layer is exclusively a Pinecone-backed wrapper.

Storage helpers are exposed alongside recall so agents can persist new
episodic turns and the precompute service can persist semantic insights
through one consistent interface.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.models.knowledge import (
    KnowledgeEntry,
    KnowledgeEntryType,
    KnowledgeEntrySubType,
    KnowledgeQuery,
)

logger = logging.getLogger(__name__)


WORKING_MEMORY_CATEGORIES = (
    "time_entry_v2",
    "habit_entry",
    "daily_habit_summary",
    "morning_intent",
    "evening_reflection",
)
SEMANTIC_MEMORY_CATEGORIES = ("insight_v2", "pattern_v2")
EPISODIC_MEMORY_CATEGORY = "chat_interaction"


class BrainService:
    """Thin orchestration over Pinecone for 3 memory types.

    Three memory types live in one Pinecone index — separated by category /
    entry_type filters. No new namespaces, no duplication.
    """

    def __init__(self, kb):
        self.kb = kb

    # ─── Recall ─────────────────────────────────────────────────────────

    async def recall_episodic(
        self,
        user_id: str,
        query: str,
        k: int = 5,
    ) -> List[KnowledgeEntry]:
        """Past conversation turns (user side only)."""
        return await self._semantic_search(
            query=query, categories=[EPISODIC_MEMORY_CATEGORY], k=k
        )

    async def recall_semantic(
        self,
        user_id: str,
        query: str,
        k: int = 8,
    ) -> List[KnowledgeEntry]:
        """Computed behavioral insights and patterns."""
        return await self._semantic_search(
            query=query,
            categories=list(SEMANTIC_MEMORY_CATEGORIES),
            entry_types=[KnowledgeEntryType.INSIGHT, KnowledgeEntryType.PATTERN],
            k=k,
        )

    async def recall_working(
        self,
        user_id: str,
        query: str,
        k: int = 10,
        recency_window_days: int = 30,
    ) -> List[KnowledgeEntry]:
        """Recent classified behavioral data across categories.

        Filters by recency_window_days post-query so vectors that pre-date the
        window are excluded even if semantically similar.
        """
        results = await self._semantic_search(
            query=query, categories=list(WORKING_MEMORY_CATEGORIES), k=k * 2
        )
        cutoff = datetime.now(timezone.utc) - timedelta(days=recency_window_days)
        filtered: List[KnowledgeEntry] = []
        for entry in results:
            ts = self._extract_entry_timestamp(entry)
            if ts is None or ts >= cutoff:
                filtered.append(entry)
            if len(filtered) >= k:
                break
        return filtered

    async def recall_all(
        self,
        user_id: str,
        query: str,
    ) -> Dict[str, List[KnowledgeEntry]]:
        """Parallel fan-out across all three memory types via asyncio.gather."""
        episodic, semantic, working = await asyncio.gather(
            self.recall_episodic(user_id, query),
            self.recall_semantic(user_id, query),
            self.recall_working(user_id, query),
            return_exceptions=False,
        )
        return {
            "episodic": episodic,
            "semantic": semantic,
            "working": working,
        }

    # ─── Storage ────────────────────────────────────────────────────────

    async def store_episodic(
        self,
        user_id: str,
        user_msg: str,
        agent_response: str,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Optional[KnowledgeEntry]:
        """Persist a conversation turn — embed user side only, response in metadata."""
        ctx = ctx or {}
        try:
            content = self.kb._build_chat_user_embedding(user_msg)
            metadata = {
                "agent_response": agent_response[:1200],
                "session_id": ctx.get("session_id"),
                "agent_type": ctx.get("agent_type"),
                "detected_intent": ctx.get("detected_intent"),
                "detected_topic": ctx.get("detected_topic"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": {
                    "sync_event_key": "chat:" + (ctx.get("turn_id") or self.kb.deterministic_entry_id(
                        user_id, "chat_turn", user_msg[:200], datetime.now(timezone.utc).isoformat()
                    )),
                },
            }
            return await self.kb.create_entry(
                entry_type=KnowledgeEntryType.INTERACTION,
                entry_sub_type=KnowledgeEntrySubType.PERSONAL_INTERACTION,
                category=EPISODIC_MEMORY_CATEGORY,
                title=user_msg[:140],
                content=content,
                metadata=metadata,
                tags=["chat", "episodic"],
            )
        except Exception as exc:
            logger.warning("BrainService.store_episodic failed: %s", exc)
            return None

    async def store_semantic(
        self,
        user_id: str,
        insight_text: str,
        metadata: Dict[str, Any],
    ) -> Optional[KnowledgeEntry]:
        """Persist an INSIGHT — usually called by IntelligencePrecomputeService."""
        try:
            sync_key = metadata.get("deterministic_id") or metadata.get("sync_event_key")
            if not sync_key:
                sync_key = self.kb.deterministic_entry_id(user_id, "insight", insight_text[:200])
            meta = {
                **metadata,
                "context": {"sync_event_key": f"insight_v2:{sync_key}"},
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            return await self.kb.create_entry(
                entry_type=KnowledgeEntryType.INSIGHT,
                entry_sub_type=KnowledgeEntrySubType.IMPORTANT_INSIGHT,
                category="insight_v2",
                title=insight_text[:140],
                content=insight_text,
                metadata=meta,
                tags=["semantic", "intelligence"],
            )
        except Exception as exc:
            logger.warning("BrainService.store_semantic failed: %s", exc)
            return None

    # ─── Internals ──────────────────────────────────────────────────────

    async def _semantic_search(
        self,
        *,
        query: str,
        categories: Optional[List[str]] = None,
        entry_types: Optional[List[KnowledgeEntryType]] = None,
        k: int = 8,
    ) -> List[KnowledgeEntry]:
        try:
            kq = KnowledgeQuery(
                query_text=query or "",
                categories=categories,
                entry_types=entry_types,
                limit=max(int(k), 1),
                similarity_threshold=0.0,
            )
            results = await self.kb.search(kq)
            return [r.entry for r in (results or []) if getattr(r, "entry", None)]
        except Exception as exc:
            logger.warning("BrainService._semantic_search failed: %s", exc)
            return []

    @staticmethod
    def _extract_entry_timestamp(entry: KnowledgeEntry) -> Optional[datetime]:
        meta = getattr(entry, "metadata", {}) or {}
        candidate = (
            meta.get("start_time")
            or meta.get("captured_at")
            or meta.get("checkup_date")
            or meta.get("timestamp")
            or meta.get("generated_at")
            or getattr(entry, "created_at", None)
        )
        if candidate is None:
            return None
        try:
            if isinstance(candidate, datetime):
                return candidate if candidate.tzinfo else candidate.replace(tzinfo=timezone.utc)
            return datetime.fromisoformat(str(candidate).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
