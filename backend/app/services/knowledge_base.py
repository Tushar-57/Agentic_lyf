"""
Knowledge base service providing CRUD operations and RAG functionality.
"""
import os
import hashlib
import json
import uuid
import logging
import re
from time import monotonic
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..models.knowledge import (
    KnowledgeEntry,
    KnowledgeEntrySubType, 
    KnowledgeEntryType, 
    KnowledgeQuery, 
    KnowledgeSearchResult,
    KnowledgeStats,
    UserPreferences
)
from ..llm.service import get_llm_service
from ..llm.base import EmbeddingRequest
from ..utils.logging import get_embedding_category_logger
from .vector_store import get_vector_store

logger = logging.getLogger(__name__)
embedding_logger = get_embedding_category_logger("app.embedding.knowledge")


EMBEDDING_CACHE_KEY_FIELD = "_embedding_cache_key"
EMBEDDING_PROVIDER_COOLDOWN_SECONDS = 30.0


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_positive_int_env(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return max(minimum, int(raw_value))
    except (TypeError, ValueError):
        return default


EMBEDDING_MAX_CHARS_PER_CHUNK = _parse_positive_int_env(
    "EMBEDDING_MAX_CHARS_PER_CHUNK",
    default=1200,
    minimum=400,
)
EMBEDDING_CHUNK_OVERLAP_CHARS = min(
    EMBEDDING_MAX_CHARS_PER_CHUNK // 2,
    _parse_positive_int_env("EMBEDDING_CHUNK_OVERLAP_CHARS", default=180, minimum=0),
)
EMBEDDING_MAX_CHUNKS_PER_ENTRY = _parse_positive_int_env(
    "EMBEDDING_MAX_CHUNKS_PER_ENTRY",
    default=8,
    minimum=1,
)
EMBEDDING_LOG_ENABLED = _parse_bool_env("EMBEDDING_LOG_ENABLED", default=True)
EMBEDDING_LOG_FULL_TEXT = _parse_bool_env("EMBEDDING_LOG_FULL_TEXT", default=True)
EMBEDDING_LOG_MAX_TEXT_CHARS = _parse_positive_int_env(
    "EMBEDDING_LOG_MAX_TEXT_CHARS",
    default=32000,
    minimum=200,
)


class KnowledgeBaseService:
    """Service for managing knowledge base operations and RAG functionality."""
    
    def __init__(self, user_id: str = "single_user"):
        self.user_id = user_id
        self.vector_store = get_vector_store(user_id)
        self._user_preferences: Optional[UserPreferences] = None
        self._sync_event_index: Dict[str, str] = {}
        self._sync_event_index_loaded = False
        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedding_cache_loaded = False
        self._embedding_provider_cooldown_until = 0.0

    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """
        Build a deterministic lexical embedding when provider embeddings are unavailable.

        This keeps retrieval and visualization usable instead of collapsing all vectors to zeros.
        """
        dimension = self.vector_store.dimension
        vector = [0.0] * dimension

        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if not tokens:
            tokens = ["empty"]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, len(digest), 4):
                chunk = digest[offset:offset + 4]
                if len(chunk) < 4:
                    continue
                value = int.from_bytes(chunk, byteorder="big", signed=False)
                idx = value % dimension
                sign = 1.0 if (value & 1) == 0 else -1.0
                vector[idx] += sign

        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector

        return [value / norm for value in vector]

    def _truncate_for_log(self, value: Any, limit: int = 140) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[:limit - 3]}..."

    def _prepare_embedding_log_text(self, text: str) -> str:
        normalized = str(text or "")
        if EMBEDDING_LOG_FULL_TEXT:
            return normalized

        return self._truncate_for_log(normalized, limit=EMBEDDING_LOG_MAX_TEXT_CHARS)

    def _log_embedding_payload(
        self,
        *,
        action: str,
        embedding_key: str,
        embedding_text: str,
        chunks: List[str],
        entry_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> None:
        if not EMBEDDING_LOG_ENABLED:
            return

        prepared_text = self._prepare_embedding_log_text(embedding_text)
        embedding_logger.info(
            "EMBEDDING_INPUT action=%s user=%s entry_id=%s category=%s key=%s chars=%d chunks=%d text=%s",
            action,
            self.user_id,
            entry_id or "pending",
            category or "unknown",
            embedding_key,
            len(embedding_text),
            len(chunks),
            prepared_text,
        )

        for index, chunk in enumerate(chunks, start=1):
            prepared_chunk = self._prepare_embedding_log_text(chunk)
            chunk_key = self._build_chunk_embedding_cache_key(chunk)
            embedding_logger.info(
                "EMBEDDING_CHUNK action=%s user=%s entry_id=%s chunk=%d/%d key=%s chars=%d text=%s",
                action,
                self.user_id,
                entry_id or "pending",
                index,
                len(chunks),
                chunk_key,
                len(chunk),
                prepared_chunk,
            )

    def _log_query_embedding_input(self, query_text: str, query_key: str) -> None:
        if not EMBEDDING_LOG_ENABLED:
            return

        prepared_text = self._prepare_embedding_log_text(query_text)
        embedding_logger.info(
            "EMBEDDING_QUERY_INPUT user=%s key=%s chars=%d text=%s",
            self.user_id,
            query_key,
            len(query_text),
            prepared_text,
        )
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured LLM provider."""
        try:
            if monotonic() < self._embedding_provider_cooldown_until:
                return self._generate_fallback_embedding(text)

            # Check if LLM service is already initialized before trying to get it
            from ..llm import service as llm_service_module
            if not llm_service_module._llm_service or not llm_service_module._llm_service._initialized:
                logger.warning("LLM service not initialized, using deterministic fallback embedding")
                return self._generate_fallback_embedding(text)
            
            llm_service = llm_service_module._llm_service
            
            request = EmbeddingRequest(text=text)
            response = await llm_service.generate_embedding(request)
            self._embedding_provider_cooldown_until = 0.0
            return response.embedding
        except ImportError as e:
            logger.warning(f"Missing dependencies for embedding generation: {e}")
            return self._generate_fallback_embedding(text)
        except Exception as e:
            error_text = str(e).lower()
            if (
                "no healthy providers available" in error_text
                or "provider_unavailable" in error_text
                or "connection error" in error_text
                or "timed out" in error_text
                or "timeout" in error_text
                or "api connection" in error_text
            ):
                self._embedding_provider_cooldown_until = monotonic() + EMBEDDING_PROVIDER_COOLDOWN_SECONDS
            logger.warning(f"Embedding generation failed: {e}")
            return self._generate_fallback_embedding(text)

    def _normalize_embedding_label(self, value: Any) -> str:
        if hasattr(value, "value"):
            value = value.value

        return " ".join(str(value or "").split()).strip().lower()

    def _stringify_embedding_value(
        self,
        value: Any,
        max_items: int = 8,
        max_chars: int = 220,
    ) -> str:
        if value is None:
            return ""

        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, (int, float)):
            return str(value)

        if isinstance(value, str):
            normalized = " ".join(value.split()).strip()
            if len(normalized) <= max_chars:
                return normalized
            return f"{normalized[:max_chars - 3]}..."

        if isinstance(value, list):
            normalized_items = []
            for item in value[:max_items]:
                item_text = self._stringify_embedding_value(item, max_items=max_items, max_chars=max_chars)
                if item_text:
                    normalized_items.append(item_text)
            return " | ".join(normalized_items)

        if isinstance(value, dict):
            normalized_pairs = []
            for key in sorted(value.keys())[:max_items]:
                item_text = self._stringify_embedding_value(value.get(key), max_items=max_items, max_chars=max_chars)
                if item_text:
                    normalized_pairs.append(f"{key}={item_text}")
            return " | ".join(normalized_pairs)

        return self._stringify_embedding_value(str(value), max_items=max_items, max_chars=max_chars)

    def _extract_embedding_metadata_signals(
        self,
        category: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[str]:
        metadata_payload = metadata if isinstance(metadata, dict) else {}
        normalized_category = self._normalize_embedding_label(category)
        context_payload = metadata_payload.get("context") if isinstance(metadata_payload.get("context"), dict) else {}

        signals: Dict[str, str] = {}

        def add_signal(label: str, value: Any) -> None:
            if label in signals:
                return

            text = self._stringify_embedding_value(value)
            if not text:
                return

            signals[label] = text

        # Always include high-value semantic fields when available.
        add_signal("role", metadata_payload.get("role"))
        add_signal("preferences", metadata_payload.get("preferences"))
        add_signal("priority", metadata_payload.get("priority"))
        add_signal("milestones", metadata_payload.get("milestones"))
        add_signal("mentor", metadata_payload.get("mentor"))
        add_signal("coach_preferences", metadata_payload.get("coach_preferences"))
        add_signal("domain_preferences", metadata_payload.get("domain_preferences"))
        add_signal("preference_profile", metadata_payload.get("preference_profile"))
        add_signal("availability", metadata_payload.get("availability"))
        add_signal("notifications", metadata_payload.get("notifications"))
        add_signal("integrations", metadata_payload.get("integrations"))
        add_signal("agent_type", metadata_payload.get("agent_type"))

        add_signal("source", context_payload.get("source"))
        add_signal("source_action", context_payload.get("source_action"))
        add_signal("description", context_payload.get("description"))
        add_signal("task_name", context_payload.get("task_name"))
        add_signal("project_name", context_payload.get("project_name"))
        add_signal("duration_minutes", context_payload.get("duration_minutes"))
        add_signal("billable", context_payload.get("billable"))
        add_signal("linked_goal", context_payload.get("linked_goal"))
        add_signal("focus_score", context_payload.get("focus_score"))
        add_signal("energy_score", context_payload.get("energy_score"))
        add_signal("blockers", context_payload.get("blockers"))
        add_signal("context_notes", context_payload.get("context_notes"))
        add_signal("ai_detail", context_payload.get("ai_detail"))
        add_signal("habits", context_payload.get("habits"))
        add_signal("summary", context_payload.get("summary"))
        add_signal("daily_completion_counts", context_payload.get("daily_completion_counts"))

        ignored_metadata_keys = {
            EMBEDDING_CACHE_KEY_FIELD,
            "timestamp",
            "created",
            "created_at",
            "updated_at",
            "last_updated",
            "approved_at",
            "sync_event_key",
            "user_id",
            "user_email",
            "context",
        }

        ignored_context_keys = {
            "sync_event_key",
            "time_entry_id",
            "project_id",
            "tag_ids",
            "user_id",
            "user_email",
            "start_time",
            "end_time",
            "position_top",
            "position_left",
            "weekday",
            "hour_of_day",
        }

        for key in sorted(metadata_payload.keys()):
            if key in ignored_metadata_keys:
                continue
            add_signal(f"meta_{key}", metadata_payload.get(key))

        for key in sorted(context_payload.keys()):
            if key in ignored_context_keys:
                continue
            add_signal(f"context_{key}", context_payload.get(key))

        if normalized_category == "time_entry" and "duration_minutes" not in signals:
            add_signal("duration_minutes", context_payload.get("duration_seconds"))

        return [f"{label}: {value}" for label, value in sorted(signals.items())]

    def _build_embedding_text(
        self,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entry_type: Optional[Any] = None,
        entry_sub_type: Optional[Any] = None,
    ) -> str:
        normalized_title = " ".join(str(title or "").split()).strip()
        normalized_content = str(content or "").strip()
        normalized_tags = [" ".join(str(tag).split()).strip() for tag in (tags or []) if str(tag).strip()]
        normalized_category = self._normalize_embedding_label(category or "uncategorized")
        normalized_entry_type = self._normalize_embedding_label(entry_type or "unknown")
        normalized_entry_sub_type = self._normalize_embedding_label(entry_sub_type or "unknown")

        parts = [
            f"entry_type: {normalized_entry_type}",
            f"entry_sub_type: {normalized_entry_sub_type}",
            f"category: {normalized_category}",
            f"title: {normalized_title}",
        ]

        if normalized_tags:
            parts.append(f"tags: {', '.join(normalized_tags)}")

        if normalized_content:
            parts.append(f"content:\n{normalized_content}")

        metadata_signals = self._extract_embedding_metadata_signals(normalized_category, metadata)
        if metadata_signals:
            parts.append("metadata_signals:\n" + "\n".join(f"- {signal}" for signal in metadata_signals))

        return "\n\n".join(part for part in parts if part).strip()

    def _build_embedding_cache_key(self, embedding_text: str) -> str:
        normalized_text = " ".join(str(embedding_text or "").split()).strip().lower() or "empty"
        return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

    def _build_chunk_embedding_cache_key(self, chunk_text: str) -> str:
        chunk_hash = self._build_embedding_cache_key(chunk_text)
        return f"chunk::{chunk_hash}"

    def _chunk_embedding_text(self, embedding_text: str) -> List[str]:
        normalized_text = " ".join(str(embedding_text or "").split()).strip()
        if not normalized_text:
            return ["empty"]

        if len(normalized_text) <= EMBEDDING_MAX_CHARS_PER_CHUNK:
            return [normalized_text]

        chunks: List[str] = []
        cursor = 0

        while cursor < len(normalized_text) and len(chunks) < EMBEDDING_MAX_CHUNKS_PER_ENTRY:
            max_end = min(len(normalized_text), cursor + EMBEDDING_MAX_CHARS_PER_CHUNK)
            split_end = max_end

            if max_end < len(normalized_text):
                preferred_break = normalized_text.rfind(". ", cursor + EMBEDDING_MAX_CHARS_PER_CHUNK // 2, max_end)
                if preferred_break == -1:
                    preferred_break = normalized_text.rfind(" ", cursor + EMBEDDING_MAX_CHARS_PER_CHUNK // 2, max_end)
                if preferred_break > cursor:
                    split_end = preferred_break + 1

            chunk = normalized_text[cursor:split_end].strip()
            if chunk:
                chunks.append(chunk)

            if split_end >= len(normalized_text):
                break

            next_cursor = max(cursor + 1, split_end - EMBEDDING_CHUNK_OVERLAP_CHARS)
            if next_cursor <= cursor:
                next_cursor = split_end
            cursor = next_cursor

        if cursor < len(normalized_text) and len(chunks) >= EMBEDDING_MAX_CHUNKS_PER_ENTRY:
            tail = normalized_text[cursor:].strip()
            if tail:
                chunks[-1] = f"{chunks[-1]} {tail}".strip()

        return chunks or [normalized_text]

    def _fit_embedding_dimension(self, embedding: List[float]) -> List[float]:
        target_dimension = self.vector_store.dimension
        if len(embedding) == target_dimension:
            return list(embedding)

        if len(embedding) > target_dimension:
            return list(embedding[:target_dimension])

        return list(embedding) + [0.0] * (target_dimension - len(embedding))

    async def _resolve_query_embedding(self, query_text: str) -> List[float]:
        query_key = f"query::{self._build_embedding_cache_key(query_text)}"
        self._log_query_embedding_input(query_text, query_key)

        cached_embedding = self._embedding_cache.get(query_key)
        if cached_embedding:
            if EMBEDDING_LOG_ENABLED:
                embedding_logger.info(
                    "EMBEDDING_QUERY_CACHE_HIT user=%s key=%s",
                    self.user_id,
                    query_key,
                )
            return list(cached_embedding)

        generated_embedding = await self._generate_embedding(query_text)
        self._cache_embedding_value(query_key, generated_embedding)
        if EMBEDDING_LOG_ENABLED:
            embedding_logger.info(
                "EMBEDDING_QUERY_GENERATED user=%s key=%s dimension=%d",
                self.user_id,
                query_key,
                len(generated_embedding),
            )
        return generated_embedding

    async def _generate_embedding_for_text(self, embedding_text: str) -> List[float]:
        chunks = self._chunk_embedding_text(embedding_text)
        if len(chunks) == 1:
            return await self._generate_embedding(chunks[0])

        weighted_embedding = [0.0] * self.vector_store.dimension
        total_weight = 0.0

        for chunk in chunks:
            chunk_key = self._build_chunk_embedding_cache_key(chunk)
            chunk_embedding = self._embedding_cache.get(chunk_key)
            if chunk_embedding:
                resolved_chunk_embedding = list(chunk_embedding)
            else:
                resolved_chunk_embedding = await self._generate_embedding(chunk)
                self._cache_embedding_value(chunk_key, resolved_chunk_embedding)

            normalized_chunk_embedding = self._fit_embedding_dimension(resolved_chunk_embedding)
            chunk_weight = float(max(1, len(chunk)))
            total_weight += chunk_weight

            for idx, value in enumerate(normalized_chunk_embedding):
                weighted_embedding[idx] += value * chunk_weight

        if total_weight <= 0:
            return await self._generate_embedding(embedding_text)

        averaged_embedding = [value / total_weight for value in weighted_embedding]
        norm = sum(value * value for value in averaged_embedding) ** 0.5
        if norm > 0:
            averaged_embedding = [value / norm for value in averaged_embedding]

        return averaged_embedding

    def _extract_embedding_cache_key(self, entry: KnowledgeEntry) -> str:
        metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
        cached_key = str(metadata.get(EMBEDDING_CACHE_KEY_FIELD, "")).strip()
        if cached_key:
            return cached_key

        fallback_text = self._build_embedding_text(
            title=entry.title,
            content=entry.content,
            tags=entry.tags,
            category=entry.category,
            metadata=entry.metadata,
            entry_type=entry.entry_type,
            entry_sub_type=entry.entry_sub_type,
        )
        return self._build_embedding_cache_key(fallback_text)

    def _cache_embedding_value(self, embedding_key: str, embedding: Optional[List[float]]) -> None:
        if not embedding_key or not embedding:
            return
        self._embedding_cache[embedding_key] = list(embedding)

    def _cache_entry_embedding(self, entry: Optional[KnowledgeEntry]) -> None:
        if not entry or not entry.embedding:
            return
        self._cache_embedding_value(self._extract_embedding_cache_key(entry), entry.embedding)

    async def _ensure_embedding_cache_loaded(self) -> None:
        if self._embedding_cache_loaded:
            return

        try:
            existing_entries = await self.get_all_entries()
            for entry in existing_entries:
                self._cache_entry_embedding(entry)
        finally:
            self._embedding_cache_loaded = True

    def _preserve_embeddings_for_entry_ids(self, entry_ids: List[str]) -> None:
        for entry_id in entry_ids:
            entry = self.vector_store.get_entry(entry_id)
            self._cache_entry_embedding(entry)

    async def _resolve_embedding(
        self,
        embedding_text: str,
        embedding_key: str,
        existing_entry: Optional[KnowledgeEntry] = None,
    ) -> List[float]:
        await self._ensure_embedding_cache_loaded()

        if existing_entry and existing_entry.embedding:
            existing_key = self._extract_embedding_cache_key(existing_entry)
            if existing_key == embedding_key:
                resolved = list(existing_entry.embedding)
                self._cache_embedding_value(embedding_key, resolved)
                return resolved

        cached_embedding = self._embedding_cache.get(embedding_key)
        if cached_embedding:
            return list(cached_embedding)

        generated_embedding = await self._generate_embedding_for_text(embedding_text)
        self._cache_embedding_value(embedding_key, generated_embedding)
        return generated_embedding

    async def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant knowledge entries based on a query."""
        try:
            # Generate embedding for the query
            query_embedding = await self._resolve_query_embedding(query)
            
            # Search the vector store
            results = self.vector_store.search(query_embedding, limit)
            
            # Format results
            formatted_results = []
            for result in results:
                entry = result.entry if hasattr(result, "entry") else None
                formatted_results.append({
                    "content": getattr(entry, "content", ""),
                    "score": float(getattr(result, "similarity_score", 0.0)),
                    "metadata": getattr(entry, "metadata", {}) if entry else {},
                })
            
            return formatted_results
            
        except Exception as e:
            logger.warning(f"Knowledge search failed: {e}")
            # Return empty results for graceful degradation
            return []
    
    async def create_entry(self, 
                          entry_type: KnowledgeEntryType,
                          entry_sub_type:KnowledgeEntrySubType,
                          category: str,
                          title: str,
                          content: str,
                          metadata: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None) -> KnowledgeEntry:
        """
        Create a new knowledge base entry.
        
        Args:
            entry_type: Type of the entry
            category: Category of the entry
            title: Human-readable title
            content: The actual content
            metadata: Additional metadata
            tags: Tags for categorization
            
        Returns:
            The created knowledge entry
        """
        try:
            metadata_payload = dict(metadata or {})

            embedding_text = self._build_embedding_text(
                title=title,
                content=content,
                tags=tags,
                category=category,
                metadata=metadata_payload,
                entry_type=entry_type,
                entry_sub_type=entry_sub_type,
            )
            embedding_key = self._build_embedding_cache_key(embedding_text)
            metadata_payload[EMBEDDING_CACHE_KEY_FIELD] = embedding_key

            entry_chunks = self._chunk_embedding_text(embedding_text)
            self._log_embedding_payload(
                action="create",
                embedding_key=embedding_key,
                embedding_text=embedding_text,
                chunks=entry_chunks,
                category=category,
            )

            embedding = await self._resolve_embedding(embedding_text, embedding_key)

            # Generate unique ID
            entry_id = str(uuid.uuid4())
            
            # Create entry
            entry = KnowledgeEntry(
                entry_id=entry_id,
                user_id=self.user_id,
                entry_type=entry_type,
                category=category,
                entry_sub_type=entry_sub_type,
                title=title,
                content=content,
                metadata=metadata_payload,
                tags=tags or []
            )
            
            # Add to vector store
            self.vector_store.add_entry(entry, embedding)
            self._index_sync_event_key(entry)
            self._cache_entry_embedding(entry)
            
            logger.info(f"Created knowledge entry: {entry_id}")
            return entry
        except Exception as e:
            logger.error(f"Failed to create knowledge entry: {e}")
            raise
    
    async def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """
        Retrieve a knowledge entry by ID.
        
        Args:
            entry_id: ID of the entry to retrieve
            
        Returns:
            The knowledge entry if found, None otherwise
        """
        try:
            return self.vector_store.get_entry(entry_id)
        except Exception as e:
            logger.error(f"Failed to get knowledge entry {entry_id}: {e}")
            return None
    
    async def update_entry(self, 
                          entry_id: str,
                          title: Optional[str] = None,
                          content: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None) -> Optional[KnowledgeEntry]:
        """
        Update an existing knowledge entry.
        
        Args:
            entry_id: ID of the entry to update
            title: New title (optional)
            content: New content (optional)
            metadata: New metadata (optional)
            tags: New tags (optional)
            
        Returns:
            The updated entry if successful, None otherwise
        """
        try:
            # Get existing entry
            existing_entry = self.vector_store.get_entry(entry_id)
            if not existing_entry:
                logger.warning(f"Entry {entry_id} not found for update")
                return None
            
            # Update fields
            updated_entry = existing_entry.model_copy()
            if not isinstance(updated_entry.metadata, dict):
                updated_entry.metadata = {}

            if title is not None:
                updated_entry.title = title
            if content is not None:
                updated_entry.content = content
            if metadata is not None:
                updated_entry.metadata.update(metadata)
            if tags is not None:
                updated_entry.tags = tags

            embedding_text = self._build_embedding_text(
                title=updated_entry.title,
                content=updated_entry.content,
                tags=updated_entry.tags,
                category=updated_entry.category,
                metadata=updated_entry.metadata,
                entry_type=updated_entry.entry_type,
                entry_sub_type=updated_entry.entry_sub_type,
            )
            embedding_key = self._build_embedding_cache_key(embedding_text)
            updated_entry.metadata[EMBEDDING_CACHE_KEY_FIELD] = embedding_key

            entry_chunks = self._chunk_embedding_text(embedding_text)
            self._log_embedding_payload(
                action="update",
                embedding_key=embedding_key,
                embedding_text=embedding_text,
                chunks=entry_chunks,
                entry_id=updated_entry.entry_id,
                category=updated_entry.category,
            )
            
            updated_entry.updated_at = datetime.utcnow()
            
            embedding = await self._resolve_embedding(
                embedding_text,
                embedding_key,
                existing_entry=existing_entry,
            )
            
            # Update in vector store
            self.vector_store.update_entry(updated_entry, embedding)
            self._index_sync_event_key(updated_entry)
            self._cache_entry_embedding(updated_entry)
            
            logger.info(f"Updated knowledge entry: {entry_id}")
            return updated_entry
        except Exception as e:
            logger.error(f"Failed to update knowledge entry {entry_id}: {e}")
            return None
    
    async def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a knowledge entry.
        
        Args:
            entry_id: ID of the entry to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            self._preserve_embeddings_for_entry_ids([entry_id])
            self._remove_indexed_sync_event_for_entry(entry_id)
            success = self.vector_store.remove_entry(entry_id)
            if success:
                logger.info(f"Deleted knowledge entry: {entry_id}")
            else:
                logger.warning(f"Entry {entry_id} not found for deletion")
            return success
        except Exception as e:
            logger.error(f"Failed to delete knowledge entry {entry_id}: {e}")
            return False

    async def delete_entries(self, entry_ids: List[str]) -> int:
        """Delete multiple knowledge entries in a single vector index rebuild pass."""
        try:
            normalized_ids = [entry_id for entry_id in entry_ids if entry_id]
            if not normalized_ids:
                return 0

            self._preserve_embeddings_for_entry_ids(normalized_ids)

            for entry_id in normalized_ids:
                self._remove_indexed_sync_event_for_entry(entry_id)

            removed = self.vector_store.remove_entries(normalized_ids)
            if removed > 0:
                logger.info("Deleted %d knowledge entries in bulk", removed)
            return removed
        except Exception as e:
            logger.error(f"Failed to bulk delete knowledge entries: {e}")
            return 0
    
    async def search(self, query: KnowledgeQuery) -> List[KnowledgeSearchResult]:
        """
        Search the knowledge base using RAG.
        
        Args:
            query: The search query
            
        Returns:
            List of search results
        """
        try:
            # Generate embedding for query
            query_embedding = await self._resolve_query_embedding(query.query_text)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                k=query.limit,
                similarity_threshold=query.similarity_threshold
            )
            
            # Filter by categories, types, and tags if specified
            filtered_results = []
            for result in results:
                entry = result.entry
                
                # Filter by categories
                if query.categories and entry.category not in query.categories:
                    continue
                
                # Filter by entry types
                if query.entry_types and entry.entry_type not in query.entry_types:
                    continue
                
                # Filter by tags
                if query.tags and not any(tag in entry.tags for tag in query.tags):
                    continue
                
                filtered_results.append(result)
            
            logger.debug(f"Knowledge search returned {len(filtered_results)} results")
            return filtered_results
        except ImportError as e:
            logger.warning(f"Knowledge search failed due to missing dependencies: {e}")
            return []
        except Exception as e:
            logger.warning(f"Failed to search knowledge base: {e}")
            return []
    
    async def get_all_entries(self, 
                             category: Optional[str] = None,
                             entry_type: Optional[KnowledgeEntryType] = None) -> List[KnowledgeEntry]:
        """
        Get all knowledge entries, optionally filtered by category or type.
        
        Args:
            category: Filter by category (optional)
            entry_type: Filter by entry type (optional)
            
        Returns:
            List of knowledge entries
        """
        try:
            all_entries = self.vector_store.get_all_entries()
            
            # Apply filters
            filtered_entries = []
            for entry in all_entries:
                if getattr(entry, "user_id", self.user_id) != self.user_id:
                    continue
                if category and entry.category != category:
                    continue
                if entry_type and entry.entry_type != entry_type:
                    continue
                filtered_entries.append(entry)
            
            return filtered_entries
        except Exception as e:
            logger.error(f"Failed to get all entries: {e}")
            return []
    
    async def get_user_preferences(self) -> UserPreferences:
        """
        Get user preferences, loading from knowledge base or creating defaults.
        
        Returns:
            User preferences object
        """
        try:
            if self._user_preferences is not None:
                return self._user_preferences

            # Try to hydrate preferences from persisted knowledge entries first.
            try:
                all_entries = await self.get_all_entries()
                prefs_dict = UserPreferences().model_dump()
                loaded_from_knowledge = False

                system_pref_entries = [
                    entry
                    for entry in all_entries
                    if entry.entry_type == KnowledgeEntryType.PREFERENCE
                    and str(entry.category).strip().lower() == "system"
                    and str(entry.title).strip().lower() == "user preferences"
                ]

                if system_pref_entries:
                    latest_system_entry = max(system_pref_entries, key=lambda entry: entry.updated_at)
                    try:
                        snapshot = json.loads(latest_system_entry.content) if latest_system_entry.content else {}
                        if isinstance(snapshot, dict):
                            for section, value in snapshot.items():
                                if (
                                    section in prefs_dict
                                    and isinstance(prefs_dict[section], dict)
                                    and isinstance(value, dict)
                                ):
                                    prefs_dict[section].update(value)
                                else:
                                    prefs_dict[section] = value
                            loaded_from_knowledge = True
                    except Exception as snapshot_error:
                        logger.warning(f"Failed to parse system preference snapshot: {snapshot_error}")

                user_entries = [
                    entry
                    for entry in all_entries
                    if entry.entry_type == KnowledgeEntryType.USER_PREFERENCE
                ]

                if user_entries:
                    loaded_from_knowledge = True
                    for entry in user_entries:
                        if entry.entry_sub_type == KnowledgeEntrySubType.USER_PROFILE:
                            metadata = entry.metadata or {}
                            prefs_dict["general"]["role"] = metadata.get("role", "professional")
                            prefs_dict["general"]["priorities"] = metadata.get("preferences", [])
                            prefs_dict["general"]["mentor"] = metadata.get("mentor", {})
                            prefs_dict["general"]["onboarding_completed"] = metadata.get("onboarding_completed", True)

                        elif entry.entry_sub_type == KnowledgeEntrySubType.GOAL:
                            metadata = entry.metadata or {}
                            category = metadata.get("category", "general").lower()
                            section_map = {
                                "career": "productivity",
                                "work": "productivity",
                                "productivity": "productivity",
                                "health": "health",
                                "wellness": "health",
                                "finance": "finance",
                                "financial": "finance",
                                "money": "finance",
                                "journal": "journal",
                                "reflection": "journal",
                            }
                            target_section = section_map.get(category, "productivity")
                            if "goals" not in prefs_dict[target_section]:
                                prefs_dict[target_section]["goals"] = []
                            prefs_dict[target_section]["goals"].append({
                                "title": entry.title,
                                "priority": metadata.get("priority", "Medium"),
                                "milestones": metadata.get("milestones", []),
                            })

                        elif entry.entry_sub_type == KnowledgeEntrySubType.SCHEDULE:
                            metadata = entry.metadata or {}
                            availability = metadata.get("availability", {})
                            prefs_dict["general"]["work_hours"] = (
                                f"{availability.get('workHours', {}).get('start', '09:00')}-"
                                f"{availability.get('workHours', {}).get('end', '17:00')}"
                            )
                            prefs_dict["general"]["timezone"] = availability.get("timezone", "UTC")
                            prefs_dict["journal"]["check_in_time"] = availability.get("checkIn", {}).get("preferredTime", "09:00")

                if loaded_from_knowledge:
                    logger.info(f"Loaded user preferences from knowledge base: {prefs_dict}")
                    prefs_dict["user_id"] = self.user_id
                    self._user_preferences = UserPreferences(**prefs_dict)
                    return self._user_preferences

            except Exception as e:
                logger.warning(f"Failed to load preferences from knowledge base: {e}, trying JSON file")

            # Fallback to JSON file
            import os
            prefs_path = os.path.join(os.path.dirname(__file__), "user_preferences.json")
            try:
                if os.path.exists(prefs_path):
                    with open(prefs_path) as f:
                        data = f.read().strip()
                        if not data:
                            raise ValueError("Empty preferences file")
                        prefs_dict = json.loads(data)
                        prefs_dict["user_id"] = self.user_id
                        self._user_preferences = UserPreferences(**prefs_dict)
                        return self._user_preferences
            except Exception as e:
                logger.warning(f"Failed to parse stored preferences: {e}. Using defaults.")

            self._user_preferences = UserPreferences(user_id=self.user_id)
            return self._user_preferences
        except Exception as e:
            logger.error(f"Failed to get user preferences: {e}")
            return UserPreferences(user_id=self.user_id)

    def _is_time_entry_entry(self, entry: KnowledgeEntry) -> bool:
        """Detect AlterEgo time entries that are persisted as interaction events."""
        metadata = entry.metadata or {}
        context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}

        category = str(entry.category or "").strip().lower()
        source = str(context.get("source", "")).strip().lower()
        source_action = str(context.get("source_action", "")).strip().lower()
        has_time_entry_id = context.get("time_entry_id") is not None
        has_time_entry_tag = any(str(tag).strip().lower() == "time_entry" for tag in (entry.tags or []))

        return (
            category == "time_entry"
            or source == "alterego_timetracker"
            or "time_entry" in source_action
            or has_time_entry_id
            or has_time_entry_tag
        )

    def _normalize_visual_category(self, entry: KnowledgeEntry) -> str:
        if self._is_time_entry_entry(entry):
            return "time_entry"

        category = str(entry.category or "uncategorized").strip().lower()
        return category if category else "uncategorized"

    def _normalize_visual_type(self, entry: KnowledgeEntry, normalized_category: str) -> str:
        if normalized_category == "time_entry":
            return "time_entry"

        if hasattr(entry.entry_type, "value"):
            return str(entry.entry_type.value)

        return str(entry.entry_type)

    def _infer_interaction_category_and_sub_type(
        self,
        agent_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, KnowledgeEntrySubType, List[str]]:
        """Derive interaction grouping so synced events surface in the right knowledge category."""
        context_payload = context or {}
        normalized_agent = (agent_type or "general").strip().lower() or "general"

        source = str(context_payload.get("source", "")).strip().lower()
        source_action = str(context_payload.get("source_action", "")).strip().lower()
        forced_category = str(context_payload.get("category", "")).strip().lower()
        has_time_entry_id = context_payload.get("time_entry_id") is not None
        approval_payload = context_payload.get("approval") if isinstance(context_payload.get("approval"), dict) else {}
        approved_as_insight = bool(
            context_payload.get("approved_as_insight")
            or approval_payload.get("approved_as_insight")
        )

        is_time_entry = (
            forced_category == "time_entry"
            or source == "alterego_timetracker"
            or "time_entry" in source_action
            or has_time_entry_id
        )
        if is_time_entry:
            return (
                "time_entry",
                KnowledgeEntrySubType.WORK_INTERACTION,
                ["interaction", "history", "time_entry", "alterego_sync", normalized_agent],
            )

        if forced_category in {"insight", "important_insight", "deep_insight"} or approved_as_insight:
            return (
                "insight",
                KnowledgeEntrySubType.IMPORTANT_INSIGHT,
                ["insight", "approved", normalized_agent],
            )

        if normalized_agent == "health":
            sub_type = KnowledgeEntrySubType.HEALTH_INTERACTION
        elif normalized_agent in {"productivity", "finance", "scheduling"}:
            sub_type = KnowledgeEntrySubType.WORK_INTERACTION
        elif normalized_agent in {"journal", "general", "orchestrator"}:
            sub_type = KnowledgeEntrySubType.PERSONAL_INTERACTION
        else:
            sub_type = KnowledgeEntrySubType.MISC_INTERACTION

        return normalized_agent, sub_type, ["interaction", "history", normalized_agent]

    def _build_time_entry_title(self, context_payload: Optional[Dict[str, Any]] = None) -> str:
        """Build a human-meaningful title for synced time entries."""
        payload = context_payload or {}

        project_name = str(payload.get("project_name", "")).strip()
        description = str(payload.get("description", "")).strip()
        task_name = str(payload.get("task_name", "")).strip()

        activity = task_name or description

        if project_name and activity and project_name.lower() != activity.lower():
            base_title = f"{project_name}: {activity}"
        else:
            base_title = activity or project_name

        duration_suffix = ""
        raw_duration = payload.get("duration_minutes")
        try:
            duration_minutes = int(round(float(raw_duration))) if raw_duration is not None else 0
            if duration_minutes > 0:
                duration_suffix = f" ({duration_minutes}m)"
        except (TypeError, ValueError):
            duration_suffix = ""

        if base_title:
            compact_title = base_title if len(base_title) <= 90 else f"{base_title[:87]}..."
            return f"Time Entry - {compact_title}{duration_suffix}"

        return f"Time Entry{duration_suffix}" if duration_suffix else "Time Entry"

    def _build_interaction_title(
        self,
        category: str,
        context_payload: Optional[Dict[str, Any]],
    ) -> str:
        if category == "time_entry":
            return self._build_time_entry_title(context_payload)

        if category == "insight":
            payload = context_payload or {}
            agent_hint = str(payload.get("agent_type") or payload.get("source_action") or "").strip()
            if agent_hint:
                return f"Approved Insight - {agent_hint.replace('_', ' ').title()}"
            return "Approved Insight"

        title_source = category.replace("_", " ").title()
        return f"Interaction with {title_source}"

    def _extract_sync_event_key(self, entry: KnowledgeEntry) -> str:
        metadata = entry.metadata or {}
        context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
        return str(context.get("sync_event_key", "")).strip()

    def _index_sync_event_key(self, entry: KnowledgeEntry) -> None:
        if entry.entry_type != KnowledgeEntryType.INTERACTION:
            return

        sync_event_key = self._extract_sync_event_key(entry)
        if sync_event_key:
            self._sync_event_index[sync_event_key] = entry.entry_id

    def _remove_indexed_sync_event_for_entry(self, entry_id: str) -> None:
        stale_keys = [key for key, mapped_entry_id in self._sync_event_index.items() if mapped_entry_id == entry_id]
        for key in stale_keys:
            self._sync_event_index.pop(key, None)

    async def _ensure_sync_event_index_loaded(self) -> None:
        if self._sync_event_index_loaded:
            return

        try:
            interaction_entries = await self.get_all_entries(entry_type=KnowledgeEntryType.INTERACTION)
            self._sync_event_index = {}
            for entry in interaction_entries:
                self._index_sync_event_key(entry)
        finally:
            self._sync_event_index_loaded = True

    async def _find_interaction_by_sync_event_key(self, sync_event_key: str) -> Optional[KnowledgeEntry]:
        """Find an existing interaction entry by external sync key."""
        normalized_sync_key = str(sync_event_key or "").strip()
        if not normalized_sync_key:
            return None

        await self._ensure_sync_event_index_loaded()

        entry_id = self._sync_event_index.get(normalized_sync_key)
        if entry_id:
            entry = self.vector_store.get_entry(entry_id)
            if entry:
                return entry
            self._sync_event_index.pop(normalized_sync_key, None)

        # Fallback scan for robustness in case index becomes stale.
        existing_entries = await self.get_all_entries(entry_type=KnowledgeEntryType.INTERACTION)
        for entry in existing_entries:
            if self._extract_sync_event_key(entry) == normalized_sync_key:
                self._sync_event_index[normalized_sync_key] = entry.entry_id
                return entry

        return None
    
    async def update_user_preferences(self, preferences: UserPreferences) -> bool:
        """
        Update user preferences in the knowledge base.
        
        Args:
            preferences: Updated preferences
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._user_preferences = preferences
            return await self._save_user_preferences()
        except Exception as e:
            logger.error(f"Failed to update user preferences: {e}")
            return False
    
    async def add_user_preference(self, category: str, key: str, value: Any, description: Optional[str] = None) -> bool:
        """
        Add a new user preference.
        
        Args:
            category: Category of the preference (e.g., 'productivity', 'health')
            key: Key name for the preference
            value: Value of the preference
            description: Optional description of the preference
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current preferences
            current_prefs = await self.get_user_preferences()
            
            # Convert to dict for manipulation
            prefs_dict = current_prefs.model_dump()
            
            # Ensure category exists
            if category not in prefs_dict:
                prefs_dict[category] = {}
            
            # Add the new preference
            prefs_dict[category][key] = value
            
            # If description provided, store it in metadata
            if description:
                metadata_key = f"__{key}_description"
                prefs_dict[category][metadata_key] = description
            
            # Update preferences
            updated_prefs = UserPreferences(**prefs_dict)
            success = await self.update_user_preferences(updated_prefs)
            
            if success:
                logger.info(f"Added user preference: {category}.{key} = {value}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to add user preference {category}.{key}: {e}")
            return False
    
    async def remove_user_preference(self, category: str, key: str) -> bool:
        """
        Remove a user preference.
        
        Args:
            category: Category of the preference
            key: Key name for the preference
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current preferences
            current_prefs = await self.get_user_preferences()
            
            # Convert to dict for manipulation
            prefs_dict = current_prefs.model_dump()
            
            # Check if category and key exist
            if category not in prefs_dict or key not in prefs_dict[category]:
                logger.warning(f"Preference {category}.{key} not found for removal")
                return False
            
            # Remove the preference
            del prefs_dict[category][key]
            
            # Also remove description if it exists
            description_key = f"__{key}_description"
            if description_key in prefs_dict[category]:
                del prefs_dict[category][description_key]
            
            # Update preferences
            updated_prefs = UserPreferences(**prefs_dict)
            success = await self.update_user_preferences(updated_prefs)
            
            if success:
                logger.info(f"Removed user preference: {category}.{key}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to remove user preference {category}.{key}: {e}")
            return False
    
    async def get_preference_categories(self) -> List[str]:
        """
        Get all available preference categories.
        
        Returns:
            List of preference category names
        """
        try:
            current_prefs = await self.get_user_preferences()
            prefs_dict = current_prefs.model_dump()
            return list(prefs_dict.keys())
        except Exception as e:
            logger.error(f"Failed to get preference categories: {e}")
            return []
    
    async def _save_user_preferences(self) -> bool:
        """Save user preferences to knowledge base."""
        try:
            if not self._user_preferences:
                return False
            
            # Check if preferences entry already exists
            existing_entries = await self.get_all_entries(
                category="system",
                entry_type=KnowledgeEntryType.PREFERENCE
            )
            
            prefs_json = self._user_preferences.model_dump_json(indent=2)
            
            if existing_entries:
                # Update existing entry
                entry = existing_entries[0]
                await self.update_entry(
                    entry_id=entry.entry_id,
                    content=prefs_json,
                    metadata={"last_updated": datetime.utcnow().isoformat()}
                )
            else:
                # Create new entry
                await self.create_entry(
                    entry_type=KnowledgeEntryType.PREFERENCE,
                    category="system",
                    entry_sub_type=KnowledgeEntrySubType.OTHER_PREFERENCE,
                    title="User Preferences",
                    content=prefs_json,
                    metadata={"created": datetime.utcnow().isoformat()},
                    tags=["preferences", "settings", "configuration"]
                )
            
            return True
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
            return False
    
    async def add_interaction_history(self, 
                                    agent_type: str,
                                    user_input: str,
                                    agent_response: str,
                                    context: Optional[Dict[str, Any]] = None) -> KnowledgeEntry:
        """
        Add an interaction to the history for learning purposes.
        
        Args:
            agent_type: Type of agent that handled the interaction
            user_input: User's input
            agent_response: Agent's response
            context: Additional context information
        
        Returns:
            The created interaction entry
        """
        try:
            context_payload = context or {}
            category, entry_sub_type, tags = self._infer_interaction_category_and_sub_type(
                agent_type=agent_type,
                context=context_payload,
            )

            entry_type = KnowledgeEntryType.INSIGHT if category == "insight" else KnowledgeEntryType.INTERACTION

            approval_payload = context_payload.get("approval") if isinstance(context_payload.get("approval"), dict) else {}
            approved_by_user = bool(context_payload.get("approved_by_user") or approval_payload.get("approved"))
            approved_at = context_payload.get("approved_at") or approval_payload.get("approved_at")
            knowledge_sources = context_payload.get("knowledge_sources")
            if not isinstance(knowledge_sources, list):
                knowledge_sources = []

            interaction_title = self._build_interaction_title(category, context_payload)
            if category == "time_entry":
                interaction_content = f"User: {user_input}\nAgent ({agent_type}): {agent_response}"
            else:
                interaction_content = json.dumps(
                    {
                        "user_input": user_input,
                        "agent_response": agent_response,
                        "agent_type": agent_type,
                        "approved_by_user": approved_by_user,
                        "approved_at": approved_at,
                        "knowledge_sources": knowledge_sources[:8],
                    },
                    ensure_ascii=False,
                )

            metadata_payload = {
                "agent_type": agent_type,
                "timestamp": datetime.utcnow().isoformat(),
                "context": context_payload,
                "user_input_length": len(user_input),
                "response_length": len(agent_response),
                "approved_by_user": approved_by_user,
                "approved_at": approved_at,
                "knowledge_source_count": len(knowledge_sources),
            }

            sync_event_key = str(context_payload.get("sync_event_key", "")).strip()
            if sync_event_key:
                existing_entry = await self._find_interaction_by_sync_event_key(sync_event_key)
                if existing_entry:
                    updated_entry = await self.update_entry(
                        entry_id=existing_entry.entry_id,
                        title=interaction_title,
                        content=interaction_content,
                        metadata=metadata_payload,
                        tags=tags,
                    )
                    if updated_entry:
                        return updated_entry

            return await self.create_entry(
                entry_type=entry_type,
                entry_sub_type=entry_sub_type,
                category=category,
                title=interaction_title,
                content=interaction_content,
                metadata=metadata_payload,
                tags=tags
            )
        except Exception as e:
            logger.error(f"Failed to add interaction history: {e}")
            raise

    async def extract_and_store_preferences(self, 
                                          user_input: str, 
                                          agent_type: str,
                                          agent_response: str) -> List[KnowledgeEntry]:
        """
        Extract and store user preferences from conversation.
        
        Args:
            user_input: User's input
            agent_type: Type of agent handling the request
            agent_response: Agent's response
            
        Returns:
            List of created preference entries
        """
        try:
            # Use LLM to extract preferences from the conversation
            llm_service = await get_llm_service()
            if not llm_service:
                logger.warning("LLM service not available for preference extraction")
                return []
            
            # Simplified and faster preference extraction prompt
            extraction_prompt = f"""
            Extract user preferences from this conversation:

            User: {user_input}
            Agent: {agent_response}

            Find explicit preferences like:
            - Foods they like/dislike
            - Exercise habits
            - Goals mentioned
            - Schedule preferences
            - Budget constraints

            Return JSON list:
            [{{"category": "health", "key": "preference_name", "value": "preference_value"}}]

            Return [] if no clear preferences found.
            """
            
            from ..llm.base import CompletionRequest, ChatMessage
            request = CompletionRequest(
                messages=[ChatMessage(role="user", content=extraction_prompt)],
                temperature=0.1,
                max_tokens=500  # Reduced for faster response
            )
            
            response = await llm_service.chat_completion(request)
            
            # Parse the response with better error handling
            try:
                import json
                # Try to extract JSON from response
                response_payload = getattr(response, "content", response)
                if isinstance(response_payload, str):
                    response_text = response_payload.strip()
                elif isinstance(response_payload, dict):
                    for key in ("content", "response", "message", "text", "output"):
                        if key in response_payload and response_payload[key]:
                            response_text = str(response_payload[key]).strip()
                            break
                    else:
                        response_text = json.dumps(response_payload)
                elif isinstance(response_payload, (list, tuple)):
                    response_text = "\n".join(
                        str(part).strip() for part in response_payload if str(part).strip()
                    )
                else:
                    response_text = str(response_payload).strip()
                
                # Handle cases where response might have extra text
                if '[' in response_text and ']' in response_text:
                    start = response_text.find('[')
                    end = response_text.rfind(']') + 1
                    json_text = response_text[start:end]
                else:
                    json_text = response_text
                
                preferences_data = json.loads(json_text)
                created_entries = []
                
                # Handle simplified format
                for pref in preferences_data:
                    if isinstance(pref, dict) and 'category' in pref and 'key' in pref and 'value' in pref:
                        # Store preference using the existing method
                        success = await self.add_user_preference(
                            category=pref['category'],
                            key=pref['key'],
                            value=pref['value'],
                            description=f"Extracted from conversation: {pref['value']}"
                        )
                        if success:
                            # Create a knowledge entry for tracking
                            entry = await self.create_entry(
                                entry_type=KnowledgeEntryType.PREFERENCE,
                                entry_sub_type=KnowledgeEntrySubType.PERSONAL_PREFERENCE,
                                category=pref['category'],
                                title=f"{pref['key']} preference",
                                content=f"User preference: {pref['key']} = {pref['value']}",
                                metadata={
                                    "extracted_from_interaction": True,
                                    "agent_type": agent_type,
                                    "timestamp": datetime.utcnow().isoformat()
                                },
                                tags=[pref['category'], "preference", "extracted", agent_type]
                            )
                            if entry:
                                created_entries.append(entry)
                
                logger.info(f"Successfully extracted {len(created_entries)} preferences")
                return created_entries
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse preferences extraction response: {e}")
                logger.debug(f"Response content: {response.content}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to extract preferences: {e}")
            return []

    async def get_contextual_knowledge_for_agent(self, 
                                                user_input: str,
                                                agent_type: str,
                                                max_results: int = 10) -> Dict[str, Any]:
        """
        Get relevant knowledge context for an agent based on user input.
        
        Args:
            user_input: User's current input
            agent_type: Type of agent requesting context
            max_results: Maximum number of results per category
            
        Returns:
            Dictionary containing relevant context organized by type
        """
        try:
            # Get user preferences for this agent type
            preferences = await self.get_user_preferences()
            agent_preferences = getattr(preferences, agent_type.lower(), {})
            
            # Search for relevant interactions
            search_query = KnowledgeQuery(
                query_text=user_input,
                categories=[agent_type],
                entry_types=[KnowledgeEntryType.INTERACTION, KnowledgeEntryType.PREFERENCE, KnowledgeEntryType.PATTERN],
                limit=max_results,
                similarity_threshold=0.6
            )
            
            search_results = await self.search(search_query)
            
            # Search for cross-category relevant information
            general_search = KnowledgeQuery(
                query_text=user_input,
                limit=max_results,
                similarity_threshold=0.6
            )
            
            general_results = await self.search(general_search)

            # Merge agent-specific and general results while preserving ranking order.
            combined_results: List[KnowledgeSearchResult] = []
            seen_entry_ids = set()
            for result_group in (search_results, general_results):
                for result in result_group:
                    entry_id = getattr(result.entry, "entry_id", None)
                    if entry_id and entry_id in seen_entry_ids:
                        continue
                    if entry_id:
                        seen_entry_ids.add(entry_id)
                    combined_results.append(result)

            interaction_results = [
                result
                for result in combined_results
                if result.entry.entry_type == KnowledgeEntryType.INTERACTION
            ]
            preference_results = [
                result
                for result in combined_results
                if result.entry.entry_type == KnowledgeEntryType.PREFERENCE
            ]
            pattern_results = [
                result
                for result in combined_results
                if result.entry.entry_type in [KnowledgeEntryType.PATTERN, KnowledgeEntryType.INSIGHT]
            ]

            recent_time_entries = self._extract_recent_time_entries(interaction_results)
            
            # Organize results by type
            context = {
                "agent_preferences": agent_preferences,
                "relevant_interactions": [
                    {
                        "content": result.entry.content,
                        "metadata": result.entry.metadata,
                        "similarity": result.similarity_score,
                        "created_at": result.entry.created_at.isoformat(),
                        "category": self._normalize_visual_category(result.entry),
                        "is_time_entry": self._is_time_entry_entry(result.entry),
                    }
                    for result in interaction_results
                ][:6],
                "user_preferences": [
                    {
                        "content": result.entry.content,
                        "category": result.entry.category,
                        "metadata": result.entry.metadata,
                        "similarity": result.similarity_score
                    }
                    for result in preference_results
                ][:5],
                "patterns_and_insights": [
                    {
                        "content": result.entry.content,
                        "metadata": result.entry.metadata,
                        "similarity": result.similarity_score
                    }
                    for result in pattern_results
                ][:3],
                "recent_time_entries": recent_time_entries,
                "context_summary": self._generate_context_summary(user_input, agent_type, combined_results)
            }

            top_matches = [
                {
                    "category": self._normalize_visual_category(result.entry),
                    "type": self._normalize_visual_type(result.entry, self._normalize_visual_category(result.entry)),
                    "score": round(float(result.similarity_score), 3),
                }
                for result in combined_results[:3]
            ]

            logger.info(
                "[RAG_CONTEXT] agent=%s query=%s total=%d interactions=%d preferences=%d patterns=%d recent_time_entries=%d top_matches=%s",
                agent_type,
                self._truncate_for_log(user_input, 150),
                len(combined_results),
                len(interaction_results),
                len(preference_results),
                len(pattern_results),
                len(recent_time_entries),
                top_matches,
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get contextual knowledge: {e}")
            return {
                "agent_preferences": {},
                "relevant_interactions": [],
                "user_preferences": [],
                "patterns_and_insights": [],
                "recent_time_entries": [],
                "context_summary": "Unable to retrieve context due to system error."
            }

    def _extract_recent_time_entries(
        self,
        interaction_results: List[KnowledgeSearchResult],
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        """Extract concise recent time-entry context from interaction search results."""
        extracted: List[Dict[str, Any]] = []

        for result in interaction_results:
            entry = result.entry
            if not self._is_time_entry_entry(entry):
                continue

            metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
            context_payload = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}

            duration_minutes = context_payload.get("duration_minutes")
            if duration_minutes is None and context_payload.get("duration_seconds") is not None:
                try:
                    duration_minutes = float(context_payload.get("duration_seconds")) / 60.0
                except (TypeError, ValueError):
                    duration_minutes = None

            try:
                normalized_duration = round(max(0.0, float(duration_minutes)), 1) if duration_minutes is not None else None
            except (TypeError, ValueError):
                normalized_duration = None

            extracted.append(
                {
                    "entry_id": entry.entry_id,
                    "project_name": str(context_payload.get("project_name") or "").strip() or "Unassigned",
                    "description": str(context_payload.get("description") or context_payload.get("task_name") or "").strip(),
                    "duration_minutes": normalized_duration,
                    "billable": bool(context_payload.get("billable", False)),
                    "start_time": context_payload.get("start_time"),
                    "end_time": context_payload.get("end_time"),
                    "created_at": entry.created_at.isoformat(),
                    "similarity": result.similarity_score,
                }
            )

        extracted.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return extracted[:limit]

    def _generate_context_summary(self, user_input: str, agent_type: str, search_results: List) -> str:
        """Generate a context summary for the agent."""
        if not search_results:
            return f"No previous context found for {agent_type} requests."
        
        relevant_count = len([r for r in search_results if r.similarity_score > 0.7])
        categories = set(r.entry.category for r in search_results)
        
        summary = f"Found {len(search_results)} related entries ({relevant_count} highly relevant) "
        summary += f"across categories: {', '.join(categories)}. "
        
        # Get most recent interaction
        recent_interactions = [r for r in search_results if r.entry.entry_type == KnowledgeEntryType.INTERACTION]
        if recent_interactions:
            most_recent = max(recent_interactions, key=lambda x: x.entry.created_at)
            summary += f"Most recent similar interaction was on {most_recent.entry.created_at.strftime('%Y-%m-%d')}."

        time_entry_results = [r for r in search_results if self._is_time_entry_entry(r.entry)]
        if time_entry_results:
            most_recent_time_entry = max(time_entry_results, key=lambda x: x.entry.created_at).entry
            metadata = most_recent_time_entry.metadata if isinstance(most_recent_time_entry.metadata, dict) else {}
            context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
            project_name = str(context.get("project_name") or "Unassigned").strip()
            description = str(context.get("description") or context.get("task_name") or "work session").strip()
            summary += f" Recent tracked focus: {project_name} - {description}."
        
        return summary
    
    async def get_relevant_context(self, 
                                  query: str,
                                  agent_type: Optional[str] = None,
                                  max_results: int = 5) -> List[KnowledgeSearchResult]:
        """
        Get relevant context for an agent query using RAG.
        
        Args:
            query: The query to find context for
            agent_type: Filter by specific agent type (optional)
            max_results: Maximum number of context entries to return
            
        Returns:
            List of relevant knowledge entries
        """
        try:
            search_query = KnowledgeQuery(
                query_text=query,
                categories=[agent_type] if agent_type else None,
                limit=max_results,
                similarity_threshold=0.6
            )
            
            return await self.search(search_query)
        except Exception as e:
            logger.error(f"Failed to get relevant context: {e}")
            return []
    
    async def get_stats(self) -> KnowledgeStats:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Knowledge base statistics
        """
        try:
            all_entries = await self.get_all_entries()
            
            # Count by type
            entries_by_type = {}
            for entry_type in KnowledgeEntryType:
                entries_by_type[entry_type] = sum(1 for e in all_entries if e.entry_type == entry_type)
            
            # Count by category
            entries_by_category = {}
            for entry in all_entries:
                entries_by_category[entry.category] = entries_by_category.get(entry.category, 0) + 1
            
            # Get current LLM model for embedding info (but don't initialize if not available)
            embedding_model = "unknown"
            try:
                from ..llm import service as llm_service_module
                if llm_service_module._llm_service and llm_service_module._llm_service._initialized:
                    llm_service = llm_service_module._llm_service
                    current_provider = llm_service.get_current_provider()
                    embedding_model = f"{current_provider}_embedding" if current_provider else "unknown"
            except Exception:
                # Don't fail stats if LLM service is not available
                pass
            
            return KnowledgeStats(
                total_entries=len(all_entries),
                entries_by_type=entries_by_type,
                entries_by_category=entries_by_category,
                last_updated=max((e.updated_at for e in all_entries), default=datetime.utcnow()),
                embedding_model=embedding_model
            )
        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return KnowledgeStats(
                total_entries=0,
                entries_by_type={},
                entries_by_category={},
                last_updated=datetime.utcnow(),
                embedding_model="unknown"
            )
    
    async def clear_all(self) -> bool:
        """
        Clear all entries from the knowledge base.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vector_store.clear()
            self._user_preferences = None
            self._sync_event_index = {}
            self._sync_event_index_loaded = False
            self._embedding_cache = {}
            self._embedding_cache_loaded = False
            self._embedding_provider_cooldown_until = 0.0
            logger.info("Cleared all knowledge base entries")
            return True
        except Exception as e:
            logger.error(f"Failed to clear knowledge base: {e}")
            return False
    
    async def get_embeddings_visualization_data(self) -> List[Dict[str, Any]]:
        """
        Get all embeddings with 3D coordinates for visualization.
        
        Returns:
            List of embedding visualization data
        """
        try:
            # Get all entries and their embeddings
            all_entries = self.vector_store.get_all_entries()
            embeddings_data = []
            
            if not all_entries:
                return []
            
            # Get embeddings from vector store
            embeddings = []
            entries_info = []
            
            for entry in all_entries:
                # Get embedding from vector store
                embedding = self.vector_store.get_embedding(entry.entry_id)
                if embedding is not None:
                    embeddings.append(embedding)
                    entries_info.append(entry)
            
            if not embeddings:
                return []

            import numpy as np
            
            # Try to use PCA for dimensionality reduction, fallback to simple projection
            try:
                from sklearn.decomposition import PCA
                
                # Reduce dimensionality to 3D using PCA
                embeddings_array = np.array(embeddings, dtype=float)

                if embeddings_array.shape[0] < 3:
                    raise ValueError("Not enough points for PCA")

                if not np.isfinite(embeddings_array).all():
                    raise ValueError("Embeddings contain non-finite values")

                if np.allclose(embeddings_array, embeddings_array[0], atol=1e-9):
                    raise ValueError("Embeddings have near-zero variance")

                pca = PCA(n_components=3)
                positions_3d = pca.fit_transform(embeddings_array)

                if not np.isfinite(positions_3d).all():
                    raise ValueError("PCA returned non-finite coordinates")
                
                # Normalize positions to a reasonable range for visualization
                positions_3d = positions_3d * 10  # Scale up for better visualization
                
                logger.info("Using PCA for dimensionality reduction")
                
            except Exception as pca_error:
                logger.warning(f"PCA failed ({pca_error}), using fallback projection")
                
                # Deterministic fallback layout grouped by category.
                import math
                unique_categories = sorted({self._normalize_visual_category(entry) for entry in entries_info})
                category_index = {category: idx for idx, category in enumerate(unique_categories)}
                category_counts: Dict[str, int] = {}
                positions_3d = []

                for entry in entries_info:
                    normalized_category = self._normalize_visual_category(entry)
                    group_idx = category_index.get(normalized_category, 0)
                    group_angle = (group_idx / max(len(unique_categories), 1)) * 2 * math.pi
                    cluster_x = math.cos(group_angle) * 26
                    cluster_z = math.sin(group_angle) * 26

                    local_index = category_counts.get(normalized_category, 0)
                    category_counts[normalized_category] = local_index + 1

                    ring = 1 + (local_index // 8)
                    local_angle = ((local_index % 8) / 8) * 2 * math.pi
                    local_radius = ring * 4.5

                    x = cluster_x + math.cos(local_angle) * local_radius
                    y = ((local_index % 5) - 2) * 3
                    z = cluster_z + math.sin(local_angle) * local_radius
                    
                    positions_3d.append([x, y, z])

            positions_array = np.array(positions_3d, dtype=float)

            # Apply a light repulsion pass to reduce overlap in dense clusters.
            if positions_array.shape[0] > 1:
                min_node_distance = 2.2
                for _ in range(12):
                    moved = False
                    for i in range(len(positions_array)):
                        for j in range(i + 1, len(positions_array)):
                            delta = positions_array[i] - positions_array[j]
                            distance = float(np.linalg.norm(delta))

                            if distance <= 1e-9:
                                delta = np.array([
                                    0.13 * (i + 1),
                                    0.07 * (j + 1),
                                    0.11 * ((i + j) + 1),
                                ], dtype=float)
                                distance = float(np.linalg.norm(delta))

                            if distance < min_node_distance:
                                push = (min_node_distance - distance) * 0.5
                                direction = delta / max(distance, 1e-9)
                                positions_array[i] += direction * push
                                positions_array[j] -= direction * push
                                moved = True

                    if not moved:
                        break

                max_norm = float(np.max(np.linalg.norm(positions_array, axis=1)))
                if max_norm > 0:
                    positions_array = (positions_array / max_norm) * 35.0

            positions_3d = positions_array
            
            # Create visualization data with similarity connections
            for i, (entry, position) in enumerate(zip(entries_info, positions_3d)):
                # Calculate similarities to other entries for connection data
                similarities = []
                current_embedding = embeddings[i]
                
                for j, other_embedding in enumerate(embeddings):
                    if i != j:
                        # Calculate cosine similarity
                        norm_current = np.linalg.norm(current_embedding)
                        norm_other = np.linalg.norm(other_embedding)
                        denominator = norm_current * norm_other

                        if denominator <= 1e-12:
                            continue

                        similarity = float(np.dot(current_embedding, other_embedding) / denominator)

                        if not np.isfinite(similarity):
                            continue

                        if similarity > 0.45:
                            similarities.append({
                                "target_id": entries_info[j].entry_id,
                                "similarity": similarity,
                            })

                similarities.sort(key=lambda value: value["similarity"], reverse=True)
                if similarities:
                    anchor_index = min(2, len(similarities) - 1)
                    adaptive_floor = max(0.55, similarities[anchor_index]["similarity"] - 0.05)
                    similarities = [
                        edge for edge in similarities[:4]
                        if edge["similarity"] >= adaptive_floor
                    ]
                
                visualization_data = {
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "content": entry.content[:200] + "..." if len(entry.content) > 200 else entry.content,
                    "category": self._normalize_visual_category(entry),
                    "entry_type": self._normalize_visual_type(entry, self._normalize_visual_category(entry)),
                    "tags": entry.tags,
                    "embedding": embeddings[i][:10] if len(embeddings[i]) > 10 else embeddings[i],  # First 10 dims for preview
                    "position_3d": position if isinstance(position, list) else position.tolist(),
                    "created_at": entry.created_at.isoformat(),
                    "updated_at": entry.updated_at.isoformat(),
                    "similarities": similarities,
                }
                embeddings_data.append(visualization_data)
            
            logger.info(f"Generated visualization data for {len(embeddings_data)} embeddings")
            return embeddings_data
            
        except Exception as e:
            logger.error(f"Failed to get embeddings visualization data: {e}")
            return []
    
    async def get_embedding_details(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific embedding.
        
        Args:
            entry_id: ID of the entry to get details for
            
        Returns:
            Detailed embedding information or None if not found
        """
        try:
            entry = self.vector_store.get_entry(entry_id)
            if not entry:
                return None
            
            embedding = self.vector_store.get_embedding(entry_id)
            
            # Find similar entries
            if embedding:
                similar_results = self.vector_store.search(
                    query_embedding=embedding,
                    k=6,  # Get 6 to exclude the entry itself
                    similarity_threshold=0.5
                )
                
                # Filter out the entry itself
                similar_entries = [
                    {
                        "entry_id": result.entry.entry_id,
                        "title": result.entry.title,
                        "similarity_score": result.similarity_score,
                        "category": self._normalize_visual_category(result.entry),
                        "entry_type": self._normalize_visual_type(
                            result.entry,
                            self._normalize_visual_category(result.entry),
                        )
                    }
                    for result in similar_results
                    if result.entry.entry_id != entry_id
                ][:5]  # Top 5 similar entries
            else:
                similar_entries = []
            
            details = {
                "entry": {
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "content": entry.content,
                    "category": self._normalize_visual_category(entry),
                    "entry_type": self._normalize_visual_type(entry, self._normalize_visual_category(entry)),
                    "tags": entry.tags,
                    "metadata": entry.metadata,
                    "created_at": entry.created_at.isoformat(),
                    "updated_at": entry.updated_at.isoformat()
                },
                "embedding_info": {
                    "has_embedding": embedding is not None,
                    "embedding_dimension": len(embedding) if embedding else 0,
                    "embedding_preview": embedding[:10] if embedding else None  # First 10 dimensions
                },
                "similar_entries": similar_entries,
                "statistics": {
                    "content_length": len(entry.content),
                    "tag_count": len(entry.tags),
                    "metadata_keys": list(entry.metadata.keys())
                }
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Failed to get embedding details for {entry_id}: {e}")
            return None


# Per-user service instances
_knowledge_base_services: Dict[str, KnowledgeBaseService] = {}


def get_knowledge_base_service(user_id: Optional[str] = None) -> KnowledgeBaseService:
    """Get a user-scoped knowledge base service instance."""
    from app.auth.user_context import get_current_user_id, normalize_user_storage_key

    resolved_user_id = normalize_user_storage_key(user_id or get_current_user_id())
    if resolved_user_id not in _knowledge_base_services:
        _knowledge_base_services[resolved_user_id] = KnowledgeBaseService(resolved_user_id)

    return _knowledge_base_services[resolved_user_id]


def reset_knowledge_base_service(user_id: Optional[str] = None) -> KnowledgeBaseService:
    """Force rebuild the user-scoped knowledge service and reload persisted vector data."""
    from app.auth.user_context import get_current_user_id, normalize_user_storage_key
    from .vector_store import reset_vector_store

    resolved_user_id = normalize_user_storage_key(user_id or get_current_user_id())

    # Reload vector index from disk first, then rebuild service-level caches.
    reset_vector_store(resolved_user_id)
    _knowledge_base_services.pop(resolved_user_id, None)
    _knowledge_base_services[resolved_user_id] = KnowledgeBaseService(resolved_user_id)

    return _knowledge_base_services[resolved_user_id]