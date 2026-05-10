"""Pinecone-backed vector store with local metadata cache for knowledge entries."""

import os
import pickle
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

# Valid user_id pattern: alphanumeric, hyphens, underscores, dots, max 64 chars
VALID_USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9._-]{1,64}$')

_UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)


def _derive_rebuild_title(metadata: dict, category: str, entry_id: str) -> str:
    """Build a human-readable title from Pinecone vector metadata during cache rebuild.

    Old vectors often lack a stored 'title' field. This derives one from the
    richer metadata (context sub-dict, date fields, etc.) so entries display
    meaningfully in the KB explorer instead of showing their UUID.
    """
    stored = str(metadata.get("title") or "").strip()
    if stored and not _UUID_PATTERN.match(stored):
        return stored

    ctx = metadata.get("context")
    ctx = ctx if isinstance(ctx, dict) else {}
    cat = str(category or "").lower()

    if cat in ("time_entry", "time_tracking"):
        project = str(ctx.get("project_name") or metadata.get("project_name") or "").strip()
        activity = str(
            ctx.get("description") or ctx.get("task_name")
            or metadata.get("description") or metadata.get("task_name") or ""
        ).strip()
        duration_raw = ctx.get("duration_minutes") or metadata.get("duration_minutes")
        try:
            dur = int(round(float(duration_raw))) if duration_raw is not None else 0
        except (TypeError, ValueError):
            dur = 0
        suffix = f" ({dur}m)" if dur > 0 else ""
        if project and activity and project.lower() != activity.lower():
            base = f"{project}: {activity}"
        else:
            base = activity or project
        if base:
            compact = base if len(base) <= 90 else f"{base[:87]}..."
            return f"Time Entry - {compact}{suffix}"
        return f"Time Entry{suffix}" if suffix else "Time Entry"

    if cat in ("habit_snapshot", "habit"):
        date_str = str(
            ctx.get("captured_at") or metadata.get("captured_at") or metadata.get("date") or ""
        ).strip()[:10]
        return f"Habit Snapshot{' - ' + date_str if date_str else ''}"

    if cat == "daily_checkup":
        ctype = str(ctx.get("checkup_type") or metadata.get("checkup_type") or "").strip()
        date_str = str(
            ctx.get("checkup_date") or metadata.get("checkup_date") or metadata.get("date") or ""
        ).strip()[:10]
        label = ctype.capitalize() if ctype else "Daily"
        return f"{label} Checkup{' - ' + date_str if date_str else ''}"

    # Intelligence-engine v2 categories: rich content already stored in metadata —
    # surface it in the title so the KB explorer shows the actual semantic
    # signal (per-habit completion, classified time entry summary, ghost-goal
    # warning, etc.) instead of "Habit Snapshot" / generic placeholder.
    if cat == "habit_entry":
        habit_name = str(metadata.get("habit_name") or ctx.get("habit_name") or "").strip()
        comp_7d = metadata.get("completion_7d") if metadata.get("completion_7d") is not None else ctx.get("completion_7d")
        streak = metadata.get("streak") if metadata.get("streak") is not None else ctx.get("streak")
        bits = []
        if habit_name:
            bits.append(habit_name)
        if comp_7d is not None:
            bits.append(f"completion_7d:{comp_7d}%")
        if streak is not None:
            bits.append(f"streak:{streak}d")
        if bits:
            return "Habit · " + " · ".join(str(b) for b in bits)
        return "Habit Entry"

    if cat == "daily_habit_summary":
        date_str = str(
            metadata.get("checkup_date") or ctx.get("checkup_date") or ""
        ).strip()[:10]
        return f"Habits Daily Summary{' - ' + date_str if date_str else ''}"

    if cat in ("time_entry_v2", "time_tracking_v2"):
        work_type = str(metadata.get("work_type") or "").replace("_", " ").strip()
        prod = metadata.get("productivity_score")
        focus = metadata.get("focus_quality")
        bits = []
        if work_type:
            bits.append(work_type)
        if prod is not None:
            bits.append(f"productivity:{prod}")
        if focus is not None:
            bits.append(f"focus:{focus}")
        return "Time · " + " · ".join(str(b) for b in bits) if bits else "Time Entry"

    if cat in ("goal_v2",):
        title_part = str(metadata.get("title") or ctx.get("title") or "").strip()
        status = str(metadata.get("status") or "").strip()
        invested = metadata.get("invested_hours")
        bits = []
        if title_part:
            bits.append(title_part)
        if status:
            bits.append(f"status:{status}")
        if invested is not None:
            bits.append(f"invested:{invested}h")
        return "Goal · " + " · ".join(str(b) for b in bits) if bits else "Goal"

    if cat == "task_entry":
        title_part = str(metadata.get("title") or ctx.get("title") or "").strip()
        status = str(metadata.get("status") or "").strip()
        priority = str(metadata.get("priority") or "").strip()
        bits = [b for b in [title_part, status, priority] if b]
        return "Task · " + " · ".join(bits) if bits else "Task"

    if cat == "morning_intent":
        date_str = str(metadata.get("checkup_date") or "").strip()[:10]
        target = str(metadata.get("focus_target") or "").strip()
        if target:
            target = target if len(target) <= 60 else f"{target[:57]}..."
        return f"Morning Intent{' - ' + date_str if date_str else ''}{' · ' + target if target else ''}"

    if cat == "evening_reflection":
        date_str = str(metadata.get("checkup_date") or "").strip()[:10]
        return f"Evening Reflection{' - ' + date_str if date_str else ''}"

    if cat in ("insight_v2", "pattern_v2"):
        kind = str(metadata.get("insight_type") or metadata.get("pattern_type") or "").replace("_", " ").strip()
        return ("Insight · " if cat == "insight_v2" else "Pattern · ") + (kind.title() if kind else cat.replace("_", " ").title())

    if cat == "tag_catalog":
        tag_name = str(metadata.get("tag_name") or metadata.get("name") or "").strip()
        return f"Tag · {tag_name}" if tag_name else "Tag"

    # Generic: use category as human-readable title
    cat_display = category.replace("_", " ").title()
    return cat_display if cat_display else entry_id


def _derive_rebuild_content(metadata: dict, category: str, title: str) -> str:
    """Reconstruct a content string from category-specific metadata fields.

    Used when the original embedding text was lost (cold rebuild from Pinecone
    against pre-enrichment vectors). Each branch reproduces the same single-line
    "intelligence engine" format the per-type embedding builders use, so the
    KB explorer card body matches what semantic search actually matches on.
    """
    cat = str(category or "").lower()
    ctx = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}

    def _get(*keys):
        for k in keys:
            v = metadata.get(k)
            if v is None or v == "":
                v = ctx.get(k) if isinstance(ctx, dict) else None
            if v not in (None, ""):
                return v
        return None

    if cat == "habit_entry":
        bits = [
            f"habit:{_get('habit_name') or 'unknown'}",
            f"streak:{_get('streak') or 0}d",
            f"completion_7d:{_get('completion_7d') or 0}%",
            f"completion_30d:{_get('completion_30d') or 0}%",
        ]
        for k in ("trend", "pattern", "priority", "last_completed", "last_skipped"):
            v = _get(k)
            if v not in (None, "", 0):
                bits.append(f"{k}:{v}")
        return " | ".join(bits)

    if cat == "daily_habit_summary":
        date = _get("checkup_date") or ""
        total = _get("total_habits") or 0
        return f"daily_habits:{date} | total:{total}"

    if cat in ("time_entry", "time_entry_v2"):
        bits = []
        wt = _get("work_type")
        if wt:
            bits.append(str(wt))
        ep = _get("energy_pattern")
        fq = _get("focus_quality")
        prod = _get("productivity_score")
        if fq is not None:
            bits.append(f"focus:{fq}")
        if ep:
            bits.append(f"energy:{ep}")
        if prod is not None:
            bits.append(f"productivity:{prod}")
        wd = _get("weekday")
        hr = _get("hour_of_day")
        if wd:
            bits.append(f"weekday:{wd}")
        if hr is not None:
            bits.append(f"hour:{hr}")
        return " | ".join(bits) if bits else title

    if cat == "goal_v2":
        bits = [f"goal:{title}"]
        for k in ("status", "invested_hours", "hours_this_month", "days_remaining", "priority"):
            v = _get(k)
            if v not in (None, ""):
                bits.append(f"{k}:{v}")
        return " | ".join(bits)

    if cat == "task_entry":
        bits = [f"task:{title}"]
        for k in ("status", "priority", "due_date", "linked_goal"):
            v = _get(k)
            if v not in (None, ""):
                bits.append(f"{k}:{v}")
        return " | ".join(bits)

    if cat == "morning_intent":
        return f"morning_intent:{_get('checkup_date') or ''} | focus_target:{_get('focus_target') or ''}"

    if cat == "evening_reflection":
        return f"evening_reflection:{_get('checkup_date') or ''}"

    if cat in ("insight_v2", "pattern_v2"):
        kind = _get("insight_type") or _get("pattern_type") or ""
        return f"{cat}:{kind}" if kind else title

    if cat == "tag_catalog":
        return f"tag:{_get('tag_name') or _get('name') or title}"

    return title or category.replace("_", " ").title()


import numpy as np

from ..models.knowledge import KnowledgeEntry, KnowledgeEntryType, KnowledgeEntrySubType, KnowledgeSearchResult
from .storage_paths import resolve_data_path
from ..utils.structured_logging import get_logger, LogComponent

logger = get_logger(__name__, LogComponent.STORE)


class PineconeVectorStore:
    """Pinecone-backed vector store that mirrors KnowledgeEntry metadata locally."""

    def __init__(
        self,
        user_id: str,
        dimension: int = 1536,
        local_metadata_path: Optional[str] = None,
        index_name: Optional[str] = None,
    ):
        self.user_id = user_id
        self.dimension = dimension
        self.index_name = (index_name or os.getenv("PINECONE_INDEX_NAME", "agentic-knowledge")).strip()
        self.metric = (os.getenv("PINECONE_METRIC", "cosine") or "cosine").strip().lower()
        self.namespace = self._resolve_namespace(user_id)

        if not self.index_name:
            raise RuntimeError("Pinecone index name is required via PINECONE_INDEX_NAME")

        api_key = (os.getenv("PINECONE_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not configured")

        try:
            from pinecone import Pinecone, ServerlessSpec
        except Exception as exc:
            raise RuntimeError(
                "pinecone package is not installed. Add it to requirements and install dependencies."
            ) from exc

        self._serverless_spec_cls = ServerlessSpec
        self._client = Pinecone(api_key=api_key)

        if not local_metadata_path:
            local_metadata_path = self._resolve_default_metadata_path(user_id)
        self.metadata_path = local_metadata_path
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        self.entry_metadata: Dict[str, KnowledgeEntry] = {}
        self._load_metadata()
        # Fix mistyped v2 entries that an older rebuild path persisted as
        # "Preference / Other Preference" because it didn't recognise the new
        # categories. Idempotent — entries already on the right type are no-ops.
        self._reclassify_v2_categories_in_cache()

        self._ensure_index_exists()
        self.index = self._client.Index(self.index_name)

        # On startup, verify local cache count matches Pinecone. If Pinecone has
        # significantly more vectors than the cache (threshold = 5), rebuild so that
        # get_all_entries() doesn't return stale/incomplete data.
        self._verify_and_refresh_cache()

    @staticmethod
    def _parse_bool_env(name: str, default: bool) -> bool:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default

        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _extract_field(payload: Any, field: str, default: Any = None) -> Any:
        if payload is None:
            return default

        if isinstance(payload, dict):
            return payload.get(field, default)

        if hasattr(payload, field):
            return getattr(payload, field)

        return default

    @staticmethod
    def _resolve_default_metadata_path(user_id: str) -> str:
        if user_id == "single_user":
            return resolve_data_path("vector_index_pinecone_metadata.pkl")

        return resolve_data_path("users", user_id, "vector_index_pinecone_metadata.pkl")

    @staticmethod
    def _resolve_namespace(user_id: str) -> str:
        configured_namespace = (os.getenv("PINECONE_NAMESPACE") or "").strip()
        if configured_namespace:
            return configured_namespace

        prefix = (os.getenv("PINECONE_NAMESPACE_PREFIX") or "agentic").strip().lower()
        normalized_user = re.sub(r"[^a-zA-Z0-9_-]", "-", user_id).strip("-").lower() or "single-user"
        namespace = f"{prefix}-{normalized_user}" if prefix else normalized_user
        return namespace[:120]

    def _load_metadata(self) -> None:
        try:
            if not os.path.exists(self.metadata_path):
                return

            with open(self.metadata_path, "rb") as metadata_file:
                payload = pickle.load(metadata_file)

            raw_entries = payload.get("entry_metadata", {}) if isinstance(payload, dict) else {}
            hydrated: Dict[str, KnowledgeEntry] = {}
            for entry_id, raw_entry in raw_entries.items():
                if isinstance(raw_entry, KnowledgeEntry):
                    hydrated[entry_id] = raw_entry
                elif isinstance(raw_entry, dict):
                    try:
                        hydrated[entry_id] = KnowledgeEntry.model_validate(raw_entry)
                    except Exception:
                        continue

            self.entry_metadata = hydrated
            logger.info(
                "metadata_cache_loaded",
                f"Loaded Pinecone metadata cache with {len(self.entry_metadata)} entries for namespace {self.namespace}",
                {"entry_count": len(self.entry_metadata), "namespace": self.namespace}
            )
        except Exception as exc:
            logger.warning("metadata_cache_load_failed", f"Failed to load Pinecone metadata cache: {exc}", {"error": str(exc)})
            self.entry_metadata = {}

    def _get_pinecone_vector_count(self) -> int:
        """Query Pinecone for the actual vector count in this namespace."""
        try:
            stats = self.index.describe_index_stats()
            namespaces = self._extract_field(stats, "namespaces", {}) or {}
            namespace_stats = namespaces.get(self.namespace, {}) if isinstance(namespaces, dict) else {}
            vector_count = self._extract_field(namespace_stats, "vector_count", 0) or 0
            return int(vector_count)
        except Exception as exc:
            logger.warning("pinecone_count_failed", f"Failed to get Pinecone vector count: {exc}", {"error": str(exc)})
            return -1

    def _rebuild_cache_from_pinecone(self) -> None:
        """Rebuild the local metadata cache by fetching all vector IDs from Pinecone and
        hydrating their metadata via index.fetch()."""
        try:
            logger.info(
                "rebuilding_metadata_cache",
                f"Rebuilding metadata cache from Pinecone for namespace {self.namespace}",
                {"namespace": self.namespace}
            )
            all_ids: List[str] = []
            # Pinecone list() paginates IDs — iterate through pages
            list_response = self.index.list(namespace=self.namespace)
            for page in list_response:
                if isinstance(page, list):
                    all_ids.extend([str(item) for item in page])
                elif isinstance(page, str):
                    all_ids.append(page)
                elif hasattr(page, "__iter__"):
                    for item in page:
                        all_ids.append(str(item))

            if not all_ids:
                logger.info("no_vectors_in_namespace", f"No vectors found in namespace {self.namespace}", {"namespace": self.namespace})
                return

            # Fetch metadata in batches of 100 (Pinecone fetch limit)
            batch_size = 100
            rebuilt: Dict[str, KnowledgeEntry] = {}
            for batch_start in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[batch_start:batch_start + batch_size]
                try:
                    fetch_response = self.index.fetch(ids=batch_ids, namespace=self.namespace)
                    vectors = self._extract_field(fetch_response, "vectors", {}) or {}
                    if not isinstance(vectors, dict):
                        continue
                    for vector_id, vector_data in vectors.items():
                        metadata = self._extract_field(vector_data, "metadata", {}) or {}
                        # Prefer existing cache entry if already hydrated
                        if vector_id in self.entry_metadata:
                            rebuilt[vector_id] = self.entry_metadata[vector_id]
                            continue
                        # Build a minimal KnowledgeEntry from Pinecone metadata
                        try:
                            entry_id = str(metadata.get("entry_id") or vector_id)
                            user_id = str(metadata.get("user_id") or self.user_id)
                            category = str(metadata.get("category") or "")

                            # Resolve entry_type — old vectors may not have it stored
                            raw_entry_type = metadata.get("entry_type")
                            raw_entry_sub_type = metadata.get("entry_sub_type")
                            if not raw_entry_type:
                                cat_lower = category.lower()
                                # Intelligence-engine v2 categories carry richer
                                # type metadata once written; this map is the
                                # rebuild fallback for older vectors that
                                # predate _build_pinecone_metadata enrichment.
                                if cat_lower in ("time_entry", "time_tracking", "time_entry_v2"):
                                    raw_entry_type = "interaction"
                                    raw_entry_sub_type = raw_entry_sub_type or "work interaction"
                                elif cat_lower in ("habit_snapshot", "habit", "habit_entry", "daily_habit_summary"):
                                    raw_entry_type = "interaction"
                                    raw_entry_sub_type = raw_entry_sub_type or "health interaction"
                                elif cat_lower in ("goal", "goals", "goal_v2"):
                                    raw_entry_type = "insight" if cat_lower == "goal_v2" else "preference"
                                    raw_entry_sub_type = raw_entry_sub_type or "goal"
                                elif cat_lower in ("morning_intent", "evening_reflection", "daily_checkup"):
                                    raw_entry_type = "insight"
                                    raw_entry_sub_type = raw_entry_sub_type or "misc insight"
                                elif cat_lower in ("insight", "insights", "insight_v2"):
                                    raw_entry_type = "insight"
                                    raw_entry_sub_type = raw_entry_sub_type or "important insight"
                                elif cat_lower == "pattern_v2":
                                    raw_entry_type = "pattern"
                                    raw_entry_sub_type = raw_entry_sub_type or "conscious patterns"
                                elif cat_lower == "task_entry":
                                    raw_entry_type = "interaction"
                                    raw_entry_sub_type = raw_entry_sub_type or "work interaction"
                                elif cat_lower == "tag_catalog":
                                    raw_entry_type = "preference"
                                    raw_entry_sub_type = raw_entry_sub_type or "other preference"
                                elif cat_lower == "chat_interaction":
                                    raw_entry_type = "interaction"
                                    raw_entry_sub_type = raw_entry_sub_type or "personal interaction"
                                elif cat_lower in ("user_preference", "preference_update"):
                                    raw_entry_type = "user_preference"
                                    raw_entry_sub_type = raw_entry_sub_type or "user profile"
                                else:
                                    raw_entry_type = "preference"
                                    raw_entry_sub_type = raw_entry_sub_type or "other preference"

                            try:
                                entry_type_enum = KnowledgeEntryType(raw_entry_type)
                            except (ValueError, KeyError):
                                entry_type_enum = KnowledgeEntryType.INTERACTION
                            try:
                                entry_sub_type_enum = KnowledgeEntrySubType(raw_entry_sub_type)
                            except (ValueError, KeyError):
                                entry_sub_type_enum = KnowledgeEntrySubType.MISC_INTERACTION

                            title = _derive_rebuild_title(metadata, category, entry_id)
                            # Surface the rich embedding text as content when
                            # the metadata captured a slice of it. Without this,
                            # the KB explorer rendered a generic placeholder
                            # ("Knowledge entry captured for X context.") for
                            # every v2 entry on cold rebuild.
                            content_value = str(
                                metadata.get("content")
                                or metadata.get("embedding_text")
                                or ""
                            )
                            if not content_value:
                                content_value = _derive_rebuild_content(metadata, category, title)
                            rebuilt[vector_id] = KnowledgeEntry(
                                entry_id=entry_id,
                                user_id=user_id,
                                entry_type=entry_type_enum,
                                entry_sub_type=entry_sub_type_enum,
                                category=category,
                                title=title,
                                content=content_value,
                                metadata=metadata,
                            )
                        except Exception as build_exc:
                            logger.warning(
                                "rebuild_entry_failed",
                                f"Failed to build entry for vector {vector_id}: {build_exc}",
                                {"vector_id": vector_id, "error": str(build_exc)}
                            )
                except Exception as batch_exc:
                    logger.warning("fetch_batch_failed", f"Failed to fetch batch from Pinecone: {batch_exc}", {"error": str(batch_exc)})

            if rebuilt:
                self.entry_metadata = rebuilt
                self._save_metadata()
                logger.info(
                    "metadata_cache_rebuilt",
                    f"Rebuilt metadata cache with {len(rebuilt)} entries from Pinecone",
                    {"entry_count": len(rebuilt), "namespace": self.namespace}
                )
        except Exception as exc:
            logger.warning("rebuild_cache_failed", f"Failed to rebuild metadata cache from Pinecone: {exc}", {"error": str(exc)})

    def _reclassify_v2_categories_in_cache(self) -> None:
        """Repair entries that an older rebuild typed as Preference/Other Preference.

        The first version of _rebuild_cache_from_pinecone had no mapping for v2
        categories (time_entry_v2, habit_entry, daily_habit_summary, goal_v2,
        task_entry, morning_intent, evening_reflection, insight_v2, pattern_v2,
        tag_catalog) and fell through to the generic
        "preference / other preference" default. Existing pickle caches retain
        that mistyping.

        This pass walks the in-memory cache, fixes the type pair for known
        v2 categories, and rewrites titles + content from the metadata slice
        so the KB explorer card shows a meaningful summary instead of the
        generic "Knowledge entry captured for X context." placeholder.
        """
        v2_type_map: Dict[str, tuple] = {
            "time_entry_v2": ("interaction", "work interaction"),
            "habit_entry": ("interaction", "health interaction"),
            "daily_habit_summary": ("interaction", "health interaction"),
            "goal_v2": ("insight", "goal"),
            "task_entry": ("interaction", "work interaction"),
            "morning_intent": ("insight", "misc insight"),
            "evening_reflection": ("insight", "misc insight"),
            "insight_v2": ("insight", "important insight"),
            "pattern_v2": ("pattern", "conscious patterns"),
            "tag_catalog": ("preference", "other preference"),
            "chat_interaction": ("interaction", "personal interaction"),
        }
        if not self.entry_metadata:
            return

        changed = 0
        for entry_id, entry in list(self.entry_metadata.items()):
            try:
                category = str(getattr(entry, "category", "") or "").lower()
                if category not in v2_type_map:
                    continue
                target_type, target_sub = v2_type_map[category]
                current_type = str(getattr(entry.entry_type, "value", entry.entry_type) or "").lower()
                current_sub = str(getattr(entry.entry_sub_type, "value", entry.entry_sub_type) or "").lower()
                needs_type_fix = (current_type, current_sub) != (target_type, target_sub)
                generic_content = str(getattr(entry, "content", "") or "").strip()
                needs_content_fix = (
                    not generic_content
                    or generic_content.lower().startswith("knowledge entry captured for")
                )

                if not needs_type_fix and not needs_content_fix:
                    continue

                metadata = getattr(entry, "metadata", None) or {}
                derived_title = _derive_rebuild_title(metadata, category, entry_id)
                derived_content = (
                    generic_content
                    if generic_content and not needs_content_fix
                    else _derive_rebuild_content(metadata, category, derived_title)
                )

                if needs_type_fix:
                    try:
                        entry.entry_type = KnowledgeEntryType(target_type)
                    except (ValueError, KeyError):
                        pass
                    try:
                        entry.entry_sub_type = KnowledgeEntrySubType(target_sub)
                    except (ValueError, KeyError):
                        pass

                if derived_title and derived_title != entry.title:
                    entry.title = derived_title
                if derived_content and derived_content != entry.content:
                    entry.content = derived_content

                changed += 1
            except Exception as exc:
                logger.debug("reclassify_failed", f"Skipping entry {entry_id}: {exc}", {"entry_id": entry_id})

        if changed:
            try:
                self._save_metadata()
            except Exception as exc:
                logger.warning("reclassify_save_failed", f"Failed to persist reclassified cache: {exc}", {"error": str(exc)})
            logger.info(
                "v2_categories_reclassified",
                f"Reclassified {changed} v2 cache entries",
                {"count": changed, "namespace": self.namespace}
            )

    def _save_metadata(self) -> None:
        try:
            payload = {
                "entry_metadata": self.entry_metadata,
                "updated_at": datetime.utcnow().isoformat(),
                "namespace": self.namespace,
                "index_name": self.index_name,
            }
            with open(self.metadata_path, "wb") as metadata_file:
                pickle.dump(payload, metadata_file)
        except Exception as exc:
            logger.error("metadata_cache_save_failed", f"Failed to save Pinecone metadata cache: {exc}", {"error": str(exc)})
            raise

    def persist_metadata_cache(self) -> None:
        """Persist the local metadata cache to disk."""
        self._save_metadata()

    def _verify_and_refresh_cache(self, threshold: int = 5) -> None:
        """Compare local cache size against Pinecone vector count. If Pinecone has
        more than `threshold` additional vectors, rebuild the local cache from Pinecone."""
        pinecone_count = self._get_pinecone_vector_count()
        if pinecone_count < 0:
            # Could not reach Pinecone — skip refresh to avoid disruption
            return
        cache_count = len(self.entry_metadata)
        if pinecone_count > cache_count + threshold:
            logger.info(
                "cache_stale_detected",
                f"Pinecone has {pinecone_count} vectors but cache has {cache_count} entries — rebuilding",
                {"pinecone_count": pinecone_count, "cache_count": cache_count, "namespace": self.namespace}
            )
            self._rebuild_cache_from_pinecone()

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        if not embedding:
            raise ValueError("Embedding is empty")

        if len(embedding) > self.dimension:
            fitted = list(embedding[: self.dimension])
        elif len(embedding) < self.dimension:
            fitted = list(embedding) + [0.0] * (self.dimension - len(embedding))
        else:
            fitted = list(embedding)

        vector = np.array(fitted, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def _iter_index_names(self) -> List[str]:
        try:
            listing = self._client.list_indexes()
        except Exception as exc:
            logger.warning("list_indexes_failed", f"Unable to list Pinecone indexes: {exc}", {"error": str(exc)})
            return []

        if hasattr(listing, "names") and callable(getattr(listing, "names")):
            try:
                return list(listing.names())
            except Exception:
                return []

        if isinstance(listing, dict):
            index_items = listing.get("indexes") or listing.get("data") or []
            return [
                str(item.get("name"))
                for item in index_items
                if isinstance(item, dict) and item.get("name")
            ]

        if isinstance(listing, list):
            names: List[str] = []
            for item in listing:
                if isinstance(item, dict) and item.get("name"):
                    names.append(str(item["name"]))
                elif hasattr(item, "name"):
                    names.append(str(getattr(item, "name")))
                elif isinstance(item, str):
                    names.append(item)
            return names

        return []

    def _ensure_index_exists(self) -> None:
        available_names = set(self._iter_index_names())
        if self.index_name in available_names:
            return

        create_enabled = self._parse_bool_env("PINECONE_CREATE_INDEX", True)
        if not create_enabled:
            raise RuntimeError(
                f"Pinecone index '{self.index_name}' does not exist and PINECONE_CREATE_INDEX is false"
            )

        cloud = (os.getenv("PINECONE_CLOUD") or "aws").strip()
        region = (os.getenv("PINECONE_REGION") or "us-east-1").strip()

        logger.info(
            "creating_pinecone_index",
            f"Creating Pinecone index '{self.index_name}' (metric={self.metric}, dimension={self.dimension}, cloud={cloud}, region={region})",
            {"index_name": self.index_name, "metric": self.metric, "dimension": self.dimension, "cloud": cloud, "region": region},
        )

        self._client.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=self._serverless_spec_cls(cloud=cloud, region=region),
        )

    def _fetch_embedding_values(self, entry_id: str) -> Optional[List[float]]:
        try:
            response = self.index.fetch(ids=[entry_id], namespace=self.namespace)
            vectors = self._extract_field(response, "vectors", {}) or {}

            vector_payload: Any = None
            if isinstance(vectors, dict):
                vector_payload = vectors.get(entry_id)
            elif hasattr(vectors, "get"):
                vector_payload = vectors.get(entry_id)

            if not vector_payload:
                return None

            values = self._extract_field(vector_payload, "values", None)
            if values is None:
                return None

            return [float(value) for value in values]
        except Exception as exc:
            logger.warning("fetch_vector_failed", f"Failed to fetch Pinecone vector {entry_id}: {exc}", {"entry_id": entry_id, "error": str(exc)})
            return None

    def _load_entry_from_db(self, entry_id: str) -> Optional[KnowledgeEntry]:
        try:
            from .knowledge_db_store import get_knowledge_db_store

            db_store = get_knowledge_db_store()
            if not db_store or not db_store.is_available:
                return None

            entry = db_store.get_entry(self.user_id, entry_id)
            if not entry:
                return None

            embedding = self._fetch_embedding_values(entry_id)
            if embedding:
                entry.embedding = embedding

            self.entry_metadata[entry.entry_id] = entry
            return entry
        except Exception as exc:
            logger.warning("hydrate_entry_failed", f"Failed to hydrate entry {entry_id} from DB: {exc}", {"entry_id": entry_id, "error": str(exc)})
            return None

    def add_entry(self, entry: KnowledgeEntry, embedding: List[float], persist: bool = True) -> None:
        normalized_embedding = self._normalize_embedding(embedding)

        # Pinecone vector metadata: keep core IDs + a category-specific
        # signal slice so cold-rebuilds don't lose per-vector intelligence
        # (without DB store this was the only source of truth, and
        # habit_entry / time_entry_v2 / goal_v2 came back null-fielded).
        pinecone_metadata = self._build_pinecone_metadata(entry)

        self.index.upsert(
            vectors=[
                {
                    "id": entry.entry_id,
                    "values": normalized_embedding,
                    "metadata": pinecone_metadata,
                }
            ],
            namespace=self.namespace,
        )

        entry_copy = entry.model_copy()
        entry_copy.embedding = list(embedding)
        self.entry_metadata[entry.entry_id] = entry_copy

        if persist:
            self._save_metadata()

    def _build_pinecone_metadata(self, entry: KnowledgeEntry) -> Dict[str, Any]:
        """Compose the metadata dict that goes alongside the vector in Pinecone.

        Always includes core identity. Pulls a small category-specific slice
        from {@code entry.metadata} so cold cache rebuilds (which only see
        Pinecone metadata) preserve enough signal to reconstruct the
        intelligence-engine view — e.g. per-habit completion rates, time
        entry productivity scores, goal status. Pinecone caps metadata at
        ~40KB per vector so we slice rather than dump everything.
        """
        pinecone_meta: Dict[str, Any] = {
            "entry_id": entry.entry_id,
            "user_id": self.user_id,
            "category": str(entry.category or ""),
            "entry_type": str(getattr(entry.entry_type, "value", entry.entry_type) or ""),
            "entry_sub_type": str(getattr(entry.entry_sub_type, "value", entry.entry_sub_type) or ""),
        }
        try:
            if entry.title:
                pinecone_meta["title"] = str(entry.title)[:200]
        except Exception:
            pass

        category = (entry.category or "").lower()
        local_meta = entry.metadata if isinstance(entry.metadata, dict) else {}

        # Per-category signal preservation. Only fields that round-trip to
        # primitives (str/int/float/bool/list-of-str) — Pinecone rejects
        # nested dicts at the top level of metadata.
        slice_keys: List[str] = []
        if category == "habit_entry":
            slice_keys = [
                "habit_id", "habit_name", "completion_7d", "completion_30d",
                "streak", "pattern", "priority", "trend",
                "last_completed", "last_skipped", "has_note_to_ai",
                "checkup_date", "captured_at",
            ]
        elif category == "daily_habit_summary":
            slice_keys = ["checkup_date", "captured_at", "total_habits"]
        elif category in ("time_entry", "time_entry_v2"):
            slice_keys = [
                "start_time", "end_time", "duration_minutes",
                "weekday", "hour_of_day", "linked_goal", "project_id",
                "work_type", "energy_pattern", "focus_quality",
                "productivity_score", "is_first_entry_of_day", "is_last_entry_of_day",
            ]
        elif category in ("goal", "goals", "goal_v2"):
            slice_keys = [
                "goal_id", "status", "invested_hours", "hours_this_month",
                "days_remaining", "progress_percent", "priority", "category",
            ]
        elif category in ("morning_intent", "evening_reflection"):
            slice_keys = ["checkup_date", "captured_at", "focus_target"]
        elif category == "task_entry":
            slice_keys = [
                "task_id", "status", "priority", "due_date",
                "linked_goal", "has_note_to_ai",
            ]
        elif category in ("insight_v2", "pattern_v2"):
            slice_keys = [
                "insight_type", "pattern_type", "severity",
                "generated_by", "generated_at", "goal_id", "habit_id",
                "frequency", "delta", "direction",
            ]
        elif category == "tag_catalog":
            slice_keys = ["tag_id", "tag_name", "name", "usage_count"]

        for key in slice_keys:
            value = local_meta.get(key)
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                pinecone_meta[key] = value
            elif isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value):
                pinecone_meta[key] = [str(v) for v in value]
            else:
                # Skip nested types — Pinecone metadata doesn't accept them.
                continue

        return pinecone_meta

    def update_entry(self, entry: KnowledgeEntry, embedding: List[float], persist: bool = True) -> None:
        self.add_entry(entry, embedding, persist=persist)

    def remove_entry(self, entry_id: str, persist: bool = True) -> bool:
        if not entry_id:
            return False

        try:
            self.index.delete(ids=[entry_id], namespace=self.namespace)
            self.entry_metadata.pop(entry_id, None)

            if persist:
                self._save_metadata()

            return True
        except Exception as exc:
            logger.error("remove_entry_failed", f"Failed to remove entry from Pinecone store: {exc}", {"error": str(exc)})
            return False

    def remove_entries(self, entry_ids: List[str], persist: bool = True) -> int:
        normalized_ids = [entry_id for entry_id in entry_ids if entry_id]
        if not normalized_ids:
            return 0

        try:
            self.index.delete(ids=normalized_ids, namespace=self.namespace)
            for entry_id in normalized_ids:
                self.entry_metadata.pop(entry_id, None)

            if persist:
                self._save_metadata()

            return len(normalized_ids)
        except Exception as exc:
            logger.error("remove_entries_failed", f"Failed to remove entries from Pinecone store: {exc}", {"error": str(exc)})
            return 0

    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[KnowledgeSearchResult]:
        if k <= 0:
            return []

        try:
            normalized_query = self._normalize_embedding(query_embedding)
            response = self.index.query(
                vector=normalized_query,
                top_k=max(1, int(k)),
                namespace=self.namespace,
                include_metadata=True,
                include_values=False,
            )
            matches = self._extract_field(response, "matches", []) or []

            results: List[KnowledgeSearchResult] = []
            for match in matches:
                entry_id = str(self._extract_field(match, "id", "") or "")
                if not entry_id:
                    continue

                score = float(self._extract_field(match, "score", 0.0) or 0.0)
                if score < similarity_threshold:
                    continue

                entry = self.entry_metadata.get(entry_id)
                if not entry:
                    entry = self._load_entry_from_db(entry_id)

                if not entry:
                    continue

                results.append(
                    KnowledgeSearchResult(
                        entry=entry,
                        similarity_score=score,
                    )
                )

            return results
        except Exception as exc:
            logger.error("search_failed", f"Failed to search Pinecone vector store: {exc}", {"error": str(exc)})
            return []

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        entry = self.entry_metadata.get(entry_id)
        if entry:
            return entry

        return self._load_entry_from_db(entry_id)

    def get_embedding(self, entry_id: str) -> Optional[List[float]]:
        entry = self.get_entry(entry_id)
        if entry and entry.embedding:
            return list(entry.embedding)

        embedding = self._fetch_embedding_values(entry_id)
        if embedding and entry:
            entry.embedding = list(embedding)
            self.entry_metadata[entry_id] = entry

        return embedding

    def get_all_embeddings(self) -> Dict[str, List[float]]:
        embeddings: Dict[str, List[float]] = {}
        for entry in self.get_all_entries():
            resolved = self.get_embedding(entry.entry_id)
            if resolved:
                embeddings[entry.entry_id] = resolved

        return embeddings

    def get_all_entries(self, force_refresh: bool = False) -> List[KnowledgeEntry]:
        if force_refresh:
            self._rebuild_cache_from_pinecone()

        if self.entry_metadata:
            return list(self.entry_metadata.values())

        try:
            from .knowledge_db_store import get_knowledge_db_store

            db_store = get_knowledge_db_store()
            if not db_store or not db_store.is_available:
                return []

            entries = db_store.list_entries(self.user_id)
            for entry in entries:
                self.entry_metadata[entry.entry_id] = entry

            return list(self.entry_metadata.values())
        except Exception as exc:
            logger.warning("list_entries_failed", f"Failed to list entries from DB for Pinecone store: {exc}", {"error": str(exc)})
            return []

    def get_stats(self) -> Dict[str, Any]:
        total_entries = len(self.entry_metadata)

        try:
            stats = self.index.describe_index_stats()
            namespaces = self._extract_field(stats, "namespaces", {}) or {}
            namespace_stats = namespaces.get(self.namespace, {}) if isinstance(namespaces, dict) else {}
            vector_count = self._extract_field(namespace_stats, "vector_count", 0) or 0
            total_entries = max(total_entries, int(vector_count))
        except Exception as exc:
            logger.warning("index_stats_failed", f"Failed to read Pinecone index stats: {exc}", {"error": str(exc)})

        return {
            "total_entries": total_entries,
            "dimension": self.dimension,
            "index_size_mb": 0,
            "last_updated": datetime.utcnow().isoformat(),
            "provider": "pinecone",
            "index_name": self.index_name,
            "namespace": self.namespace,
        }

    def clear(self) -> None:
        self.index.delete(delete_all=True, namespace=self.namespace)
        self.entry_metadata = {}
        self._save_metadata()
