"""Pinecone-backed vector store with local metadata cache for knowledge entries."""

import os
import pickle
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

# Valid user_id pattern: alphanumeric, hyphens, underscores, dots, max 64 chars
VALID_USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9._-]{1,64}$')

import numpy as np

from ..models.knowledge import KnowledgeEntry, KnowledgeSearchResult
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

        self._ensure_index_exists()
        self.index = self._client.Index(self.index_name)

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

        self.index.upsert(
            vectors=[
                {
                    "id": entry.entry_id,
                    "values": normalized_embedding,
                    "metadata": {
                        "entry_id": entry.entry_id,
                        "user_id": self.user_id,
                        "category": str(entry.category or ""),
                    },
                }
            ],
            namespace=self.namespace,
        )

        entry_copy = entry.model_copy()
        entry_copy.embedding = list(embedding)
        self.entry_metadata[entry.entry_id] = entry_copy

        if persist:
            self._save_metadata()

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

    def get_all_entries(self) -> List[KnowledgeEntry]:
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
