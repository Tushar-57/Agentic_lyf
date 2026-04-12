"""
FAISS-based vector store for knowledge base embeddings.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
import numpy as np
from datetime import datetime

# Maximum entries in FAISS index to prevent unbounded memory growth
MAX_VECTOR_STORE_ENTRIES = 5000

try:
    import faiss
except Exception:  # pragma: no cover - environment-specific optional dependency
    faiss = None

from ..models.knowledge import KnowledgeEntry, KnowledgeSearchResult
from .storage_paths import resolve_data_path

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for storing and retrieving embeddings."""
    
    def __init__(self, dimension: int = 1536, index_path: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embeddings (1536 for standardized embeddings across providers)
            index_path: Path to store the FAISS index and metadata
        """
        if not index_path:
            index_path = resolve_data_path("vector_index")

        if faiss is None:
            raise RuntimeError(
                "faiss is not installed. Install faiss-cpu or use Pinecone provider/in-memory fallback."
            )

        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = f"{index_path}_metadata.pkl"
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.entry_metadata: Dict[int, KnowledgeEntry] = {}
        self.id_to_faiss_id: Dict[str, int] = {}
        self.next_faiss_id = 0
        # Track insertion order for LRU eviction (faiss_id -> insertion_order)
        self._insertion_order: OrderedDict[int, int] = OrderedDict()
        self._insertion_counter = 0

        # Load existing index if available
        self._load_index()
    
    def _load_index(self) -> None:
        """Load existing FAISS index and metadata from disk."""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.entry_metadata = data.get('entry_metadata', {})
                    self.id_to_faiss_id = data.get('id_to_faiss_id', {})
                    self.next_faiss_id = data.get('next_faiss_id', 0)
                
                logger.info(f"Loaded vector store with {self.index.ntotal} entries")
            else:
                logger.info("No existing vector store found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            # Reset to empty state on error
            self.index = faiss.IndexFlatIP(self.dimension)
            self.entry_metadata = {}
            self.id_to_faiss_id = {}
            self.next_faiss_id = 0
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            metadata = {
                'entry_metadata': self.entry_metadata,
                'id_to_faiss_id': self.id_to_faiss_id,
                'next_faiss_id': self.next_faiss_id
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.debug(f"Saved vector store with {self.index.ntotal} entries")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def _evict_oldest_entries_if_needed(self) -> None:
        """Evict oldest entries if at capacity to prevent unbounded growth."""
        while self.index.ntotal >= MAX_VECTOR_STORE_ENTRIES and self._insertion_order:
            # Get oldest entry (first in OrderedDict)
            oldest_faiss_id, _ = self._insertion_order.popitem(last=False)
            # Remove from index (FAISS doesn't support single deletion, rebuild without it)
            self._rebuild_without_entry(oldest_faiss_id)
            logger.debug(f"Evicted oldest entry {oldest_faiss_id} to maintain size limit")

    def _rebuild_without_entry(self, exclude_faiss_id: int) -> None:
        """Rebuild index excluding a specific faiss_id."""
        entries_to_keep = []
        for fid, entry in self.entry_metadata.items():
            if fid != exclude_faiss_id:
                embedding = self.get_embedding(entry.entry_id)
                if embedding:
                    entries_to_keep.append((entry, embedding))

        # Clear old metadata references
        entry_id_to_remove = None
        for eid, fid in list(self.id_to_faiss_id.items()):
            if fid == exclude_faiss_id:
                entry_id_to_remove = eid
                break
        if entry_id_to_remove:
            del self.id_to_faiss_id[entry_id_to_remove]
        del self.entry_metadata[exclude_faiss_id]

        # Rebuild index
        self._rebuild_from_entries(entries_to_keep, persist=False)

    def _insert_entry_without_persist(self, entry: KnowledgeEntry, embedding: List[float]) -> None:
        """Insert a single entry into FAISS/index metadata without persisting to disk."""
        # Enforce size limit before adding
        self._evict_oldest_entries_if_needed()

        embedding_array = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(embedding_array)

        faiss_id = self.next_faiss_id
        self.index.add(embedding_array)

        self.entry_metadata[faiss_id] = entry
        self.id_to_faiss_id[entry.entry_id] = faiss_id
        self._insertion_order[faiss_id] = self._insertion_counter
        self._insertion_counter += 1
        self.next_faiss_id += 1
        entry.embedding = embedding

    def _rebuild_from_entries(
        self,
        entries_with_embeddings: List[Tuple[KnowledgeEntry, List[float]]],
        persist: bool = True,
    ) -> None:
        """Rebuild the FAISS index from entries and embeddings in a single pass."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.entry_metadata = {}
        self.id_to_faiss_id = {}
        self.next_faiss_id = 0
        # Reset insertion tracking
        self._insertion_order = OrderedDict()
        self._insertion_counter = 0

        for entry, embedding in entries_with_embeddings:
            if not embedding:
                continue
            self._insert_entry_without_persist(entry, embedding)

        if persist:
            self._save_index()

    def add_entry(self, entry: KnowledgeEntry, embedding: List[float], persist: bool = True) -> None:
        """
        Add a knowledge entry with its embedding to the vector store.
        
        Args:
            entry: The knowledge entry to add
            embedding: The embedding vector for the entry
        """
        try:
            self._insert_entry_without_persist(entry, embedding)

            if persist:
                self._save_index()
            
            logger.debug(f"Added entry {entry.entry_id} to vector store")
        except Exception as e:
            logger.error(f"Failed to add entry to vector store: {e}")
            raise
    
    def update_entry(self, entry: KnowledgeEntry, embedding: List[float], persist: bool = True) -> None:
        """
        Update an existing entry in the vector store.
        
        Args:
            entry: The updated knowledge entry
            embedding: The new embedding vector
        """
        try:
            if entry.entry_id in self.id_to_faiss_id:
                self.remove_entry(entry.entry_id, persist=False)
            
            self.add_entry(entry, embedding, persist=False)

            if persist:
                self._save_index()
            
            logger.debug(f"Updated entry {entry.entry_id} in vector store")
        except Exception as e:
            logger.error(f"Failed to update entry in vector store: {e}")
            raise
    
    def remove_entry(self, entry_id: str, persist: bool = True) -> bool:
        """
        Remove an entry from the vector store.
        
        Args:
            entry_id: ID of the entry to remove
            
        Returns:
            True if entry was removed, False if not found
        """
        try:
            if entry_id not in self.id_to_faiss_id:
                return False

            ids_to_remove = {entry_id}
            remaining_entries: List[Tuple[KnowledgeEntry, List[float]]] = []
            for entry in self.entry_metadata.values():
                if entry.entry_id in ids_to_remove:
                    continue
                if entry.embedding:
                    remaining_entries.append((entry, entry.embedding))

            self._rebuild_from_entries(remaining_entries, persist=persist)
            
            logger.debug(f"Removed entry {entry_id} from vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to remove entry from vector store: {e}")
            return False

    def remove_entries(self, entry_ids: List[str], persist: bool = True) -> int:
        """
        Remove multiple entries from the vector store in a single rebuild pass.

        Args:
            entry_ids: Entry IDs to remove
            persist: Whether to persist index to disk after rebuild

        Returns:
            Number of entries removed
        """
        try:
            ids_to_remove = {entry_id for entry_id in entry_ids if entry_id in self.id_to_faiss_id}
            if not ids_to_remove:
                return 0

            remaining_entries: List[Tuple[KnowledgeEntry, List[float]]] = []
            for entry in self.entry_metadata.values():
                if entry.entry_id in ids_to_remove:
                    continue
                if entry.embedding:
                    remaining_entries.append((entry, entry.embedding))

            self._rebuild_from_entries(remaining_entries, persist=persist)
            logger.debug("Removed %d entries from vector store", len(ids_to_remove))
            return len(ids_to_remove)
        except Exception as e:
            logger.error(f"Failed to remove entries from vector store: {e}")
            return 0
    
    def search(self, query_embedding: List[float], k: int = 10, 
               similarity_threshold: float = 0.7) -> List[KnowledgeSearchResult]:
        """
        Search for similar entries in the vector store.
        
        Args:
            query_embedding: The query embedding vector
            k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of search results with similarity scores
        """
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query embedding
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # No more results
                    break
                
                similarity_score = float(score)
                if similarity_score >= similarity_threshold:
                    entry = self.entry_metadata.get(idx)
                    if entry:
                        results.append(KnowledgeSearchResult(
                            entry=entry,
                            similarity_score=similarity_score
                        ))
            
            logger.debug(f"Vector search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            return []
    
    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """
        Get a specific entry by ID.
        
        Args:
            entry_id: ID of the entry to retrieve
            
        Returns:
            The knowledge entry if found, None otherwise
        """
        try:
            if entry_id in self.id_to_faiss_id:
                faiss_id = self.id_to_faiss_id[entry_id]
                return self.entry_metadata.get(faiss_id)
            return None
        except Exception as e:
            logger.error(f"Failed to get entry from vector store: {e}")
            return None
    
    def get_embedding(self, entry_id: str) -> Optional[List[float]]:
        """
        Get the embedding for a specific entry.
        
        Args:
            entry_id: ID of the entry to get embedding for
            
        Returns:
            The embedding vector if found, None otherwise
        """
        try:
            entry = self.get_entry(entry_id)
            if entry and hasattr(entry, 'embedding') and entry.embedding:
                return entry.embedding
            return None
        except Exception as e:
            logger.error(f"Failed to get embedding from vector store: {e}")
            return None
    
    def get_all_embeddings(self) -> Dict[str, List[float]]:
        """
        Get all embeddings in the vector store.
        
        Returns:
            Dictionary mapping entry IDs to their embeddings
        """
        try:
            embeddings = {}
            for entry_id, faiss_id in self.id_to_faiss_id.items():
                entry = self.entry_metadata.get(faiss_id)
                if entry and hasattr(entry, 'embedding') and entry.embedding:
                    embeddings[entry_id] = entry.embedding
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get all embeddings from vector store: {e}")
            return {}
    
    def get_all_entries(self) -> List[KnowledgeEntry]:
        """
        Get all entries in the vector store.
        
        Returns:
            List of all knowledge entries
        """
        try:
            return list(self.entry_metadata.values())
        except Exception as e:
            logger.error(f"Failed to get all entries from vector store: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        try:
            return {
                'total_entries': self.index.ntotal,
                'dimension': self.dimension,
                'index_size_mb': os.path.getsize(self.index_path) / (1024 * 1024) if os.path.exists(self.index_path) else 0,
                'last_updated': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return {}
    
    def clear(self) -> None:
        """Clear all entries from the vector store."""
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.entry_metadata = {}
            self.id_to_faiss_id = {}
            self.next_faiss_id = 0
            self._save_index()
            logger.info("Cleared vector store")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            raise


class InMemoryVectorStore:
    """Numpy-based local vector store fallback used when FAISS is unavailable."""

    def __init__(self, dimension: int = 1536, index_path: Optional[str] = None):
        if not index_path:
            index_path = resolve_data_path("vector_index")

        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = f"{index_path}_metadata.pkl"
        self.entry_metadata: Dict[str, KnowledgeEntry] = {}
        self.embeddings_by_entry_id: Dict[str, List[float]] = {}

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self._load_index()

    def _load_index(self) -> None:
        try:
            if not os.path.exists(self.metadata_path):
                return

            with open(self.metadata_path, "rb") as metadata_file:
                payload = pickle.load(metadata_file)

            raw_entries = payload.get("entry_metadata", {}) if isinstance(payload, dict) else {}
            raw_embeddings = payload.get("embeddings_by_entry_id", {}) if isinstance(payload, dict) else {}

            normalized_entries: Dict[str, KnowledgeEntry] = {}
            if isinstance(raw_entries, dict):
                for _, raw_entry in raw_entries.items():
                    if not isinstance(raw_entry, KnowledgeEntry):
                        continue
                    normalized_entries[raw_entry.entry_id] = raw_entry

            normalized_embeddings: Dict[str, List[float]] = {}
            if isinstance(raw_embeddings, dict):
                for entry_id, embedding in raw_embeddings.items():
                    if not isinstance(embedding, list):
                        continue
                    normalized_embeddings[str(entry_id)] = embedding

            # Hydrate vector map from entry payloads when loading legacy FAISS metadata.
            for entry_id, entry in normalized_entries.items():
                if entry_id in normalized_embeddings:
                    continue
                if not entry.embedding:
                    continue
                normalized_embeddings[entry_id] = self._fit_and_normalize_embedding(entry.embedding)

            self.entry_metadata = normalized_entries
            self.embeddings_by_entry_id = normalized_embeddings
        except Exception as error:
            logger.warning("Failed to load in-memory vector store metadata: %s", error)
            self.entry_metadata = {}
            self.embeddings_by_entry_id = {}

    def _save_index(self) -> None:
        payload = {
            "entry_metadata": self.entry_metadata,
            "embeddings_by_entry_id": self.embeddings_by_entry_id,
        }
        with open(self.metadata_path, "wb") as metadata_file:
            pickle.dump(payload, metadata_file)

    def _fit_and_normalize_embedding(self, embedding: List[float]) -> List[float]:
        if len(embedding) > self.dimension:
            fitted = list(embedding[: self.dimension])
        elif len(embedding) < self.dimension:
            fitted = list(embedding) + [0.0] * (self.dimension - len(embedding))
        else:
            fitted = list(embedding)

        embedding_array = np.array(fitted, dtype=np.float32)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm

        return embedding_array.tolist()

    def add_entry(self, entry: KnowledgeEntry, embedding: List[float], persist: bool = True) -> None:
        normalized_embedding = self._fit_and_normalize_embedding(embedding)
        entry.embedding = list(embedding)
        self.entry_metadata[entry.entry_id] = entry
        self.embeddings_by_entry_id[entry.entry_id] = normalized_embedding

        if persist:
            self._save_index()

    def update_entry(self, entry: KnowledgeEntry, embedding: List[float], persist: bool = True) -> None:
        self.add_entry(entry, embedding, persist=persist)

    def remove_entry(self, entry_id: str, persist: bool = True) -> bool:
        removed = False
        if entry_id in self.entry_metadata:
            self.entry_metadata.pop(entry_id, None)
            removed = True
        if entry_id in self.embeddings_by_entry_id:
            self.embeddings_by_entry_id.pop(entry_id, None)
            removed = True

        if removed and persist:
            self._save_index()

        return removed

    def remove_entries(self, entry_ids: List[str], persist: bool = True) -> int:
        removed = 0
        for entry_id in entry_ids:
            if self.remove_entry(entry_id, persist=False):
                removed += 1

        if removed and persist:
            self._save_index()

        return removed

    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[KnowledgeSearchResult]:
        if not self.embeddings_by_entry_id:
            return []

        normalized_query = np.array(
            self._fit_and_normalize_embedding(query_embedding),
            dtype=np.float32,
        )

        scored: List[Tuple[float, KnowledgeEntry]] = []
        for entry_id, normalized_embedding in self.embeddings_by_entry_id.items():
            entry = self.entry_metadata.get(entry_id)
            if not entry:
                continue

            score = float(np.dot(normalized_query, np.array(normalized_embedding, dtype=np.float32)))
            if score >= similarity_threshold:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            KnowledgeSearchResult(entry=entry, similarity_score=score)
            for score, entry in scored[: max(1, int(k))]
        ]

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        return self.entry_metadata.get(entry_id)

    def get_embedding(self, entry_id: str) -> Optional[List[float]]:
        entry = self.entry_metadata.get(entry_id)
        if entry and entry.embedding:
            return list(entry.embedding)

        normalized = self.embeddings_by_entry_id.get(entry_id)
        if not normalized:
            return None

        return list(normalized)

    def get_all_embeddings(self) -> Dict[str, List[float]]:
        payload: Dict[str, List[float]] = {}
        for entry_id in self.entry_metadata:
            embedding = self.get_embedding(entry_id)
            if embedding:
                payload[entry_id] = embedding

        return payload

    def get_all_entries(self) -> List[KnowledgeEntry]:
        return list(self.entry_metadata.values())

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self.entry_metadata),
            "dimension": self.dimension,
            "index_size_mb": os.path.getsize(self.metadata_path) / (1024 * 1024)
            if os.path.exists(self.metadata_path)
            else 0,
            "last_updated": datetime.utcnow().isoformat(),
            "provider": "in-memory",
        }

    def clear(self) -> None:
        self.entry_metadata = {}
        self.embeddings_by_entry_id = {}
        self._save_index()


# Per-user vector store instances
_vector_stores_by_user: Dict[str, Any] = {}


def _resolve_index_path_for_user(user_id: str) -> str:
    if user_id == "single_user":
        return resolve_data_path("vector_index")

    return resolve_data_path("users", user_id, "vector_index")


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_store_provider() -> str:
    configured = (os.getenv("VECTOR_STORE_PROVIDER") or "faiss").strip().lower()
    if configured in {"pinecone", "pinecone-serverless", "pinecone_serverless"}:
        return "pinecone"

    return "faiss"


def _build_local_vector_store(index_path: str) -> Any:
    try:
        return VectorStore(index_path=index_path)
    except Exception as local_error:
        logger.warning(
            "FAISS vector store unavailable at %s, using in-memory fallback: %s",
            index_path,
            local_error,
        )
        return InMemoryVectorStore(index_path=index_path)


def _build_vector_store_for_user(resolved_user_id: str) -> Any:
    index_path = _resolve_index_path_for_user(resolved_user_id)
    provider = _resolve_store_provider()

    if provider == "pinecone":
        try:
            from .pinecone_vector_store import PineconeVectorStore

            pinecone_store = PineconeVectorStore(
                user_id=resolved_user_id,
                dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
                local_metadata_path=f"{index_path}_pinecone_metadata.pkl",
            )
            logger.info("Using Pinecone vector store for user %s", resolved_user_id)

            if _parse_bool_env("PINECONE_BACKFILL_FROM_FAISS_ON_BOOT", False):
                try:
                    faiss_store = _build_local_vector_store(index_path)
                    seeded = 0
                    for entry in faiss_store.get_all_entries():
                        if not entry.embedding:
                            continue
                        pinecone_store.add_entry(entry, entry.embedding, persist=False)
                        seeded += 1

                    if seeded > 0:
                        logger.info(
                            "Backfilled %d vectors from FAISS into Pinecone for user %s",
                            seeded,
                            resolved_user_id,
                        )
                except Exception as backfill_error:
                    logger.warning(
                        "Pinecone backfill from FAISS failed for user %s: %s",
                        resolved_user_id,
                        backfill_error,
                    )

            return pinecone_store
        except Exception as pinecone_error:
            logger.warning(
                "Pinecone vector store unavailable for user %s, falling back to FAISS: %s",
                resolved_user_id,
                pinecone_error,
            )

    return _build_local_vector_store(index_path)


def get_vector_store(user_id: Optional[str] = None) -> Any:
    """Get a user-scoped vector store instance."""
    from app.auth.user_context import get_current_user_id, normalize_user_storage_key

    resolved_user_id = normalize_user_storage_key(user_id or get_current_user_id())
    if resolved_user_id not in _vector_stores_by_user:
        _vector_stores_by_user[resolved_user_id] = _build_vector_store_for_user(resolved_user_id)

    return _vector_stores_by_user[resolved_user_id]


def reset_vector_store(user_id: Optional[str] = None) -> Any:
    """Force reload a user-scoped vector store from persisted index files."""
    from app.auth.user_context import get_current_user_id, normalize_user_storage_key

    resolved_user_id = normalize_user_storage_key(user_id or get_current_user_id())
    _vector_stores_by_user.pop(resolved_user_id, None)
    _vector_stores_by_user[resolved_user_id] = _build_vector_store_for_user(resolved_user_id)

    return _vector_stores_by_user[resolved_user_id]