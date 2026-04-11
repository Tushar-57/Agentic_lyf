#!/usr/bin/env python3
"""Backfill Pinecone vectors from local FAISS index for a user namespace."""

import argparse
import os
import sys
from pathlib import Path

# Ensure backend package imports resolve when script is run directly.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.pinecone_vector_store import PineconeVectorStore
from app.services.vector_store import InMemoryVectorStore, VectorStore
from app.services.storage_paths import resolve_data_path


def resolve_index_path_for_user(user_id: str) -> str:
    if user_id == "single_user":
        return resolve_data_path("vector_index")

    return resolve_data_path("users", user_id, "vector_index")


def has_signal(embedding: list[float] | None) -> bool:
    if not embedding:
        return False

    return any(abs(float(value)) > 1e-8 for value in embedding)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Pinecone vectors from local FAISS index")
    parser.add_argument("--user-id", default="single_user", help="User storage key to migrate")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max entries to backfill (0 means all)",
    )
    args = parser.parse_args()

    user_id = args.user_id.strip() or "single_user"
    index_path = resolve_index_path_for_user(user_id)

    if not (os.getenv("PINECONE_API_KEY") or "").strip():
        print("PINECONE_API_KEY is missing. Export it first and rerun.")
        return 1

    try:
        source = VectorStore(index_path=index_path)
    except Exception:
        source = InMemoryVectorStore(index_path=index_path)
    target = PineconeVectorStore(
        user_id=user_id,
        dimension=source.dimension,
        local_metadata_path=f"{index_path}_pinecone_metadata.pkl",
    )

    all_entries = source.get_all_entries()
    if args.limit > 0:
        all_entries = all_entries[: args.limit]

    backfilled = 0
    skipped = 0

    for entry in all_entries:
        if not has_signal(entry.embedding):
            skipped += 1
            continue

        target.add_entry(entry, entry.embedding, persist=False)
        backfilled += 1

    if backfilled > 0:
        target.persist_metadata_cache()

    print(f"User: {user_id}")
    print(f"Source entries scanned: {len(all_entries)}")
    print(f"Backfilled: {backfilled}")
    print(f"Skipped (no embedding): {skipped}")
    print(f"Pinecone namespace: {target.namespace}")
    print(f"Pinecone index: {target.index_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
