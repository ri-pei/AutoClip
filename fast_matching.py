"""Batched Hamming search with exact backends and optional approximate search."""

import numpy as np
from sklearn.neighbors import BallTree


def pack_hashes(values):
    rows = []
    for value in values:
        raw = bytes.fromhex(str(value))
        if len(raw) != 32:
            raise ValueError("Expected a 256-bit pHash (64 hex characters)")
        rows.append(np.frombuffer(raw, dtype=np.uint8))
    return np.ascontiguousarray(rows, dtype=np.uint8).reshape(-1, 32)


def search_hashes(database, queries, k=3, backend="numpy", batch_size=32):
    """Return integer bit distances and row indices; bound temporary memory."""
    k = min(k, len(database))
    if k < 1 or batch_size < 1:
        raise ValueError("Nonempty database, positive k and batch_size required")
    if backend == "balltree_batch":
        tree = BallTree(np.unpackbits(database, axis=1), metric="hamming")
        d, i = tree.query(np.unpackbits(queries, axis=1), k=k)
        return np.rint(d * 256).astype(np.int16), i
    if backend in ("faiss", "faiss_hnsw"):
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError("This backend requires optional dependency faiss-cpu") from error
        if backend == "faiss":
            index = faiss.IndexBinaryFlat(256)
        else:
            index = faiss.IndexBinaryHNSW(256, 32)
            index.hnsw.efSearch = 128
        # Small binary searches otherwise suffer excessive OpenMP overhead.
        old_threads = faiss.omp_get_max_threads()
        try:
            faiss.omp_set_num_threads(1)
            index.add(database)
            return index.search(queries, k)
        finally:
            faiss.omp_set_num_threads(old_threads)
    if backend != "numpy":
        raise ValueError(f"Unknown search backend: {backend}")
    table = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    distances, indices = [], []
    # Database chunks bound memory even for movie-scale collections.
    for begin in range(0, len(queries), batch_size):
        q = queries[begin:begin + batch_size]
        best_d = np.empty((len(q), 0), dtype=np.int16)
        best_i = np.empty((len(q), 0), dtype=np.int64)
        for db_begin in range(0, len(database), 8192):
            db = database[db_begin:db_begin + 8192]
            xor = np.bitwise_xor(q[:, None, :], db[None, :, :])
            counts = np.bitwise_count(xor) if hasattr(np, "bitwise_count") else table[xor]
            d = counts.sum(axis=2, dtype=np.int16)
            ids = np.broadcast_to(np.arange(db_begin, db_begin + len(db)), d.shape)
            d = np.concatenate((best_d, d), axis=1)
            ids = np.concatenate((best_i, ids), axis=1)
            # Lexicographic order makes equal-distance results reproducible.
            order = np.argsort(d.astype(np.int64) * (len(database) + 1) + ids, axis=1)[:, :k]
            best_d = np.take_along_axis(d, order, axis=1)
            best_i = np.take_along_axis(ids, order, axis=1)
        distances.append(best_d)
        indices.append(best_i)
    if not distances:
        return np.empty((0, k), dtype=np.int16), np.empty((0, k), dtype=np.int64)
    return np.vstack(distances), np.vstack(indices)
