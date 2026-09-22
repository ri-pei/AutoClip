"""Step 3: batched pHash search over the currently selected source videos."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import discover_source_videos
from fast_matching import pack_hashes, search_hashes


def load_phash_data_from_csv(csv_path):
    """Reject missing or malformed caches instead of silently dropping frames."""
    df = pd.read_csv(csv_path, dtype={"phash": str}, keep_default_na=False)
    required = {"video_name", "frame_number", "timestamp_ms", "phash"}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError(f"Invalid or empty pHash cache: {csv_path}; rerun Steps 1–2")
    df = df.sort_values("frame_number").reset_index(drop=True)
    if (not np.array_equal(df.frame_number, np.arange(len(df)))
            or not np.isfinite(df.timestamp_ms).all()
            or np.any(np.diff(df.timestamp_ms) <= 0)):
        raise ValueError(f"Invalid frame IDs or timestamps in {csv_path}")
    pack_hashes(df.phash)  # Validate every 256-bit hash before matching.
    return df


def main_step3(settings):
    output = Path(settings["output_dir"])
    edited_name = Path(settings["edited_video_path"]).stem
    query = load_phash_data_from_csv(output / edited_name / f"{edited_name}_phash.csv")
    sources = []
    for path in discover_source_videos(settings["source_dir"], settings["edited_video_path"]):
        name = Path(path).stem
        data = load_phash_data_from_csv(output / name / f"{name}_phash.csv")
        data["video_name"] = name
        sources.append(data)
    source = pd.concat(sources, ignore_index=True)
    backend, k = settings["matcher"], settings["top_k"]
    print(f"Step 3: {len(query)} target frames, {len(source)} source frames, {backend}, top-{k}")
    distances, indices = search_hashes(pack_hashes(source.phash), pack_hashes(query.phash), k, backend)
    originals = source.to_dict("records")
    results = []
    for row, ds, ids in zip(query.to_dict("records"), distances, indices):
        candidates = []
        for distance, index in zip(ds, ids):
            original = originals[int(index)]
            candidates.append({
                "original_video_name": original["video_name"],
                "original_frame_number": int(original["frame_number"]),
                "original_timestamp_ms": float(original["timestamp_ms"]),
                "original_phash": original["phash"],
                "phash_distance": int(distance),
            })
        results.append({
            "edited_video_name": row["video_name"],
            "edited_frame_number": int(row["frame_number"]),
            "edited_timestamp_ms": float(row["timestamp_ms"]),
            "edited_phash": row["phash"],
            "top_n_matches": json.dumps(candidates),
        })
    destination = output / settings["coarse_match_csv"]
    pd.DataFrame(results).to_csv(destination, index=False, lineterminator="\n")
    print(f"Coarse matches: {destination}")


if __name__ == "__main__":
    from settings import load_settings
    main_step3(load_settings("config.toml"))
