"""Step 4: temporal alignment, per-frame refinement and short-jump review."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import discover_source_videos


def load_coarse_matches_with_timestamps(filepath):
    """Load explicit frame IDs and PTS; obsolete filename-based CSVs are rejected."""
    df = pd.read_csv(filepath, dtype={"edited_phash": str})
    required = {"edited_frame_number", "edited_timestamp_ms", "edited_phash", "top_n_matches"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing coarse-match columns: {sorted(required - set(df.columns))}; rerun Step 3")
    df["top_n_matches"] = df["top_n_matches"].apply(json.loads)
    df = df.sort_values("edited_frame_number").reset_index(drop=True)
    if (not np.array_equal(df.edited_frame_number, np.arange(len(df)))
            or not np.isfinite(df.edited_timestamp_ms).all()
            or np.any(np.diff(df.edited_timestamp_ms) <= 0)):
        raise ValueError("Coarse matches must contain contiguous frame IDs and increasing timestamps")
    return df


def load_all_original_frames_data(phash_csvs_base_dir, edited_video_name_no_ext, allowed_sources):
    """Only read active sources, never recursively include backups or stale caches."""
    frames = []
    for name in sorted(set(allowed_sources) - {edited_video_name_no_ext}):
        path = Path(phash_csvs_base_dir) / name / f"{name}_phash.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing active source cache; rerun Steps 1–3: {path}")
        df = pd.read_csv(path, dtype={"phash": str})
        df = df.rename(columns={"frame_number": "original_frame_number",
                                "timestamp_ms": "original_timestamp_ms",
                                "phash": "original_phash"})
        df["original_video_name"] = name
        frames.append(df[["original_video_name", "original_frame_number",
                          "original_timestamp_ms", "original_phash"]])
    if not frames:
        raise ValueError("No active source frame data")
    return pd.concat(frames, ignore_index=True)


def main_step4(settings):
    """Write initial alignment, local diagnostics and the final editable segments."""
    ABS_OUTPUT_DIR = settings["output_dir"]
    edited_video_name_no_ext_param = Path(settings["edited_video_path"]).stem
    FINAL_SEGMENTS_CSV_FILENAME = settings["final_segments_csv"]
    df_processed_frames = load_coarse_matches_with_timestamps(
        Path(ABS_OUTPUT_DIR) / settings["coarse_match_csv"])
    if df_processed_frames.empty:
        raise RuntimeError("No coarse match data loaded")
    allowed_sources = {Path(path).stem for path in discover_source_videos(
        settings["source_dir"], settings["edited_video_path"])}
    df_all_original_frames = load_all_original_frames_data(
        ABS_OUTPUT_DIR, edited_video_name_no_ext_param, allowed_sources)
    from temporal_alignment import align_segments
    result = align_segments(df_processed_frames, df_all_original_frames,
                            method=settings["alignment"], output_dir=ABS_OUTPUT_DIR,
                            edited_name=edited_video_name_no_ext_param,
                            max_pixel_mae=settings.get("max_pixel_mae", 18.))
    if result.empty:
        raise RuntimeError("No reliable matched segments")
    destination = Path(ABS_OUTPUT_DIR) / FINAL_SEGMENTS_CSV_FILENAME
    if settings.get("frame_refinement", False):
        from frame_refinement import refine_segments, save_review_sheet
        result.to_csv(destination.with_suffix(".affine.csv"), index=False)
        result, details, report = refine_segments(
            result, df_processed_frames, ABS_OUTPUT_DIR, edited_video_name_no_ext_param,
            radius=settings["refine_radius"], review_mae=settings["review_mae"],
            sampling=settings["frame_sampling"])
        if settings.get("short_jump_review", False):
            from short_jump_review import review_short_jumps
            result.to_csv(destination.with_suffix(".pre_jump.csv"), index=False)
            details.to_csv(destination.with_suffix(".pre_jump.frames.csv"), index=False)
            destination.with_suffix(".pre_jump.audit.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            result, details, report, jump_audit = review_short_jumps(
                result, details, report, ABS_OUTPUT_DIR, edited_video_name_no_ext_param,
                radius=settings["refine_radius"], review_mae=settings["review_mae"],
                sampling=settings["frame_sampling"], max_duration_ms=settings["jump_max_duration_ms"])
            destination.with_suffix(".jump_review.json").write_text(
                json.dumps(jump_audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Short-jump review: {jump_audit['reviewed_clips']} checked, "
                  f"{jump_audit['repaired_frames']} frames bridged; {jump_audit['clips_after']} clips remain")
        details.to_csv(destination.with_suffix(".frames.csv"), index=False)
        save_review_sheet(details, ABS_OUTPUT_DIR, edited_video_name_no_ext_param,
                          destination.with_suffix(".review.jpg"))
        destination.with_suffix(".audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        print(f"Local frame verification: {report['adjusted_frames']} adjusted, "
              f"status counts {report['status_counts']}")
        print(f"Frame diagnostics: {destination.with_suffix('.frames.csv')}")
    result.to_csv(destination, index=False)


if __name__ == "__main__":
    from settings import load_settings
    main_step4(load_settings("config.toml"))
