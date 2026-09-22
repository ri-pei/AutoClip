"""Load and validate Autoclip TOML settings without introducing config classes."""

from __future__ import annotations

import os
import math
import tomllib
from pathlib import Path


ALLOWED_KEYS = {
    "paths": {"working_dir", "edited_video", "source_dir", "output_dir"},
    "step1": {
        "reference_original",
        "reference_edited",
        "mask_rect",
        "color_lut",
        "frame_storage",
    },
    "step3": {
        "coarse_match_csv",
        "matcher",
        "top_k",
    },
    "step4": {"final_segments_csv", "alignment", "frame_refinement",
              "refine_radius", "review_mae", "frame_sampling", "max_pixel_mae",
              "short_jump_review", "jump_max_duration_ms"},
    "step5": {
        "frame_rate",
        "event_name",
        "project_name",
        "include_edited_audio",
    },
}


DEFAULTS = {
    "paths": {
        "working_dir": ".",
        "output_dir": "output",
        "source_dir": "source",
    },
    "step1": {"frame_storage": "compact"},
    "step3": {
        "coarse_match_csv": "coarse_match_results2.csv",
        "matcher": "numpy",
        "top_k": 20,
    },
    "step4": {"final_segments_csv": "final_video_segments_refined.csv", "alignment": "affine_pixels",
              "frame_refinement": True, "refine_radius": 4, "review_mae": 12.0,
              "frame_sampling": "auto", "max_pixel_mae": 18.0,
              "short_jump_review": True, "jump_max_duration_ms": 100.0},
    "step5": {
        "frame_rate": 0,
        "event_name": "Final Video Segments",
        "project_name": "My Project",
        "include_edited_audio": True,
    },
}


def _require_type(value, expected_type, label):
    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} 必须是数字。")
        return float(value)
    if not isinstance(value, expected_type):
        raise ValueError(f"{label} 的类型必须是 {expected_type.__name__}。")
    return value


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_settings(config_path: str | os.PathLike[str]) -> dict:
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.is_file():
        raise FileNotFoundError(
            f"配置文件不存在: {config_file}\n"
            "请复制 config.example.toml 为 config.toml 后再修改。"
        )
    with config_file.open("rb") as handle:
        raw = tomllib.load(handle)

    unknown_sections = set(raw) - set(ALLOWED_KEYS)
    if unknown_sections:
        raise ValueError(f"未知配置分区: {sorted(unknown_sections)}")
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"[{section}] 必须是 TOML 表。")
        unknown_keys = set(values) - ALLOWED_KEYS[section]
        if unknown_keys:
            raise ValueError(f"[{section}] 中存在未知配置: {sorted(unknown_keys)}")

    merged = {
        section: {**defaults, **raw.get(section, {})}
        for section, defaults in DEFAULTS.items()
    }
    paths = merged["paths"]
    if "edited_video" not in paths or not paths["edited_video"]:
        raise ValueError("[paths] edited_video 是必填项。")
    for key in ("working_dir", "edited_video", "source_dir", "output_dir"):
        _require_type(paths[key], str, f"[paths] {key}")

    config_base = config_file.parent
    working_dir_value = Path(os.path.expanduser(paths["working_dir"]))
    if not working_dir_value.is_absolute():
        working_dir_value = config_base / working_dir_value
    working_dir = working_dir_value.resolve()

    step1 = merged["step1"]
    for key in ("reference_original", "reference_edited", "color_lut"):
        if key in step1:
            _require_type(step1[key], str, f"[step1] {key}")
    mask_rect = step1.get("mask_rect")
    if mask_rect is not None:
        if (
            not isinstance(mask_rect, list)
            or len(mask_rect) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in mask_rect)
        ):
            raise ValueError("[step1] mask_rect 必须是四个整数 [x, y, w, h]。")
        mask_rect = tuple(mask_rect)

    step3 = merged["step3"]
    _require_type(step3["coarse_match_csv"], str, "[step3] coarse_match_csv")

    step4 = merged["step4"]
    _require_type(step4["final_segments_csv"], str, "[step4] final_segments_csv")

    if step1["frame_storage"] != "compact":
        raise ValueError("Only compact frame storage is supported; use review sheets and the final renderer")
    if step3["matcher"] not in ("balltree_batch", "numpy", "faiss", "faiss_hnsw"):
        raise ValueError("Unknown matcher")
    _require_type(step3["top_k"], int, "[step3] top_k")
    if isinstance(step3["top_k"], bool) or not 1 <= step3["top_k"] <= 100:
        raise ValueError("top_k must be between 1 and 100")
    if step4["alignment"] not in ("affine", "affine_ransac", "affine_pixels"):
        raise ValueError("Unknown alignment")
    _require_type(step4["frame_refinement"], bool, "[step4] frame_refinement")
    _require_type(step4["refine_radius"], int, "[step4] refine_radius")
    if isinstance(step4["refine_radius"], bool) or not 1 <= step4["refine_radius"] <= 12:
        raise ValueError("refine_radius must be between 1 and 12 source frames")
    review_mae = _require_type(step4["review_mae"], float, "[step4] review_mae")
    if not math.isfinite(review_mae) or not 0 < review_mae <= 255:
        raise ValueError("review_mae must be finite and between 0 and 255")
    if step4["frame_refinement"] and step4["alignment"] != "affine_pixels":
        raise ValueError("frame_refinement requires alignment=affine_pixels")
    _require_type(step4["short_jump_review"], bool, "[step4] short_jump_review")
    if step4["short_jump_review"] and not step4["frame_refinement"]:
        raise ValueError("short_jump_review requires frame_refinement=true")
    jump_duration = _require_type(step4["jump_max_duration_ms"], float, "[step4] jump_max_duration_ms")
    if not math.isfinite(jump_duration) or not 0 < jump_duration <= 1000:
        raise ValueError("jump_max_duration_ms must be finite and in (0, 1000]")
    if step4["frame_sampling"] not in ("auto", "floor", "frame-blending"):
        raise ValueError("frame_sampling must be auto, floor or frame-blending")
    max_pixel_mae = _require_type(step4["max_pixel_mae"], float, "[step4] max_pixel_mae")
    if not math.isfinite(max_pixel_mae) or not 0 < max_pixel_mae <= 255:
        raise ValueError("max_pixel_mae must be finite and between 0 and 255")

    step5 = merged["step5"]
    frame_rate = _require_type(step5["frame_rate"], float, "[step5] frame_rate")
    if not math.isfinite(frame_rate) or frame_rate < 0:
        raise ValueError("[step5] frame_rate 不能小于 0。")
    for key in ("event_name", "project_name"):
        _require_type(step5[key], str, f"[step5] {key}")
    _require_type(step5["include_edited_audio"], bool, "[step5] include_edited_audio")

    return {
        "config_path": str(config_file),
        "frame_storage": step1["frame_storage"],
        "matcher": step3["matcher"],
        "top_k": step3["top_k"],
        "alignment": step4["alignment"],
        "frame_refinement": step4["frame_refinement"],
        "refine_radius": step4["refine_radius"],
        "review_mae": review_mae,
        "frame_sampling": step4["frame_sampling"],
        "max_pixel_mae": max_pixel_mae,
        "short_jump_review": step4["short_jump_review"],
        "jump_max_duration_ms": jump_duration,
        "working_dir": str(working_dir),
        "edited_video_path": _resolve_path(working_dir, paths["edited_video"]),
        "source_dir": _resolve_path(working_dir, paths["source_dir"]),
        "output_dir": _resolve_path(working_dir, paths["output_dir"]),
        "reference_original_path": _resolve_path(
            working_dir, step1.get("reference_original")
        ),
        "reference_edited_path": _resolve_path(
            working_dir, step1.get("reference_edited")
        ),
        "mask_rect": mask_rect,
        "color_lut_path": _resolve_path(working_dir, step1.get("color_lut")),
        "coarse_match_csv": step3["coarse_match_csv"],
        "final_segments_csv": step4["final_segments_csv"],
        "frame_rate": frame_rate,
        "event_name": step5["event_name"],
        "project_name": step5["project_name"],
        "include_edited_audio": step5["include_edited_audio"],
    }
