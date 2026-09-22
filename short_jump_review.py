"""Step 4: compare short excursions with an anchored continuous alternative.

No new retrieval model or LLM is required. Only already assigned short clips are
reviewed; gaps, media endpoints and unrelated flanking sources are never filled.
The before/after diagnostics remain separate from the editable time maps.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frame_refinement import _summary, low_information_frames, refine_sampling_path
from time_mapping import source_frame_samples


def _positions(details, times):
    ids = details.original_frame_number.to_numpy(int)
    weights = details.blend_weight.to_numpy(float)
    if np.any(ids < 0) or np.any(ids >= len(times)):
        raise ValueError("Review source frame outside media")
    return times[ids] + weights * (times[np.minimum(ids + 1, len(times) - 1)] - times[ids])


def _frame_records(details):
    columns = ["edited_frame_number", "original_video_name", "original_frame_number",
               "blend_weight", "pixel_mae", "status"]
    return json.loads(details[columns].to_json(orient="records", double_precision=12))


def _merge_reviewed_rows(left, right, details, times, review_mae):
    """Join maps without refitting either neighbour or changing its frame samples."""
    start, end = int(left["edited_start_frame"]), int(right["edited_end_frame"]) + 1
    chosen = details.iloc[start:end]
    values = _positions(chosen, times)
    if np.any(np.diff(values) < -1e-9):
        raise ValueError("Reviewed bridge would run backwards")
    mode = "frame-blending" if (chosen.blend_weight > 1e-8).any() else "floor"
    # Keep one knot per frame in a repaired range. This deliberately avoids
    # resimplifying the surrounding maps and moving already verified samples.
    points = [[i, float(t * 1000)] for i, t in enumerate(values)]
    points.append([len(values), points[-1][1]])
    sampled = np.interp(np.arange(len(values)), np.asarray(points)[:, 0], np.asarray(points)[:, 1]) / 1000
    ids, _, weights = source_frame_samples(times, sampled, mode)
    np.testing.assert_array_equal(ids, chosen.original_frame_number.to_numpy(int))
    np.testing.assert_allclose(weights, chosen.blend_weight, atol=1e-8, rtol=0)
    row = dict(left)
    duration = (chosen.edited_timestamp_ms.iloc[-1] - chosen.edited_timestamp_ms.iloc[0]) / 1000
    row.update(edited_end_frame=end - 1, edited_end_time_ms=right["edited_end_time_ms"],
               original_start_frame=int(ids[0]), original_end_frame=int(ids[-1]),
               original_start_time_ms=float(times[ids[0]] * 1000),
               original_end_time_ms=float(times[ids[-1]] * 1000),
               source_start_time_ms=points[0][1], source_end_time_ms=points[-1][1],
               speed=float((values[-1] - values[0]) / duration) if duration else 0.,
               time_map=json.dumps(points, separators=(",", ":")), frame_sampling=mode,
               pixel_mae=float(chosen.pixel_mae.mean()),
               pixel_p95=float(np.percentile(chosen.pixel_mae, 95)),
               review_frames=int((chosen.pixel_mae > review_mae).sum()),
               confidence=max(0., 1 - float(chosen.pixel_mae.mean()) / 64))
    return row


def review_short_jumps(segments, details, report, output_dir, edited_name,
                       radius=4, review_mae=12., sampling="auto", max_duration_ms=100.):
    """Return clips, frame diagnostics, updated audit and explicit review decisions.

    Acceptance: reliable flanks from one source, a plausible forward bridge,
    low-information majority OR a clear visual improvement, and bounded error
    on every proposed frame. A repaired frame stays ambiguous about provenance.
    """
    if not np.isfinite(max_duration_ms) or not 0 < max_duration_ms <= 1000:
        raise ValueError("max_duration_ms must be finite and in (0, 1000]")
    result = details.copy(deep=True).reset_index(drop=True)
    if not np.array_equal(result.edited_frame_number, np.arange(len(result))):
        raise ValueError("Short-jump review requires one ordered row per target frame")
    target_times = result.edited_timestamp_ms.to_numpy(float) / 1000
    if len(target_times) < 2 or not np.isfinite(target_times).all() or np.any(np.diff(target_times) <= 0):
        raise ValueError("Short-jump review requires increasing target timestamps")
    target_step = float(np.median(np.diff(target_times)))
    rows = segments.sort_values("edited_start_frame").to_dict("records")
    result["affine_original_video_name"] = result.get("affine_original_video_name", result.original_video_name)
    result["jump_review_id"] = -1
    result["jump_review_action"] = ""
    root = Path(output_dir)
    query = np.load(root / edited_name / "compact_frames.npy", mmap_mode="r", allow_pickle=False)
    if len(query) != len(result):
        raise ValueError("Review target cache and diagnostics disagree")
    sources = {}

    def source_data(name):
        if name not in sources:
            folder = root / name
            pixels = np.load(folder / "compact_frames.npy", mmap_mode="r", allow_pickle=False)
            times = np.asarray(json.loads((folder / "compact_frames.json").read_text())["timestamps_ms"], dtype=float) / 1000
            if len(pixels) != len(times) or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f"Invalid review source cache: {name}")
            sources[name] = pixels, times
        return sources[name]

    decisions = []
    index = 1
    while index < len(rows) - 1:
        left, middle, right = rows[index - 1:index + 2]
        start, end = int(middle["edited_start_frame"]), int(middle["edited_end_frame"]) + 1
        duration = target_times[end] - target_times[start]
        if duration * 1000 > max_duration_ms + 1e-6:
            index += 1
            continue
        event = dict(id=len(decisions), start_frame=start, end_frame=end - 1,
                     duration_ms=float(duration * 1000), action="retained",
                     reason="", before=_frame_records(result.iloc[start:end]))
        decisions.append(event)
        result.loc[start:end - 1, "jump_review_id"] = event["id"]
        result.loc[start:end - 1, "jump_review_action"] = "retained"

        def retain(reason):
            event["reason"] = reason
            if reason in ("unreliable_flanking_frames", "insufficient_evidence_to_change_a_detailed_cut"):
                event["needs_review"] = True
                result.loc[start:end - 1, "status"] = np.where(
                    result.status.iloc[start:end] == "review", "review", "ambiguous")
                result.loc[start:end - 1, "jump_review_action"] = "needs_review"

        if int(left["edited_end_frame"]) != start - 1 or int(right["edited_start_frame"]) != end:
            retain("non_adjacent_target_ranges")
        elif left["original_video_name"] != right["original_video_name"]:
            retain("different_flanking_sources")
        elif any(result.iloc[i].status != "verified" for i in (start - 1, end)):
            retain("unreliable_flanking_frames")
        else:
            name = left["original_video_name"]
            pixels, times = source_data(name)
            dt = float(np.median(np.diff(times)))
            anchors = _positions(result.iloc[[start - 1, end]], times)
            elapsed = target_times[end] - target_times[start - 1]
            advance = anchors[1] - anchors[0]
            predicted = np.interp(target_times[start:end], target_times[[start - 1, end]], anchors)
            same_source = middle["original_video_name"] == name
            excursion = (not same_source or
                         np.min(np.abs(_positions(result.iloc[start:end], times) - predicted)) > max(.25, (radius + 1) * dt))
            if advance < -1e-9 or advance > 4 * elapsed + dt:
                retain("incompatible_flanking_times")
            elif not excursion:
                retain("not_a_remote_excursion")
            else:
                # Context admits blends using the existing support test, but only
                # the island may change. Original flanking samples remain locked.
                context = max(3, int(round(.5 / target_step)))
                begin = max(int(left["edited_start_frame"]), start - context)
                finish = min(int(right["edited_end_frame"]) + 1, end + context)
                prior = np.r_[_positions(result.iloc[begin:start], times), predicted,
                              _positions(result.iloc[end:finish], times)]
                ids, weights, losses, _, margins, mode = refine_sampling_path(
                    query[begin:finish], pixels, times, prior, radius, sampling)
                sl = slice(start - begin, end - begin)
                ids, weights, losses, margins = ids[sl], weights[sl], losses[sl], margins[sl]
                proposed_times = times[ids] + weights * (times[np.minimum(ids + 1, len(times) - 1)] - times[ids])
                old_loss = result.pixel_mae.iloc[start:end].to_numpy(float)
                flat = low_information_frames(query[start:end])
                clear_gain = old_loss.mean() - losses.mean() > 1 and losses.mean() < old_loss.mean() * .8
                event.update(source=name, anchor_source_times_ms=(anchors * 1000).tolist(),
                             old_mean_mae=float(old_loss.mean()), proposed_mean_mae=float(losses.mean()),
                             maximum_frame_mae_increase=float(np.max(losses - old_loss)),
                             low_information_fraction=float(flat.mean()),
                             proposed_source_frames=ids.tolist(), proposed_blend_weights=weights.tolist(),
                             proposed_frame_mae=losses.tolist())
                if proposed_times[0] < anchors[0] - 1e-9 or proposed_times[-1] > anchors[1] + 1e-9:
                    retain("candidate_crosses_locked_anchors")
                elif (not np.isfinite(old_loss).all() or not np.isfinite(losses).all()
                      or losses.max() > review_mae or losses.mean() > old_loss.mean() + 1.
                      or np.max(losses - old_loss) > 2.):
                    retain("continuous_picture_not_equivalent")
                elif flat.mean() < .5 and not clear_gain:
                    retain("insufficient_evidence_to_change_a_detailed_cut")
                else:
                    result.loc[start:end - 1, "original_video_name"] = name
                    result.loc[start:end - 1, "original_frame_number"] = ids
                    result.loc[start:end - 1, "blend_weight"] = weights
                    result.loc[start:end - 1, "pixel_mae"] = losses
                    result.loc[start:end - 1, "alternative_margin"] = np.nan if mode == "frame-blending" else margins
                    result.loc[start:end - 1, "low_information"] = flat
                    result.loc[start:end - 1, "status"] = "ambiguous"
                    result.loc[start:end - 1, "frame_adjustment"] = np.where(
                        result.affine_original_video_name.iloc[start:end] == name,
                        ids - result.affine_source_frame.iloc[start:end], np.nan)
                    result.loc[start:end - 1, "jump_review_action"] = "bridged"
                    merged = _merge_reviewed_rows(left, right, result, times, review_mae)
                    rows[index - 1:index + 2] = [merged]
                    event.update(action="bridged", reason="continuous_alternative_with_weak_or_worse_jump_evidence",
                                 after=_frame_records(result.iloc[start:end]))
                    index = max(1, index - 1)
                    continue
        index += 1

    updated_report = deepcopy(report)
    clips = []
    for clip_index, row in enumerate(rows):
        start, end = int(row["edited_start_frame"]), int(row["edited_end_frame"]) + 1
        result.loc[start:end - 1, "segment_index"] = clip_index
        result.loc[start:end - 1, "frame_sampling"] = row["frame_sampling"]
        chunk = result.iloc[start:end]
        adjusted = ((chunk.original_frame_number != chunk.affine_source_frame)
                    | (chunk.original_video_name != chunk.affine_original_video_name))
        clips.append(dict(segment_index=clip_index, source=row["original_video_name"],
                          start_frame=start, end_frame=end - 1, map_points=len(json.loads(row["time_map"])),
                          before=_summary(chunk.affine_pixel_mae), after=_summary(chunk.pixel_mae),
                          adjusted_frames=int(adjusted.sum()), sampling=row["frame_sampling"],
                          blended_frames=int((chunk.blend_weight > .05).sum()),
                          review_frames=int((chunk.status == "review").sum()),
                          ambiguous_frames=int((chunk.status == "ambiguous").sum())))
    audit = dict(version=1, decisions=decisions,
                 policy=dict(max_duration_ms=max_duration_ms, mean_mae_slack=1., frame_mae_slack=2.,
                             maximum_mae=review_mae, minimum_low_information_fraction=.5,
                             minimum_jump_seconds=.25, maximum_bridge_speed=4., context_seconds=.5),
                 reviewed_clips=len(decisions), repaired_clips=sum(d["action"] == "bridged" for d in decisions),
                 unresolved_clips=sum(d.get("needs_review", False) for d in decisions),
                 repaired_frames=int((result.jump_review_action == "bridged").sum()),
                 clips_before=len(segments), clips_after=len(rows))
    updated_report.update(clips=clips, status_counts={str(k): int(v) for k, v in result.status.value_counts().items()},
                          adjusted_frames=sum(c["adjusted_frames"] for c in clips),
                          blended_frames=int((result.blend_weight > .05).sum()),
                          pixel_error_after=_summary(result.pixel_mae),
                          short_jump_review={k: v for k, v in audit.items() if k not in ("decisions", "policy")})
    return pd.DataFrame(rows), result, updated_report, audit
