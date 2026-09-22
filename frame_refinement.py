"""Step 4's optional local frame verification, with inspectable diagnostics.

First keep the source and cut decisions from affine alignment.  Then find a
monotone path through a narrow band of neighbouring source frames.  Pixel error
supplies evidence; weak penalties retain the affine cadence in ambiguous shots.
No filenames, clip positions or known frame mappings enter the algorithm.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from time_mapping import (compress_blended_map, compress_frame_map, sample_source_frames,
                          segment_source_times, source_frame_samples)


def low_information_frames(frames):
    """Detect almost uniform content without counting common letterbox borders."""
    h, w = frames.shape[1:]
    content = frames[:, h // 8:h - h // 8, w // 8:w - w // 8]
    contrast = np.percentile(content, [10, 90], axis=(1, 2))
    return contrast[1] - contrast[0] < 5


def _monotone_path(candidates, losses, base):
    costs = losses + .20 * np.abs(candidates - base[:, None])
    history = np.empty(candidates.shape, dtype=np.int16)
    previous = costs[0].copy()
    states = np.arange(candidates.shape[1])
    for i in range(1, len(base)):
        steps = candidates[i, None, :] - candidates[i - 1, :, None]
        transition = .25 * np.abs(steps - (base[i] - base[i - 1]))
        transition[steps < -1e-8] = np.inf
        total = previous[:, None] + transition
        history[i] = total.argmin(axis=0)
        previous = costs[i] + total[history[i], states]
    chosen = np.empty(len(base), dtype=int)
    chosen[-1] = previous.argmin()
    for i in range(len(base) - 1, 0, -1):
        chosen[i - 1] = history[i, chosen[i]]
    return chosen


def refine_frame_path(query, source, source_times, predicted_times, radius=4):
    """Banded dynamic programming: O(query frames * (2*radius+1)^2)."""
    base = sample_source_frames(source_times, predicted_times)
    offsets = np.arange(-radius, radius + 1)
    candidates = (base[:, None] + offsets).clip(0, len(source) - 1)
    losses = np.empty(candidates.shape, dtype=np.float32)
    # Bound the expanded 64x64 pixel buffer even for a long episode.
    for begin in range(0, len(base), 64):
        end = begin + 64
        pixels = np.asarray(source[candidates[begin:end]], dtype=np.float32)
        q = np.asarray(query[begin:end], dtype=np.float32)
        losses[begin:end] = np.abs(pixels - q[:, None]).mean(axis=(2, 3))
    # These are small grayscale-level costs, not probabilities.  A different
    # frame must improve the picture to overcome the prior and cadence penalty.
    chosen = _monotone_path(candidates, losses, base)
    rows = np.arange(len(base))
    selected = candidates[rows, chosen]
    selected_loss = losses[rows, chosen]
    alternatives = np.where(candidates == selected[:, None], np.inf, losses)
    margin = alternatives.min(axis=1) - selected_loss
    return selected, selected_loss, losses[:, radius], margin


def refine_sampling_path(query, source, source_times, predicted_times, radius=4, sampling="auto"):
    """Detect adjacent-frame blends only when supported by a clear visual gain.

    Auto requires at least three strong frames and 5% support in a window of up
    to 60 target frames. A long static tail must not hide a short mixed-frame
    interval. Each admitted sample must beat the best unmixed candidate by >1
    grayscale level and >20%. Pure-frame states remain available throughout.
    """
    hard, hard_error, base_error, margin = refine_frame_path(query, source, source_times, predicted_times, radius)
    fallback = (hard, np.zeros(len(hard)), hard_error, base_error, margin, "floor")
    if sampling == "floor":
        return fallback
    if sampling not in ("auto", "frame-blending"):
        raise ValueError(f"Unknown sampling mode: {sampling}")
    base = sample_source_frames(source_times, predicted_times)
    candidates = (base[:, None] + np.arange(-radius, radius + 1)).clip(0, len(source) - 1)
    hard_losses = np.empty(candidates.shape, dtype=np.float32)
    mixed_losses = np.empty_like(hard_losses)
    alpha = np.empty_like(hard_losses)
    for begin in range(0, len(base), 64):
        end = begin + 64
        ids = candidates[begin:end]
        left = np.asarray(source[ids], dtype=np.float32)
        delta = np.asarray(source[np.minimum(ids + 1, len(source)-1)], dtype=np.float32) - left
        q = np.asarray(query[begin:end], dtype=np.float32)[:, None]
        weight = np.clip(((q-left)*delta).sum(axis=(2, 3)) /
                         np.maximum(np.square(delta).sum(axis=(2, 3)), 1e-8), 0, 1)
        alpha[begin:end] = weight
        hard_losses[begin:end] = np.abs(q-left).mean(axis=(2, 3))
        mixed_losses[begin:end] = np.abs(q-left-weight[:, :, None, None]*delta).mean(axis=(2, 3))
    best_hard = hard_losses.min(axis=1, keepdims=True)
    strong = ((best_hard - mixed_losses > 1.) & (mixed_losses < best_hard * .8)
              & (alpha > .05) & (alpha < .95))
    if sampling == "auto":
        window = min(60, len(base))
        support = np.convolve(strong.any(axis=1).astype(int), np.ones(window, dtype=int), mode="valid")
        if support.max() < max(3, window * .05):
            return fallback
    mixed_losses[~strong] = np.inf
    positions = np.c_[candidates, candidates + alpha]
    losses = np.c_[hard_losses, mixed_losses]
    chosen = _monotone_path(positions, losses, base)
    selected = positions[np.arange(len(base)), chosen]
    ids = np.floor(selected + 1e-8).astype(int)
    weights = selected - ids
    if not np.any(weights > .05):
        return fallback
    error = losses[np.arange(len(base)), chosen]
    return ids, weights, error, base_error, margin, "frame-blending"


def _summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), p99=float(np.percentile(values, 99)),
                maximum=float(values.max()))


def save_review_sheet(details, output_dir, edited_name, destination, count=10):
    """Show the largest remaining errors without decoding the full videos again."""
    from PIL import Image, ImageDraw
    chosen = details[details.original_frame_number >= 0].nlargest(count, "pixel_mae")
    if chosen.empty:
        return
    root = Path(output_dir)
    query = np.load(root / edited_name / "compact_frames.npy", mmap_mode="r", allow_pickle=False)
    sources = {}
    width, height, label = 256, 144, 24
    canvas = Image.new("RGB", (width * 3, (height + label) * len(chosen)), "#202020")
    draw = ImageDraw.Draw(canvas)
    for position, row in enumerate(chosen.itertuples()):
        name = row.original_video_name
        affine_name = getattr(row, "affine_original_video_name", name)
        for source_name in (name, affine_name):
            if source_name not in sources:
                sources[source_name] = np.load(root / source_name / "compact_frames.npy", mmap_mode="r", allow_pickle=False)
        refined = sources[name][row.original_frame_number].astype(float)
        weight = getattr(row, "blend_weight", 0.)
        if weight:
            next_frame = min(row.original_frame_number + 1, len(sources[name]) - 1)
            refined = refined * (1-weight) + sources[name][next_frame] * weight
        images = [query[row.edited_frame_number], sources[affine_name][row.affine_source_frame],
                  np.rint(refined).astype(np.uint8)]
        titles = [f"Target #{row.edited_frame_number}",
                  f"Affine #{row.affine_source_frame} MAE {row.affine_pixel_mae:.2f}",
                  f"Refined #{row.original_frame_number} MAE {row.pixel_mae:.2f}"]
        y = position * (height + label)
        for column, (picture, title) in enumerate(zip(images, titles)):
            draw.text((column * width + 4, y + 4), title, fill="white")
            image = Image.fromarray(picture).resize((width, height)).convert("RGB")
            canvas.paste(image, (column * width, y + label))
    canvas.save(destination)


def refine_segments(segments, edited, output_dir, edited_name, radius=4, review_mae=12., sampling="auto"):
    """Return editable clips, one diagnostic row per target frame, and an audit."""
    root = Path(output_dir)
    frame_numbers = edited["edited_frame_number"].to_numpy(int)
    if not np.array_equal(frame_numbers, np.arange(len(edited))):
        raise ValueError("Local refinement requires every target frame; rerun Steps 1–3")
    times = edited["edited_timestamp_ms"].to_numpy(float) / 1000
    query = np.load(root / edited_name / "compact_frames.npy", mmap_mode="r", allow_pickle=False)
    if len(query) != len(edited):
        raise ValueError("Target compact frames and coarse matches disagree")
    details = pd.DataFrame(dict(edited_frame_number=frame_numbers,
                                edited_timestamp_ms=times * 1000,
                                segment_index=-1, original_video_name="",
                                affine_source_frame=-1, original_frame_number=-1,
                                affine_pixel_mae=np.nan, pixel_mae=np.nan,
                                alternative_margin=np.nan, frame_adjustment=0,
                                blend_weight=0., frame_sampling="floor",
                                status="unmatched", low_information=False))
    result, clip_reports = [], []
    source_cache = {}
    for clip_index, row in enumerate(segments.to_dict("records")):
        name = row["original_video_name"]
        if name not in source_cache:
            folder = root / name
            manifest = json.loads((folder / "compact_frames.json").read_text())
            source = np.load(folder / "compact_frames.npy", mmap_mode="r", allow_pickle=False)
            source_times = np.asarray(manifest["timestamps_ms"], dtype=float) / 1000
            if len(source) != len(source_times):
                raise ValueError(f"Compact source timestamps disagree: {name}")
            source_cache[name] = source, source_times
        source, source_times = source_cache[name]
        start, end = int(row["edited_start_frame"]), int(row["edited_end_frame"]) + 1
        predicted = segment_source_times(row, times)
        base = sample_source_frames(source_times, predicted)
        selected, weights, loss, base_loss, margin, frame_sampling = refine_sampling_path(
            query[start:end], source, source_times, predicted, radius, sampling
        )
        if frame_sampling == "frame-blending":
            next_frames = np.minimum(selected + 1, len(source_times) - 1)
            values = source_times[selected] + weights * (source_times[next_frames] - source_times[selected])
            points = compress_blended_map(values, source_times)
            # Audit the actual exported curve, including simplification error.
            sampled_times = np.interp(np.arange(end-start), np.asarray(points)[:, 0], np.asarray(points)[:, 1]) / 1000
            selected, next_frames, weights = source_frame_samples(source_times, sampled_times, frame_sampling)
            for begin in range(0, len(selected), 64):
                sl = slice(begin, begin + 64)
                w = weights[sl, None, None]
                reconstructed = np.asarray(source[selected[sl]], dtype=float) * (1-w) + source[next_frames[sl]] * w
                loss[sl] = np.abs(reconstructed - query[start:end][sl]).mean(axis=(1, 2))
        else:
            points = compress_frame_map(selected, source_times, predicted)
        # The time_map is authoritative.  speed remains the nominal affine rate
        # for inspection and is not used to flatten a variable map at export.
        row.update(time_map=json.dumps(points, separators=(",", ":")), frame_sampling=frame_sampling,
                   original_start_frame=int(selected[0]), original_end_frame=int(selected[-1]),
                   original_start_time_ms=float(source_times[selected[0]] * 1000),
                   original_end_time_ms=float(source_times[selected[-1]] * 1000),
                   source_start_time_ms=points[0][1], source_end_time_ms=points[-1][1],
                   pixel_mae=float(loss.mean()), pixel_p95=float(np.percentile(loss, 95)),
                   review_frames=int(np.sum(loss > review_mae)))
        result.append(row)
        # Percentile contrast ignores a small black watermark mask on otherwise
        # uniform frames.  Such frames can match well without identifying a time.
        flat = low_information_frames(query[start:end])
        if frame_sampling == "frame-blending":
            # The returned margin describes the unmixed candidate path, not the
            # blended state.  Leave it unspecified instead of mislabelling it.
            margin = np.full(len(selected), np.nan)
        status = np.where(loss > review_mae, "review", np.where(flat, "ambiguous", "verified"))
        positions = np.arange(start, end)
        details.loc[positions, "segment_index"] = clip_index
        details.loc[positions, "original_video_name"] = name
        for key, value in (("affine_source_frame", base), ("original_frame_number", selected),
                           ("affine_pixel_mae", base_loss), ("pixel_mae", loss),
                           ("alternative_margin", margin), ("frame_adjustment", selected - base),
                           ("blend_weight", weights), ("frame_sampling", frame_sampling),
                           ("status", status), ("low_information", flat)):
            details.loc[positions, key] = value
        clip_reports.append(dict(segment_index=clip_index, source=name,
                                 start_frame=start, end_frame=end - 1,
                                 map_points=len(points), before=_summary(base_loss), after=_summary(loss),
                                 adjusted_frames=int(np.sum(selected != base)),
                                 sampling=frame_sampling, blended_frames=int(np.sum(weights > .05)),
                                 review_frames=int(np.sum(loss > review_mae))))
    assigned = details.original_frame_number >= 0
    report = dict(total_frames=len(details), assigned_frames=int(assigned.sum()),
                  assigned_fraction=float(assigned.mean()),
                  status_counts={str(k): int(v) for k, v in details.status.value_counts().items()},
                  adjusted_frames=int((details.frame_adjustment != 0).sum()),
                  blended_frames=int((details.blend_weight > .05).sum()),
                  pixel_error_before=_summary(details.affine_pixel_mae),
                  pixel_error_after=_summary(details.pixel_mae),
                  radius_source_frames=radius, review_mae=review_mae, clips=clip_reports,
                  metric_note="64x64 preprocessed grayscale MAE (0..255); assignment is not ground-truth accuracy. Low-information frames do not uniquely identify source time.")
    return pd.DataFrame(result), details, report
