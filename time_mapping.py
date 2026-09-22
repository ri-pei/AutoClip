"""Shared, frame-verifiable time maps for Step 4, XML export and rendering.

Points are [local edited frame offset, source time in milliseconds].  The final
point is at the exclusive clip end.  Linear interpolation followed by floor
sampling must reproduce the selected source frames exactly.
"""

import json

import numpy as np


def source_frame_ends(times):
    times = np.asarray(times, dtype=float)
    if (len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Source frame timestamps must be finite and strictly increasing")
    return np.r_[times[1:], times[-1] + np.median(np.diff(times))]


def sample_source_frames(times, targets):
    """Floor sample, rejecting extrapolation rather than clipping invalid indices."""
    ends = source_frame_ends(times)
    targets = np.asarray(targets, dtype=float)
    ids = np.searchsorted(times, targets + 1e-8, side="right") - 1
    if (not np.isfinite(targets).all() or np.any(ids < 0)
            or np.any(targets >= ends[-1])):
        raise ValueError("Time map samples outside source media")
    return ids


def source_frame_samples(times, targets, sampling="floor"):
    left = sample_source_frames(times, targets)
    right = np.minimum(left + 1, len(times) - 1)
    weights = np.zeros(len(left), dtype=float)
    if sampling == "frame-blending":
        duration = source_frame_ends(times)[left] - np.asarray(times)[left]
        weights = np.clip((np.asarray(targets) - np.asarray(times)[left]) / duration, 0, 1)
        weights[right == left] = 0
    elif sampling != "floor":
        raise ValueError(f"Unsupported frame sampling: {sampling}")
    return left, right, weights


def parse_time_map(value, frame_count):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    points = np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.isfinite(points).all()
            or points[0, 0] != 0 or points[-1, 0] != frame_count
            or np.any(np.diff(points[:, 0]) <= 0)
            or np.any(points[:, 0] != np.floor(points[:, 0]))
            or np.any(np.diff(points[:, 1]) < -1e-6) or np.any(points[:, 1] < 0)):
        raise ValueError("Invalid time_map: require ordered frame offsets, full clip coverage and forward source times")
    return points


def segment_source_times(row, edited_times):
    """Sample a per-frame map or the explicit affine model used in ablations."""
    start, end = int(row["edited_start_frame"]), int(row["edited_end_frame"]) + 1
    count = end - start
    points = parse_time_map(row.get("time_map"), count)
    if points is not None:
        return np.interp(np.arange(count), points[:, 0], points[:, 1]) / 1000
    x = np.asarray(edited_times, dtype=float)[start:end]
    return float(row["source_start_time_ms"]) / 1000 + float(row["speed"]) * (x - x[0])


def compress_frame_map(ids, source_times, preferred_times):
    """Greedily extend lines inside each selected source frame's time interval.

    A slope corridor preserves discrete frame identities, unlike fitting a curve
    with a fixed millisecond tolerance.  This works with mixed source frame rates
    and timestamp quantization.  Knots stay inside a clip, not as extra cuts.
    """
    ids = np.asarray(ids, dtype=int)
    times = np.asarray(source_times, dtype=float)
    preferred = np.asarray(preferred_times, dtype=float)
    ends = source_frame_ends(times)
    if (not len(ids) or preferred.shape != ids.shape or np.any(ids < 0)
            or np.any(ids >= len(times)) or np.any(np.diff(ids) < 0)):
        raise ValueError("Cannot encode an invalid or backwards frame path")
    # Stay away from source-frame boundaries to tolerate rational serialization
    # and a container's rounded timestamps.  Units are seconds, not frames.
    margin = np.minimum((ends[ids] - times[ids]) * .05, .001)
    lower, upper = times[ids] + margin, ends[ids] - margin
    preferred = np.clip(preferred, lower, upper)
    count = len(ids)

    # Preserve a constant rate with two knots whenever it explains every frame.
    if count > 1:
        slope = (preferred[-1] - preferred[0]) / (count - 1)
        line = preferred[0] + slope * np.arange(count)
        terminal = preferred[0] + slope * count
        if (slope >= 0 and terminal <= ends[-1]
                and np.all(line >= lower) and np.all(line <= upper)):
            return [[0, float(preferred[0] * 1000)], [count, float(terminal * 1000)]]

    points = [[0, float(preferred[0])]]
    begin = 0
    while begin < count - 1:
        anchor = points[-1][1]
        low_slope, high_slope = 0., np.inf
        end = begin
        for index in range(begin + 1, count):
            distance = index - begin
            lo = max(low_slope, (lower[index] - anchor) / distance)
            hi = min(high_slope, (upper[index] - anchor) / distance)
            if lo > hi:
                break
            low_slope, high_slope, end = lo, hi, index
        if end == begin:
            raise ValueError("No forward time map through selected source frames")
        desired = (preferred[end] - anchor) / (end - begin)
        slope = np.clip(desired, low_slope, high_slope)
        points.append([end, float(anchor + slope * (end - begin))])
        begin = end
    # The exclusive endpoint does not sample a new frame.  Holding it avoids
    # extrapolating beyond the source, including a freeze at the very last frame.
    points.append([count, points[-1][1]])
    result = [[offset, value * 1000] for offset, value in points]
    parsed = np.asarray(result)
    sampled = sample_source_frames(times, np.interp(np.arange(count), parsed[:, 0], parsed[:, 1]) / 1000)
    if not np.array_equal(sampled, ids):
        raise AssertionError("Compressed time map changed source frame identities")
    return result


def compress_blended_map(values, source_times):
    """Simplify a continuous blend map, keeping pure frames exactly on their PTS.

    Mixed samples tolerate at most 0.2% of a source-frame interval.  An iterative
    line simplification avoids recursion limits on long clips with irregular
    cadence.  Its endpoints preserve monotonicity.
    """
    values = np.asarray(values, dtype=float)
    left, _, weights = source_frame_samples(source_times, values, "frame-blending")
    tolerance = (source_frame_ends(source_times)[left] - np.asarray(source_times)[left]) * .002
    tolerance[(weights < 1e-6) | (weights > 1 - 1e-6)] = 1e-10
    values = np.r_[values, values[-1]]
    tolerance = np.r_[tolerance, 1e-10]
    count = len(values) - 1
    keep, pending = {0, count}, [(0, count)]
    while pending:
        begin, end = pending.pop()
        if end - begin < 2:
            continue
        line = np.linspace(values[begin], values[end], end - begin + 1)
        relative_error = np.abs(line - values[begin:end + 1]) / tolerance[begin:end + 1]
        local = int(relative_error.argmax())
        if relative_error[local] > 1:
            index = begin + local
            keep.add(index)
            pending.extend(((begin, index), (index, end)))
    return [[index, float(values[index] * 1000)] for index in sorted(keep)]
