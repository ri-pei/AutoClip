"""Piecewise affine temporal alignment, independent of fixture ground truth.

Local consensus proposes speed/offset models. A switching-cost decoder chooses
consistent models; each resulting clip is refined against the original hashes.
All model coordinates are seconds, not frame numbers.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from fast_matching import pack_hashes


def _pixel_fallback_matches(query_pixels, source_pixels, query_indices, frames,
                            top_k=3, maximum_mae=12.0):
    """Find visually close compact frames when pHash is unstable.

    pHash is deliberately insensitive to luminance, but that makes it unstable on
    nearly uniform images such as fades to white.  A small exact L2 search on 16x16
    mean-pooled pixels supplies candidates for weak pHash retrieval or nearly
    uniform images. Full 64x64 MAE then rejects unrelated low-resolution neighbours.
    """
    query_indices = np.asarray(query_indices, dtype=np.int64)
    if not len(query_indices) or not len(source_pixels):
        return {}

    def descriptors(values):
        values = np.asarray(values, dtype=np.float32)
        return values.reshape(len(values), 16, 4, 16, 4).mean(axis=(2, 4)).reshape(len(values), -1)

    queries = np.ascontiguousarray(descriptors(query_pixels[query_indices]), dtype=np.float32)
    # Mean-pool in chunks: expanding an entire episode to float32 first would
    # allocate four times its compact-frame storage before the search starts.
    database = np.empty((len(source_pixels), 256), dtype=np.float32)
    for begin in range(0, len(source_pixels), 2048):
        database[begin:begin + 2048] = descriptors(source_pixels[begin:begin + 2048])
    search_k = min(max(top_k * 3, top_k), len(database))
    neighbours = np.empty((len(queries), search_k), dtype=np.int64)
    for query_begin in range(0, len(queries), 128):
        batch = queries[query_begin:query_begin + 128]
        query_norm = np.einsum("ij,ij->i", batch, batch)
        best_distances = np.empty((len(batch), 0), dtype=np.float32)
        best_indices = np.empty((len(batch), 0), dtype=np.int64)
        for source_begin in range(0, len(database), 8192):
            source = database[source_begin:source_begin + 8192]
            distances = (query_norm[:, None]
                         + np.einsum("ij,ij->i", source, source)[None, :]
                         - 2 * batch @ source.T)
            indices = np.broadcast_to(
                np.arange(source_begin, source_begin + len(source)), distances.shape
            )
            distances = np.concatenate((best_distances, distances), axis=1)
            indices = np.concatenate((best_indices, indices), axis=1)
            keep = min(search_k, distances.shape[1])
            partial = np.argpartition(distances, keep - 1, axis=1)[:, :keep]
            best_distances = np.take_along_axis(distances, partial, axis=1)
            best_indices = np.take_along_axis(indices, partial, axis=1)
        order = best_distances.argsort(axis=1, kind="stable")
        neighbours[query_begin:query_begin + len(batch)] = np.take_along_axis(
            best_indices, order, axis=1
        )

    result = {}
    for query_index, ids in zip(query_indices, neighbours):
        losses = np.mean(np.abs(np.asarray(source_pixels[ids], dtype=np.float32)
                                - np.asarray(query_pixels[query_index], dtype=np.float32)), axis=(1, 2))
        order = np.argsort(losses, kind="stable")
        matches = []
        for position in order:
            loss = float(losses[position])
            if loss > maximum_mae:
                continue
            source_index = int(ids[position])
            row = frames.iloc[source_index]
            matches.append(dict(
                original_video_name=row["original_video_name"],
                original_frame_number=int(row["original_frame_number"]),
                original_timestamp_ms=float(row["original_timestamp_ms"]),
                # Keep the existing temporal cost scale while recording why this
                # candidate was admitted.  Refinement still uses the actual pixels.
                evidence_cost=min(64.0, loss * 4.0),
                pixel_mae=loss,
            ))
            if len(matches) == top_k:
                break
        if matches:
            result[int(query_index)] = matches
    return result


def _hypotheses(x, y, distances, dt, randomized=False, window=24):
    rng = np.random.default_rng(20260922)
    n, k = y.shape
    models = []
    width = min(window, n)
    tolerance = max(dt * 2.5, .055)
    starts = sorted(set(range(0, max(1, n - width + 1), max(1, width // 3))) | {max(0, n-width)})
    for start in starts:
        xx, yy, dd = x[start:start + width], y[start:start + width], distances[start:start + width]
        if len(xx) < 4:
            continue
        if randomized:
            left = rng.integers(0, len(xx) // 2, 128)
            right = rng.integers(len(xx) // 2, len(xx), 128)
            kl = rng.integers(0, min(k, 5), 128)
            kr = rng.integers(0, min(k, 5), 128)
        else:
            pairs = [(0, len(xx) - 1), (len(xx) // 4, 3 * len(xx) // 4),
                     (min(3, len(xx) // 4), max(len(xx) - 4, 3 * len(xx) // 4))]
            combinations = [(a, b, i, j) for a, b in pairs
                            for i in range(min(k, 4)) for j in range(min(k, 4))]
            left, right, kl, kr = np.array(combinations).T
        slopes = (yy[right, kr] - yy[left, kl]) / (xx[right] - xx[left])
        intercepts = yy[left, kl] - slopes * xx[left]
        valid = np.isfinite(slopes) & (slopes >= -1e-6) & (slopes <= 4)
        slopes, intercepts = slopes[valid], intercepts[valid]
        slopes = np.maximum(slopes, 0)
        if not len(slopes):
            continue
        residual = np.abs(slopes[:, None, None] * xx[None, :, None]
                          + intercepts[:, None, None] - yy[None, :, :])
        allowed = dd <= np.minimum(64, dd.min(axis=1, keepdims=True) + 12)
        residual = np.where(allowed[None, :, :], residual, np.inf)
        best = residual.min(axis=2)
        score = (best < tolerance).sum(axis=1) - np.minimum(best / tolerance, 1).sum(axis=1) * .15
        for model_index in np.argsort(score)[-2:]:
            if (best[model_index] < tolerance).sum() < max(4, len(xx) * .55):
                continue
            a, b = slopes[model_index], intercepts[model_index]
            for _ in range(3):
                costs = np.abs(a * xx[:, None] + b - yy)
                costs = np.where(allowed, costs, np.inf)
                ids = costs.argmin(axis=1)
                good = costs[np.arange(len(xx)), ids] < tolerance
                if good.sum() < 4:
                    break
                a, b = np.polyfit(xx[good], yy[np.arange(len(xx)), ids][good], 1)
            if -1e-6 <= a <= 4:
                models.append((float(max(0, a)), float(b)))
    # Retain genuinely different local hypotheses without a huge state space.
    unique = {}
    for a, b in models:
        unique.setdefault((round(a, 2), round(b / dt)), (a, b))
    return list(unique.values())


def _decode(costs, switch_penalty=4.0):
    """O(frames * states) Viterbi with an equal inter-state switch cost."""
    previous = costs[0].copy()
    history = np.empty(costs.shape, dtype=np.int32)
    history[0] = np.arange(costs.shape[1])
    state_ids = np.arange(costs.shape[1])
    for i in range(1, len(costs)):
        best = previous.argmin()
        change = previous[best] + switch_penalty < previous
        history[i] = np.where(change, best, state_ids)
        previous = costs[i] + np.where(change, previous[best] + switch_penalty, previous)
    path = np.empty(len(costs), dtype=np.int32)
    path[-1] = previous.argmin()
    for i in range(len(costs) - 1, 0, -1):
        path[i - 1] = history[i, path[i]]
    return path


def _refine_mapping(x, query_hashes, times, source_hashes, a, b, query_pixels=None, source_pixels=None):
    """Optimize a floor-sampled linear time map using full-frame visual evidence."""
    dt = float(np.median(np.diff(times)))
    relative = x - x[0]
    duration = max(x[-1] - x[0], dt)
    # Hypotheses fit frame timestamps; floor sampling needs a subframe phase.
    center = a * x[0] + b + dt * .5
    span = max(.035, 2 * dt / duration)
    slopes = np.linspace(max(0, a - span), min(4, a + span), 65)
    slopes = np.unique(np.r_[slopes, np.linspace(0, 4, 161),
                            [float(Fraction(float(a)).limit_denominator(d)) for d in (4, 10, 20, 100)]])
    starts = center + np.arange(-12, 13) * dt / 4
    starts = np.unique(np.r_[starts, times[np.clip(np.searchsorted(times, center), 0, len(times)-1)]])
    best = (np.inf, a, center)
    table = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    trials = []
    for slope in slopes:
        predicted = starts[:, None] + slope * relative
        ids = np.searchsorted(times, predicted + 1e-8, side="right") - 1
        invalid = (ids < 0) | (ids >= len(times)) | (predicted > times[-1] + dt)
        ids = ids.clip(0, len(times)-1)
        xor = source_hashes[ids] ^ query_hashes[None, :, :]
        distances = table[xor].sum(axis=2).astype(float)
        distances[invalid] = 256
        scores = distances.mean(axis=1)
        if query_pixels is not None:
            for j in range(len(starts)):
                trials.append((float(scores[j]), float(slope), float(starts[j])))
        # Weak tie break favors the original consensus, not any known test rate.
        scores += 1e-5 * (np.abs(starts - center) / dt + abs(slope - a))
        index = scores.argmin()
        if scores[index] < best[0]:
            best = (float(scores[index]), float(slope), float(starts[index]))
    if query_pixels is not None:
        # Always retain the local temporal hypothesis for pixel scoring.  Its pHash
        # score can be arbitrarily bad on flat/faded frames even when the pixels are
        # an almost exact match.
        trials.append((float("inf"), float(max(0, a)), float(center)))
        for denominator in (4, 10, 20, 50, 100, 1000):
            slope = float(Fraction(best[1]).limit_denominator(denominator))
            ids = (np.searchsorted(times, starts[:,None]+slope*relative+1e-8, side="right")-1).clip(0,len(times)-1)
            scores = table[source_hashes[ids]^query_hashes[None,:,:]].sum(axis=2).mean(axis=1)
            trials.extend((float(score), slope, float(start)) for score, start in zip(scores, starts))
        sample = np.unique(np.linspace(0, len(relative)-1, min(256, len(relative))).astype(int))
        q = np.asarray(query_pixels[sample], dtype=np.float32)
        unique = {}
        for score, slope, start in trials:
            if np.isfinite(score) and score > best[0] + 2:
                continue
            if start < times[0] or start + slope * relative[-1] >= times[-1] + dt:
                continue
            ids = np.searchsorted(times, start + slope*relative[sample] + 1e-8, side="right")-1
            key = ids.tobytes()
            complexity = Fraction(slope).limit_denominator(1000).denominator
            rank = (complexity, score)
            if key not in unique or rank < unique[key][0]:
                unique[key] = (rank, score, slope, start, ids)
        ranked = sorted(unique.values(), key=lambda item: item[1])
        shortlist = ranked[:256]
        chosen_paths = {item[-1].tobytes() for item in shortlist}
        shortlist.extend(item for item in ranked
                         if not np.isfinite(item[1]) and item[-1].tobytes() not in chosen_paths)
        pixel_results = []
        for rank, score, slope, start, ids in shortlist:
            pixels = np.asarray(source_pixels[ids], dtype=np.float32)
            loss = float(np.mean(np.abs(pixels-q)))
            candidate = (round(loss, 6), rank[0], slope, start, score)
            pixel_results.append(candidate)
        if not pixel_results:
            return a, center, float("inf")
        # The small sample ranks trials only.  Score finalists on EVERY frame so
        # a short cut/error cannot disappear between the 256 sampled positions.
        finalists = []
        for _, complexity, slope, start, score in sorted(pixel_results)[:8]:
            total = 0.
            for begin in range(0, len(relative), 128):
                local = relative[begin:begin + 128]
                ids = np.searchsorted(times, start + slope * local + 1e-8, side="right") - 1
                pixels = np.asarray(source_pixels[ids], dtype=np.float32)
                queries = np.asarray(query_pixels[begin:begin + 128], dtype=np.float32)
                total += float(np.abs(pixels - queries).sum(dtype=np.float64))
            loss = total / (len(relative) * query_pixels.shape[1] * query_pixels.shape[2])
            finalists.append((loss, complexity, slope, start, score))
        minimum = min(item[0] for item in finalists)
        # Compression noise cannot identify subframe speed; prefer simpler rates
        # only within 0.005 grayscale levels of the minimum (range 0..255).
        pixel_best = min((item for item in finalists if item[0] <= minimum + .005),
                         key=lambda item: (item[1], item[0]))
        return pixel_best[2], pixel_best[3], pixel_best[0]
    # Prefer a simple rational only when it produces equally good visual evidence.
    for denominator in (4, 10, 20, 100):
        slope = float(Fraction(best[1]).limit_denominator(denominator))
        predicted = starts[:, None] + slope * relative
        ids = (np.searchsorted(times, predicted + 1e-8, side="right") - 1).clip(0, len(times)-1)
        scores = table[source_hashes[ids] ^ query_hashes[None, :, :]].sum(axis=2).mean(axis=1)
        index = scores.argmin()
        if scores[index] <= best[0] + 1e-4:
            return slope, float(starts[index]), float(scores[index])
    return best[1], best[2], best[0]


def _recover_short_gaps(rows, edited, candidates, source_data, query_pixels, source_pixels,
                        maximum_mae=12.):
    """Verify isolated 1–5 frame cutaways that cannot establish a long model.

    Only existing retrieval candidates are considered, and each recovered frame
    needs its own pixel evidence.  Long unknown intervals are never filled by
    extrapolating a neighbouring clip.
    """
    frame_numbers = edited["edited_frame_number"].to_numpy(int)
    x = edited["edited_timestamp_ms"].to_numpy(float) / 1000
    covered = np.zeros(len(edited), dtype=bool)
    for row in rows:
        covered |= ((frame_numbers >= row["edited_start_frame"])
                    & (frame_numbers <= row["edited_end_frame"]))
    edges = np.flatnonzero(np.diff(np.r_[False, ~covered, False]))
    indices = {name: {int(number): index for index, number in enumerate(frames.original_frame_number)}
               for name, (frames, _, _) in source_data.items()}
    step = float(np.median(np.diff(x)))
    for begin, end in zip(edges[::2], edges[1::2]):
        if end - begin > 5:
            continue
        matches = []
        for position in range(begin, end):
            best = None
            for match in candidates[position]:
                name = match["original_video_name"]
                source_index = indices.get(name, {}).get(int(match["original_frame_number"]))
                if source_index is None:
                    continue
                loss = float(np.abs(np.asarray(source_pixels[name][source_index], dtype=float)
                                    - query_pixels[position]).mean())
                if loss <= maximum_mae and (best is None or loss < best[0]):
                    best = (loss, name, source_index)
            if best is not None:
                matches.append((position, *best))
        # Each run has one source and a plausible forward traversal.  A source
        # switch, a jump, or missing visual evidence is a real boundary here.
        runs = []
        for match in matches:
            position, _, name, index = match
            times = source_data[name][1]
            if runs:
                previous = runs[-1][-1]
                contiguous = (position == previous[0] + 1 and name == previous[2]
                              and 0 <= times[index] - times[previous[3]]
                              <= 4 * (x[position] - x[previous[0]]))
                if contiguous:
                    runs[-1].append(match)
                    continue
            runs.append([match])
        for run in runs:
            first, last = run[0], run[-1]
            start, finish, name = first[0], last[0], first[2]
            frames, times, _ = source_data[name]
            dt = float(np.median(np.diff(times)))
            source_start = float(times[first[3]] + dt * .5)
            speed = 0. if start == finish else float((times[last[3]] - times[first[3]]) / (x[finish] - x[start]))
            length = x[finish] - x[start] + step
            rows.append(dict(
                edited_start_frame=int(frame_numbers[start]), edited_end_frame=int(frame_numbers[finish]),
                edited_start_time_ms=x[start] * 1000, edited_end_time_ms=x[finish] * 1000,
                original_video_name=name, original_start_frame=int(frames.iloc[first[3]].original_frame_number),
                original_end_frame=int(frames.iloc[last[3]].original_frame_number),
                original_start_time_ms=times[first[3]] * 1000, original_end_time_ms=times[last[3]] * 1000,
                source_start_time_ms=source_start * 1000,
                source_end_time_ms=min(source_start + speed * length, times[-1] + dt) * 1000,
                speed=speed, confidence=max(0., 1 - np.mean([item[1] for item in run]) / 64),
                edited_total_frames=int(frame_numbers[-1]) + 1))
    return sorted(rows, key=lambda row: row["edited_start_frame"])


def align_segments(edited, originals, method="affine", output_dir=None, edited_name=None,
                   max_pixel_mae=18.):
    edited = edited.sort_values("edited_frame_number").reset_index(drop=True)
    if len(edited) < 4:
        raise ValueError("Affine alignment requires at least four query frames")
    x = edited["edited_timestamp_ms"].to_numpy(float) / 1000
    query_hashes = pack_hashes(edited["edited_phash"])
    query_pixels, source_pixels = None, {}
    if method == "affine_pixels":
        if output_dir is None or edited_name is None:
            raise ValueError("Pixel refinement requires compact frame intermediates")
        path = Path(output_dir) / edited_name / "compact_frames.npy"
        if not path.exists():
            raise ValueError("affine_pixels requires frame_storage=compact")
        query_pixels = np.load(path, mmap_mode="r", allow_pickle=False)[edited["edited_frame_number"].to_numpy(int)]
    candidates = edited["top_n_matches"].tolist()
    original_best = np.array([
        min((match.get("phash_distance", 256) for match in matches), default=256)
        for matches in candidates
    ], dtype=float)
    weak_queries = original_best > 48
    flat_queries = np.zeros(len(edited), dtype=bool)
    if query_pixels is not None:
        from frame_refinement import low_information_frames
        # Solid black and solid white can have identical pHashes.  A low hash
        # distance is not enough evidence to skip brightness-aware retrieval.
        flat_queries = low_information_frames(query_pixels)
        weak_queries |= flat_queries
    weak_pixel_queries = np.flatnonzero(weak_queries)
    all_models, emissions, source_data = [], [], {}
    for name, frames in originals.groupby("original_video_name", sort=False):
        frames = frames.sort_values("original_timestamp_ms", kind="stable").reset_index(drop=True)
        times = frames["original_timestamp_ms"].to_numpy(float) / 1000
        if len(times) < 2:
            continue
        dt = np.median(np.diff(times))
        if dt <= 0:
            continue
        source_data[name] = (frames, times, pack_hashes(frames["original_phash"]))
        if query_pixels is not None:
            pixels = np.load(Path(output_dir)/name/"compact_frames.npy", mmap_mode="r", allow_pickle=False)
            ids = frames["original_frame_number"].to_numpy(int)
            source_pixels[name] = pixels if np.array_equal(ids, np.arange(len(pixels))) else pixels[ids]
            fallbacks = _pixel_fallback_matches(
                query_pixels, source_pixels[name], weak_pixel_queries, frames
            )
            for query_index, matches in fallbacks.items():
                # Pixel candidates lead because deterministic hypothesis generation
                # intentionally samples only the first few candidates per endpoint.
                # Do not deduplicate against pHash matches: the same source frame
                # may be present there with a misleadingly large pHash distance,
                # which is precisely the failure this fallback is meant to repair.
                candidates[query_index] = matches + candidates[query_index]
        k = max(1, max(map(len, candidates)))
        y = np.full((len(x), k), np.nan)
        distances = np.full((len(x), k), 256.)
        frame_indices = {int(number): index for index, number in enumerate(frames.original_frame_number)}
        for i, matches in enumerate(candidates):
            selected = [m for m in matches if m["original_video_name"] == name]
            for j, match in enumerate(selected):
                y[i, j] = match["original_timestamp_ms"] / 1000
                distances[i, j] = (match["evidence_cost"] if "evidence_cost" in match
                                   else match["phash_distance"])
                if query_pixels is not None and flat_queries[i]:
                    source_index = frame_indices.get(int(match["original_frame_number"]))
                    if source_index is not None:
                        loss = np.abs(np.asarray(source_pixels[name][source_index], dtype=np.float32)
                                      - np.asarray(query_pixels[i], dtype=np.float32)).mean()
                        # Hash collisions on flat images must not outrank the
                        # brightness-aware candidate with an apparent distance 0.
                        distances[i, j] = min(256., float(loss) * 4.)
        proposed = []
        for window in (24, 12, 6):
            proposed.extend(_hypotheses(x, y, distances, dt, method == "affine_ransac", window))
        # Windows can propose the same model repeatedly; do not duplicate whole
        # decoder columns for equivalent hypotheses.
        proposed = list(dict.fromkeys(proposed))
        for a, b in proposed:
            residual = np.abs(a * x[:, None] + b - y) / dt
            costs = distances / 16 + residual * .35
            costs[(distances > 64) | ~np.isfinite(costs)] = 20
            emissions.append(np.minimum(costs.min(axis=1), 20))
            all_models.append((name, a, b))
    if not all_models:
        return pd.DataFrame()
    # Explicit unknown state prevents forcing unrelated frames onto a source.
    emissions.append(np.full(len(x), 3.5))
    path = _decode(np.array(emissions).T)
    boundaries = np.r_[0, np.flatnonzero(np.diff(path)) + 1, len(path)]
    groups = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        model = path[start]
        if model == len(all_models):
            continue
        name, a, b = all_models[model]
        # Keep decoder boundaries until every local model has been refined.  Merely
        # being continuous at the boundary does not mean that two neighbouring
        # models have the same speed.  Combining them here used to turn genuine
        # piecewise retimes into one invalid constant-speed model; when refinement
        # rejected that model, both otherwise valid clips disappeared.  The merge
        # below is evidence based: it refines the combined range and accepts it only
        # when its visual error is no worse than the separate clips.
        groups.append((start, end, name, a, b))
    rows = []
    frame_step = float(np.median(np.diff(x)))
    for start, end, name, a, b in groups:
        # Hash-only hypotheses need four observations, but a compact-pixel group
        # has already survived exact 64x64 visual verification.  Preserve two- or
        # three-frame flashes/cutaways instead of turning them into black gaps.
        minimum_group = 2 if query_pixels is not None else 4
        if end - start < minimum_group:
            continue
        frames, times, hashes = source_data[name]
        speed, source_start, error = _refine_mapping(x[start:end], query_hashes[start:end],
                                                    times, hashes, a, b,
                                                    None if query_pixels is None else query_pixels[start:end],
                                                    source_pixels.get(name))
        if error > (max_pixel_mae if query_pixels is not None else 56):
            continue
        length = x[end - 1] - x[start] + frame_step
        sample_times = source_start + speed * (x[[start, end - 1]] - x[start])
        ids = (np.searchsorted(times, sample_times + 1e-8, side="right") - 1).clip(0, len(times)-1)
        rows.append(dict(
            edited_start_frame=int(edited.iloc[start]["edited_frame_number"]),
            edited_start_time_ms=x[start] * 1000,
            original_video_name=name,
            original_start_frame=int(frames.iloc[ids[0]]["original_frame_number"]),
            original_start_time_ms=times[ids[0]] * 1000,
            edited_end_frame=int(edited.iloc[end-1]["edited_frame_number"]),
            edited_end_time_ms=x[end-1] * 1000,
            original_end_frame=int(frames.iloc[ids[1]]["original_frame_number"]),
            original_end_time_ms=times[ids[1]] * 1000,
            speed=speed, edited_total_frames=int(edited.iloc[-1]["edited_frame_number"]) + 1,
            source_start_time_ms=max(0, source_start) * 1000,
            source_end_time_ms=(source_start + speed * length) * 1000,
            confidence=max(0., 1 - error / 64)))
    merged = []
    for row in rows:
        if merged:
            prev = merged[-1]
            if (prev["original_video_name"] == row["original_video_name"]
                    and prev["edited_end_frame"] + 1 == row["edited_start_frame"]
                    # Preserve visible retime boundaries.  A weighted mean error can
                    # hide a badly fitted short clip inside a much longer neighbour.
                    and abs(prev["speed"] - row["speed"]) < .02
                    and abs(prev["source_end_time_ms"] - row["source_start_time_ms"]) < 250):
                begin = int(np.searchsorted(edited["edited_frame_number"], prev["edited_start_frame"]))
                end = int(np.searchsorted(edited["edited_frame_number"], row["edited_end_frame"])) + 1
                frames, times, hashes = source_data[row["original_video_name"]]
                a = prev["speed"]
                b = prev["source_start_time_ms"] / 1000 - a * x[begin]
                a, start_time, error = _refine_mapping(x[begin:end], query_hashes[begin:end], times, hashes, a, b,
                                                       None if query_pixels is None else query_pixels[begin:end],
                                                       source_pixels.get(row["original_video_name"]))
                n1 = prev["edited_end_frame"] - prev["edited_start_frame"] + 1
                n2 = row["edited_end_frame"] - row["edited_start_frame"] + 1
                separate = ((1-prev["confidence"])*64*n1 + (1-row["confidence"])*64*n2)/(n1+n2)
                if error <= separate + .25:
                    ids = (np.searchsorted(times, start_time + a*(x[[begin,end-1]]-x[begin]) + 1e-8, side="right")-1).clip(0,len(times)-1)
                    prev.update(edited_end_frame=row["edited_end_frame"], edited_end_time_ms=row["edited_end_time_ms"],
                                original_start_frame=int(frames.iloc[ids[0]]["original_frame_number"]),
                                original_start_time_ms=times[ids[0]]*1000,
                                original_end_frame=int(frames.iloc[ids[1]]["original_frame_number"]),
                                original_end_time_ms=times[ids[1]]*1000,
                                speed=a, source_start_time_ms=start_time*1000,
                                source_end_time_ms=(start_time+a*(x[end-1]-x[begin]+frame_step))*1000,
                                confidence=max(0.,1-error/64))
                    continue
        merged.append(row)
    if query_pixels is not None:
        merged = _recover_short_gaps(merged, edited, candidates, source_data, query_pixels,
                                     source_pixels, min(12., max_pixel_mae))
    return pd.DataFrame(merged)
