"""Provenance and sampling shared by reproducible video experiments."""

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_frames import get_ffmpeg_identity
from time_mapping import segment_source_times, source_frame_samples


def fixture_identity(inputs, recipe):
    media = []
    for value in inputs:
        path = Path(value).resolve()
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        media.append(dict(path=str(path), sha256=digest))
    return dict(version=1, inputs=media, recipe=recipe, ffmpeg=get_ffmpeg_identity(),
                opencv=cv2.__version__, numpy=np.__version__)


def check_fixture(destination, identity, required, force=False):
    """Reuse only complete, identical fixtures; never silently reuse other input."""
    destination = Path(destination)
    manifest = destination / 'fixture.json'
    if destination.exists() and any(destination.iterdir()):
        same = manifest.is_file() and json.loads(manifest.read_text()) == identity
        if same and all((destination / name).is_file() for name in required) and not force:
            return True
        if not force:
            raise ValueError('Fixture inputs, recipe or toolchain changed, or output is incomplete; '
                             'use a new --destination or rebuild with --force')
    destination.mkdir(parents=True, exist_ok=True)
    # A failed rebuild must never retain a completion marker.
    manifest.unlink(missing_ok=True)
    return False


def finish_fixture(destination, identity):
    (Path(destination) / 'fixture.json').write_text(json.dumps(identity, indent=2) + '\n')


def predict_frames(segments_path, output_dir, edited_times):
    """Evaluate the same time map and blend weights used by export and rendering."""
    rows = pd.read_csv(segments_path)
    count = len(edited_times)
    names = np.full(count, '', dtype=object)
    ids = np.full(count, -1, dtype=int)
    weights = np.zeros(count)
    previous_end = 0
    for row in rows.sort_values('edited_start_frame').to_dict('records'):
        start, end = int(row['edited_start_frame']), int(row['edited_end_frame']) + 1
        if start < previous_end or end <= start or end > count:
            raise ValueError('Invalid or overlapping prediction ranges')
        previous_end = end
        name = row['original_video_name']
        manifest = Path(output_dir) / name / 'compact_frames.json'
        source_times = np.asarray(json.loads(manifest.read_text())['timestamps_ms']) / 1000
        targets = segment_source_times(row, edited_times)
        sampling = row.get('frame_sampling', 'floor')
        sampling = 'floor' if pd.isna(sampling) else sampling
        left, _, blend = source_frame_samples(source_times, targets, sampling)
        names[start:end], ids[start:end], weights[start:end] = name, left, blend
    return rows, names, ids, weights


def frame_metrics(expected_names, expected_ids, names, ids, weights):
    expected_names, expected_ids = np.asarray(expected_names), np.asarray(expected_ids)
    known = expected_names != ''
    correct = names == expected_names
    error = np.abs(ids - expected_ids)
    # Generated fixtures select single frames: an unintended blend is not exact.
    exact = correct & (error == 0) & (weights <= 1e-8)
    return dict(total_frames=len(ids), matched_truth_frames=int(known.sum()),
                source_accuracy=float(correct[known].mean()) if known.any() else None,
                exact_frame_accuracy=float(exact[known].mean()) if known.any() else None,
                within_one_frame=float((correct[known] & (error[known] <= 1)).mean()) if known.any() else None,
                matched_coverage=float((ids[known] >= 0).mean()) if known.any() else None,
                unknown_frames=int((~known).sum()), false_matches=int((ids[~known] >= 0).sum()),
                blended_frames=int((weights > 1e-8).sum()))
