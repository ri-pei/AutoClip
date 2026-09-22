"""Decode rendered outputs once and compare their pixels, frame counts and audio."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import load_settings
from compact_frames import get_ffmpeg_identity


def decode_small(path, mask=None, stacked=False):
    filters = []
    if mask:
        x, y, width, height = mask
        filters.append(f'drawbox=x={x}:y={y}:w={width}:h={height}:color=black:t=fill')
        if stacked:
            filters.append(f'drawbox=x={x}:y=ih/2+{y}:w={width}:h={height}:color=black:t=fill')
    small_height = 128 if stacked else 64
    filters += ['format=gray', f'scale=64:{small_height}:flags=lanczos']
    data = subprocess.check_output(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path), '-map', '0:v:0',
                                    '-filter_threads', '2',
                                    '-an', '-vf', ','.join(filters), '-vsync', '0',
                                    '-pix_fmt', 'gray', '-f', 'rawvideo', 'pipe:1'])
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, small_height, 64)


def audio_hash(path):
    return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0',
                                    '-c:a', 'copy', '-f', 'hash', '-hash', 'sha256', 'pipe:1'], text=True).strip()


def audit(settings, outputs, query=None):
    target = Path(settings['edited_video_path'])
    # Never mix a cached reference from another FFmpeg build with fresh decoded
    # outputs: 7.1 and 8.0, for example, can produce different gray conversions.
    if query is None:
        query = decode_small(target, settings['mask_rect'])
    expected_audio = audio_hash(target) if settings['include_edited_audio'] else None
    result = {'reference': dict(path=str(target), frames=len(query), freshly_decoded=True,
                                ffmpeg=get_ffmpeg_identity())}
    for path in outputs:
        actual = decode_small(path, settings['mask_rect'])
        if len(actual) != len(query):
            raise AssertionError(f'Render frame count: {len(actual)} != {len(query)}')
        errors = np.empty(len(query))
        for begin in range(0, len(query), 128):
            errors[begin:begin+128] = np.abs(actual[begin:begin+128].astype(float)-query[begin:begin+128]).mean((1, 2))
        audio_equal = audio_hash(path) == expected_audio if expected_audio else None
        if expected_audio and not audio_equal:
            raise AssertionError(f'Render changed audio packet content: {path}')
        result[str(path)] = dict(frames=len(actual), mean_mae=float(errors.mean()),
                                 p95_mae=float(np.percentile(errors, 95)), p99_mae=float(np.percentile(errors, 99)),
                                 maximum_mae=float(errors.max()), audio_packets_identical=audio_equal)
    return result


def audit_comparison(settings, path, query=None):
    target = Path(settings['edited_video_path'])
    if query is None:
        query = decode_small(target, settings['mask_rect'])
    actual = decode_small(path, settings['mask_rect'], stacked=True)
    if len(actual) != len(query):
        raise AssertionError(f'Comparison frame count: {len(actual)} != {len(query)}')
    result = dict(frames=len(actual))
    for name, pictures in [('target_panel', actual[:, :64]), ('reconstructed_panel', actual[:, 64:])]:
        errors = np.empty(len(query))
        for begin in range(0, len(query), 128):
            errors[begin:begin+128] = np.abs(pictures[begin:begin+128].astype(float)-query[begin:begin+128]).mean((1, 2))
        result[name] = dict(mean_mae=float(errors.mean()), p99_mae=float(np.percentile(errors, 99)),
                            maximum_mae=float(errors.max()))
    result['audio_packets_identical'] = audio_hash(path) == audio_hash(target) if settings['include_edited_audio'] else None
    if settings['include_edited_audio'] and not result['audio_packets_identical']:
        raise AssertionError('Comparison changed the original audio packets')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.toml')
    parser.add_argument('--video', type=Path, action='append', required=True)
    parser.add_argument('--comparison', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = load_settings(args.config)
    query = decode_small(settings['edited_video_path'], settings['mask_rect'])
    result = audit(settings, args.video, query)
    if args.comparison:
        result[str(args.comparison)] = audit_comparison(settings, args.comparison, query)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
