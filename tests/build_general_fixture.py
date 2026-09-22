"""Two-source, mixed-FPS MV with known cadence, freezes and unmatched frames.

The ground truth is only read by the evaluator, never by the matching pipeline.
Inputs and a seed can be changed to repeat the experiment on other video genres.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixture_support import (check_fixture, finish_fixture, fixture_identity,
                             predict_frames, frame_metrics)


def build(first, second, destination, seed=73041, force=False, review_jumps=True):
    first, second, destination = Path(first).resolve(), Path(second).resolve(), Path(destination).resolve()
    if first.is_relative_to(destination) or second.is_relative_to(destination):
        raise ValueError("Fixture destination must not contain input media")
    identity = fixture_identity([first, second], dict(kind="mixed", version=2, seed=seed,
                                                     review_jumps=review_jumps))
    if check_fixture(destination, identity,
                     ["sources/episode_a.mp4", "sources/episode_b.mp4", "mixed.mp4",
                      "truth.json", "config.toml"], force):
        print(f"Verified fixture cache: {destination}")
        return
    (destination / 'sources').mkdir(exist_ok=True)
    fps = {'episode_a': 24, 'episode_b': 30}
    inputs = {'episode_a': (first, 105), 'episode_b': (second, 40)}
    pictures = {}
    for name, (path, start) in inputs.items():
        output = destination / 'sources' / f'{name}.mp4'
        resolution = '320:180' if name == 'episode_a' else '640:360'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(start), '-i', str(path),
                            '-t', '20', '-an', '-vf', f'fps={fps[name]},scale={resolution}',
                            '-c:v', 'libx264', '-crf', '16', str(output)], check=True)
        capture = cv2.VideoCapture(str(output))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA))
        capture.release()
        pictures[name] = frames
    # All offsets are local to these newly normalized source files.  Alternating
    # sources prevents a single-source ordering heuristic from passing the test.
    clips = [('episode_a', 40, 1.1, 96), ('episode_b', 60, .65, 80),
             ('episode_a', 180, 2., 60), ('episode_b', 470, 0., 3),
             ('episode_b', 260, 1.25, 112), ('episode_a', 370, 0., 40),
             ('episode_a', 260, .8, 100), ('', -1, 0., 24),
             ('episode_b', 15, 1.7, 90)]
    rng = np.random.default_rng(seed)
    clips = [(name, int(rng.integers(5, len(pictures[name]) -
                                   int(np.ceil(length * speed * fps[name] / 60)) - 5)) if name else -1,
              speed, length) for name, _, speed, length in clips]
    truth, frames, segments = [], [], []
    for name, first_frame, speed, length in clips:
        start = len(frames)
        segments.append(dict(start=start, end=start+length-1, source=name))
        for local in range(length):
            if name:
                index = first_frame + int(np.floor(local * speed * fps[name] / 60 + 1e-8))
                # A short duplicated-frame interval exercises cadence correction
                # independently of the constant nominal speed.
                if name == 'episode_b' and speed == 1.25 and 30 <= local < 58:
                    index = first_frame + int(np.floor((local-1) * speed * fps[name] / 60 + 1e-8))
                picture = pictures[name][index].copy()
                # Small exposure change + recompression; no exact pixel copies.
                picture = np.clip(picture.astype(np.int16) + 2, 0, 255).astype(np.uint8)
            else:
                index = -1
                picture = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
            # A target-only watermark tests masking in target coordinates, even
            # when the referenced source has a different spatial resolution.
            picture[:18, :48] = 255
            frames.append(picture)
            truth.append([name, index])
    output = destination / 'mixed.mp4'
    process = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                                '-s', '320x180', '-r', '60', '-i', 'pipe:0',
                                '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', str(output)],
                               stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait():
        raise RuntimeError('Fixture encoding failed')
    (destination / 'truth.json').write_text(json.dumps(dict(seed=seed, fps=fps, frames=truth, clips=segments), indent=2))
    (destination / 'config.toml').write_text(f'''[paths]
working_dir = "."
edited_video = "mixed.mp4"
source_dir = "sources"
output_dir = "output"
[step1]
frame_storage = "compact"
mask_rect = [0, 0, 48, 18]
[step3]
matcher = "numpy"
top_k = 20
[step4]
alignment = "affine_pixels"
frame_refinement = true
short_jump_review = {str(review_jumps).lower()}
[step5]
frame_rate = 60
project_name = "Mixed source validation"
include_edited_audio = false
''')
    finish_fixture(destination, identity)
    print(f'Built {len(frames)} target frames, two source frame rates; truth: {destination / "truth.json"}')


def evaluate(destination, segments_path=None, output_dir=None):
    destination = Path(destination)
    output_dir = Path(output_dir) if output_dir else destination / 'output'
    truth = json.loads((destination / 'truth.json').read_text())
    segments_path = segments_path or output_dir / 'final_video_segments_refined.csv'
    edited_times = np.asarray(json.loads((output_dir/'mixed'/'compact_frames.json').read_text())['timestamps_ms']) / 1000
    if len(edited_times) != len(truth['frames']):
        raise ValueError("Target cache and truth have different frame counts")
    rows, names, ids, weights = predict_frames(segments_path, output_dir, edited_times)
    metrics = frame_metrics([x[0] for x in truth['frames']], [x[1] for x in truth['frames']],
                            names, ids, weights)
    metrics['output_clips'] = len(rows)
    metrics['segments_path'] = str(Path(segments_path).resolve())
    print(json.dumps(metrics, indent=2))
    report = Path(segments_path).with_suffix('.evaluation.json')
    report.write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path)
    parser.add_argument('--second', type=Path)
    parser.add_argument('--destination', type=Path, default=ROOT/'tests/.work/generalization')
    parser.add_argument('--seed', type=int, default=73041)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--no-jump-review', action='store_true')
    parser.add_argument('--segments', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.evaluate:
        evaluate(args.destination, args.segments, args.output_dir)
    else:
        if args.first is None or args.second is None:
            parser.error("building requires --first and --second")
        build(args.first, args.second, args.destination, args.seed, args.force, not args.no_jump_review)
