"""Generate reproducible MV copies and independent frame-level ground truth."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from fixture_support import check_fixture, finish_fixture, fixture_identity

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tests" / ".work" / "retime"
FPS = 24
WIDTH, HEIGHT = 640, 360
# Speed and output frame count. Source positions are sampled from the actual input.
CASES = {
    "development": [(1, 96), (2, 72), (.5, 120), (1.25, 96), (.75, 96),
                    (1.5, 72), (1, 72), (2, 48), (.5, 12)],
    "holdout": [(1.37, 96), (.6, 120), (2.5, 72), (1, 96), (.8, 72),
                (1.75, 48), (1, 24), (2, 12)],
}

CASES["validation"] = list(zip(
    (1, .5, 2, .7, 1.3, 2.2, .55, 1.6, 1.37, 2.75, .8, 1),
    (72, 96, 48, 72, 72, 48, 96, 48, 72, 48, 12, 24)))


def build(input_path, force=False, destination=WORK, seed=91847):
    input_path, destination = Path(input_path).resolve(), Path(destination).resolve()
    if input_path.is_relative_to(destination):
        raise ValueError("Fixture destination must not contain the input media")
    identity = fixture_identity([input_path], dict(kind="retime", version=3, fps=FPS, seed=seed,
                                                 width=WIDTH, height=HEIGHT, cases=CASES))
    # Normalize tuple/list representations before comparing serialized manifests.
    identity = json.loads(json.dumps(identity))
    required = ["sources/reference.mp4"] + [f"{name}{suffix}" for name in CASES
                                             for suffix in (".mp4", ".truth.json", ".toml")]
    if check_fixture(destination, identity, required, force):
        print(f"Verified fixture cache: {destination}")
        return
    source_dir = destination / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    source = source_dir / "reference.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(input_path),
                        "-map", "0:v:0", "-an", "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "16", str(source)], check=True)
    reader = cv2.VideoCapture(str(source))
    source_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    reader.release()
    for case_index, (name, clips) in enumerate(CASES.items()):
        rng = np.random.default_rng(seed + case_index)
        output = destination / f"{name}.mp4"
        truth_path = destination / f"{name}.truth.json"
        mapping, segments = [], []
        for speed, length in clips:
            rate = Fraction(str(speed))
            needed_count = (length - 1) * rate.numerator // rate.denominator + 1
            if needed_count > source_count:
                raise ValueError(f"Input needs at least {needed_count / FPS:.2f} seconds for {name}")
            start = int(rng.integers(0, source_count - needed_count + 1))
            offset = len(mapping)
            mapping.extend(start + (i * rate.numerator // rate.denominator) for i in range(length))
            segments.append(dict(edited_start_frame=offset, edited_end_frame=offset + length - 1,
                                 original_video_name="reference", original_start_frame=start,
                                 speed=speed, output_frames=length))
        needed = set(mapping)
        pictures = {}
        reader = cv2.VideoCapture(str(source))
        i = 0
        while needed:
            ok, frame = reader.read()
            if not ok:
                raise RuntimeError(f"Source too short, missing frames: {min(needed)}")
            if i in needed:
                pictures[i] = frame
                needed.remove(i)
            i += 1
        reader.release()
        process = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
             "-stream_loop", "-1", "-i", str(input_path), "-map", "0:v:0", "-map", "1:a:0",
             "-t", str(len(mapping) / FPS), "-c:v", "libx264", "-preset", "fast",
             "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(output)],
            stdin=subprocess.PIPE)
        try:
            for frame_id in mapping:
                process.stdin.write(pictures[frame_id].tobytes())
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError("MV encoding failed")
        truth_path.write_text(json.dumps(dict(input=str(input_path), source=str(source),
                                             fps=FPS, segments=segments,
                                             source_frames=mapping), indent=2))
        config = f'''[paths]
working_dir = "."
edited_video = "{name}.mp4"
source_dir = "sources"
output_dir = "compact_{name}"

[step1]
frame_storage = "compact"

[step3]
matcher = "numpy"
top_k = 20

[step4]
alignment = "affine_pixels"
frame_refinement = true
short_jump_review = true

[step5]
frame_rate = 24
project_name = "Reconstructed {name}"
include_edited_audio = true
'''
        (destination / f"{name}.toml").write_text(config)
        print(f"Built {name}: {len(mapping)} frames, {len(segments)} clips", flush=True)

    finish_fixture(destination, identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=WORK)
    parser.add_argument("--seed", type=int, default=91847)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.input, args.force, args.destination, args.seed)
