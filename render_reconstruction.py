"""Render the final segment CSV, including speed changes, against the target MV."""

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pandas as pd

from common import discover_source_videos, get_frame_timestamps_map_json, get_video_metadata
from settings import load_settings
from step5 import get_fcpxml_time_params
from time_mapping import source_frame_samples, segment_source_times


def render_timestamps(settings, video_path):
    """Reuse verified Step 1 PTS instead of decoding whole episodes again."""
    video = Path(video_path)
    manifest = Path(settings["output_dir"]) / video.stem / "compact_frames.json"
    if settings.get("frame_storage") == "compact" and manifest.is_file():
        data = json.loads(manifest.read_text())
        signature, stat = data.get("signature", {}), video.stat()
        if (signature.get("version") != 2 or signature.get("path") != str(video.resolve())
                or signature.get("size") != stat.st_size or signature.get("mtime_ns") != stat.st_mtime_ns):
            raise ValueError(f"Stale frame timestamps; rerun Steps 1–4: {video}")
        times = np.asarray(data["timestamps_ms"], dtype=float) / 1000
        if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"Invalid frame timestamps: {manifest}")
        return times
    timestamps = get_frame_timestamps_map_json(str(video))
    if not timestamps or sorted(timestamps) != list(range(len(timestamps))):
        raise ValueError(f"Missing frame timestamps: {video}")
    return np.asarray(list(timestamps.values()), dtype=float)


def comparison_filter(frame_numerator, frame_denominator):
    # Equal FPS is insufficient: 1/16000 and 1/15360 input clocks round the
    # same frame to different PTS. Both streams must use the same exact clock.
    clock = f"{frame_numerator}/{frame_denominator}"
    return (f"[0:v]settb=expr={clock},setpts=N[top];"
            f"[1:v]settb=expr={clock},setpts=N[bottom];"
            "[top][bottom]vstack=inputs=2:shortest=1[v]")


def render(settings, output_path, comparison_path=None, preset="fast", threads=4):
    if preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
        raise ValueError("Unsupported H.264 encoding preset")
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("Encoding threads must be a positive integer")
    output_path = Path(output_path)
    temporary = output_path.with_name(output_path.stem+".partial.mp4")
    edited_path = Path(settings["edited_video_path"])
    source_paths = {Path(p).stem: p for p in discover_source_videos(settings["source_dir"], str(edited_path))}
    input_paths = {edited_path.resolve(), *(Path(p).resolve() for p in source_paths.values())}
    if output_path.resolve() in input_paths or temporary.resolve() in input_paths:
        raise ValueError("Output and temporary files must not overwrite input media")
    rows = pd.read_csv(Path(settings["output_dir"])/settings["final_segments_csv"])
    metadata = get_video_metadata(str(edited_path))
    width, height = metadata["width"], metadata["height"]
    _, numerator, denominator = get_fcpxml_time_params(settings["frame_rate"] or metadata["avg_frame_rate"])
    fps = denominator / numerator
    edited_time_array = render_timestamps(settings, edited_path)
    frame_count = len(edited_time_array)
    source_times = {name: render_timestamps(settings, path)
                    for name, path in source_paths.items() if name in set(rows.original_video_name)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]
    if settings["include_edited_audio"]:
        command += ["-i", str(edited_path), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    command += ["-c:v", "libx264", "-threads", str(threads), "-preset", preset,
                "-crf", "18", "-pix_fmt", "yuv420p", str(temporary)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    next_output = 0
    black = np.zeros((height, width, 3), dtype=np.uint8)
    try:
        for clip_index, (_, row) in enumerate(rows.sort_values("edited_start_frame").iterrows()):
            start, end = int(row.edited_start_frame), int(row.edited_end_frame)+1
            if start < next_output or end > frame_count:
                raise ValueError("Invalid or overlapping timeline ranges")
            while next_output < start:
                encoder.stdin.write(black.tobytes()); next_output += 1
            name = row.original_video_name
            targets = segment_source_times(row, edited_time_array)
            sampling = row.get("frame_sampling", "floor")
            sampling = "floor" if pd.isna(sampling) else sampling
            ids, right_ids, weights = source_frame_samples(source_times[name], targets, sampling)
            if np.any(ids < 0) or np.any(ids >= len(source_times[name])) or np.any(np.diff(ids) < 0):
                raise ValueError("Segment maps outside source media")
            print(f"Render clip {clip_index + 1}/{len(rows)}: frames {start}–{end - 1}, {name}", flush=True)
            if hasattr(cv2, "CAP_PROP_N_THREADS"):
                capture = cv2.VideoCapture(source_paths[name], cv2.CAP_FFMPEG,
                                           [cv2.CAP_PROP_N_THREADS, threads])
            else:
                capture = cv2.VideoCapture(source_paths[name])
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Cannot open source video: {source_paths[name]}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(ids[0]))
            current = int(ids[0])-1
            pictures = {}
            try:
                for target, following, weight in zip(ids, right_ids, weights):
                    needed = following if weight > 1e-8 else target
                    while current < needed:
                        ok, picture = capture.read()
                        if not ok:
                            raise RuntimeError(f"Cannot decode {name} frame {target}")
                        current += 1
                        pictures[current] = picture
                        if len(pictures) > 2:
                            del pictures[min(pictures)]
                    frame = pictures[int(target)]
                    if weight > 1e-8:
                        frame = cv2.addWeighted(frame, 1-weight, pictures[int(following)], weight, 0)
                    if frame.shape[:2] != (height,width):
                        frame = cv2.resize(frame, (width,height), interpolation=cv2.INTER_AREA)
                    encoder.stdin.write(frame.tobytes())
                    next_output += 1
            finally:
                capture.release()
        while next_output < frame_count:
            encoder.stdin.write(black.tobytes());next_output += 1
    finally:
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError("Reconstruction encoder failed")
    temporary.replace(output_path)
    if comparison_path:
        comparison_path = Path(comparison_path)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        protected = {edited_path.resolve(), output_path.resolve(), *(Path(p).resolve() for p in source_paths.values())}
        if comparison_path.resolve() in protected:
            raise ValueError("Comparison path conflicts with input or reconstruction")
        comparison_temporary = comparison_path.with_name(
            comparison_path.stem + ".partial" + comparison_path.suffix
        )
        if comparison_temporary.resolve() in protected:
            raise ValueError("Comparison temporary path conflicts with input or reconstruction")
        # The edited input may report a nominal/average rate such as 59.9995 while
        # the reconstruction is encoded at exactly 60 fps.  vstack's normal PTS
        # framesync can then pair neighbouring frame numbers, making a correct
        # reconstruction look temporally wrong.  Reset both streams from their frame
        # indices; shortest=1 ends the video after the shared frame count while the
        # target audio is still allowed to keep its original (slightly longer) tail.
        filter_graph = comparison_filter(numerator, denominator)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", str(threads), "-i", str(edited_path),
                        "-threads", str(threads), "-i", str(output_path), "-filter_complex_threads", str(threads),
                        "-filter_complex", filter_graph, "-map", "[v]", "-map", "0:a?",
                        "-r", str(fps),
                        "-c:v", "libx264", "-threads", str(threads), "-preset", preset, "-crf", "20",
                        "-c:a", "copy", str(comparison_temporary)], check=True)
        comparison_temporary.replace(comparison_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--preset", default="fast",
                        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"),
                        help="H.264 encoding effort only; does not change the selected frames or time map")
    parser.add_argument("--threads", type=int, default=4,
                        help="Limit decode/encode workers so previews do not monopolize the machine")
    args = parser.parse_args()
    render(load_settings(args.config), args.output, args.comparison, args.preset, args.threads)
