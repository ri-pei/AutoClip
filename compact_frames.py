"""Compact Step 1/2 intermediates: 64x64 grayscale arrays, with real PTS."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from scipy.fftpack import dct

from common import get_frame_timestamps_map_json


def get_ffmpeg_identity():
    """Decoder/scaler versions can change pixels with unchanged filter text."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise FileNotFoundError("ffmpeg was not found on PATH")
    version = subprocess.check_output([executable, "-version"], text=True, stderr=subprocess.PIPE)
    return {"path": str(Path(executable).resolve()), "version": version.splitlines()[0],
            "build_sha256": hashlib.sha256(version.encode()).hexdigest()}


def extract_compact(video_path, output_dir, filters):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    video = Path(video_path)
    stat = video.stat()
    signature = {"version": 2, "path": str(video.resolve()), "size": stat.st_size,
                 "mtime_ns": stat.st_mtime_ns, "filters": filters,
                 "lut_files": [], "ffmpeg": get_ffmpeg_identity()}
    # Filter text alone cannot detect a LUT modified at the same path.
    for item in filters:
        if item.startswith("lut3d=file='"):
            lut = Path(item[len("lut3d=file='"):-1])
            signature["lut_files"].append(hashlib.sha256(lut.read_bytes()).hexdigest())
    manifest_path = root / "compact_frames.json"
    array_path = root / "compact_frames.npy"
    if manifest_path.exists() and array_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("signature") == signature:
            frames = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if frames.shape == (len(manifest["timestamps_ms"]), 64, 64):
                print(f"Compact frame cache: {video.stem}")
                return True
    timestamps = get_frame_timestamps_map_json(str(video))
    if not timestamps or sorted(timestamps) != list(range(len(timestamps))):
        raise RuntimeError(f"Missing video frame timestamps: {video}")
    filter_text = ",".join([*filters, "format=gray", "scale=64:64:flags=lanczos"])
    temp_path = root / "compact_frames.partial.npy"
    frames = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.uint8,
                                     shape=(len(timestamps), 64, 64))
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-map", "0:v:0",
                 "-an", "-vf", filter_text, "-vsync", "0", "-f", "rawvideo",
                 "-pix_fmt", "gray", "pipe:1"], stdout=subprocess.PIPE, stderr=errors)
            try:
                buffer = memoryview(frames).cast("B")
                position = 0
                while position < len(buffer):
                    chunk = process.stdout.read(min(1024 * 1024, len(buffer) - position))
                    if not chunk:
                        break
                    buffer[position:position + len(chunk)] = chunk
                    position += len(chunk)
                extra = process.stdout.read(1)
                code = process.wait()
                errors.seek(0)
                if code or position != len(buffer) or extra:
                    raise RuntimeError(f"Compact decode failed or frame count changed: {errors.read().decode(errors='replace')}")
                del buffer
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
        frames.flush()
        del frames
        manifest_path.unlink(missing_ok=True)
        os.replace(temp_path, array_path)
        manifest = {"signature": signature,
                    "timestamps_ms": [timestamps[i] * 1000 for i in range(len(timestamps))]}
        temp_manifest = root / "compact_frames.partial.json"
        temp_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        os.replace(temp_manifest, manifest_path)
        return True
    finally:
        if temp_path.exists():
            temp_path.unlink()


def hash_compact(video_name, output_dir):
    root = Path(output_dir) / video_name
    manifest_bytes = (root / "compact_frames.json").read_bytes()
    # Bind both pixels and the four-column CSV schema to the hash cache.
    identity = hashlib.sha256(b"compact-phash-v2\n" + manifest_bytes).hexdigest()
    cache = root / f"{video_name}_phash.csv"
    stamp = root / "compact_phash.sha256"
    frames = np.load(root / "compact_frames.npy", mmap_mode="r", allow_pickle=False)
    times = json.loads(manifest_bytes)["timestamps_ms"]
    if len(frames) != len(times):
        raise RuntimeError("Compact frame manifest mismatch")
    if cache.exists() and stamp.exists() and stamp.read_text() == identity:
        cached = pd.read_csv(cache, dtype={"phash": str})
        if len(cached) == len(times):
            return True
    hashes = []
    for begin in range(0, len(frames), 256):
        transformed = dct(dct(frames[begin:begin + 256].astype(np.float64), axis=1), axis=2)
        low = transformed[:, :16, :16]
        bits = low > np.median(low, axis=(1, 2))[:, None, None]
        hashes.extend(row.tobytes().hex() for row in np.packbits(bits.reshape(len(bits), 256), axis=1))
    rows = []
    for i, (ts, phash) in enumerate(zip(times, hashes)):
        rows.append(dict(video_name=video_name, frame_number=i, timestamp_ms=ts, phash=phash))
    temporary = cache.with_suffix(".partial.csv")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, cache)
    stamp.write_text(identity)
    return True
