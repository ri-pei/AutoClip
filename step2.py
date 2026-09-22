"""Step 2: batch pHash calculation for active compact-frame caches."""

from pathlib import Path

from common import discover_source_videos
from compact_frames import hash_compact


def main_step2(settings):
    output = Path(settings["output_dir"])
    videos = [settings["edited_video_path"], *discover_source_videos(
        settings["source_dir"], settings["edited_video_path"])]
    for path in videos:
        name = Path(path).stem
        if not (output / name / "compact_frames.json").is_file():
            raise FileNotFoundError(f"Missing compact frames for {name}; run Step 1 first")
        hash_compact(name, output)
    print("Step 2: active video pHash caches ready")


if __name__ == "__main__":
    from settings import load_settings
    main_step2(load_settings("config.toml"))
