import os
import sys
import json
import shutil
import subprocess


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv")


# ==============================================================================
# 2. 启动时环境检查
# ==============================================================================


def check_dependencies():
    """
    检查核心外部依赖是否存在。
    如果缺少依赖，打印错误信息并退出程序。
    """
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print(
            "--------------------------------------------------------------------",
            file=sys.stderr,
        )
        print("错误: FFMPEG/FFPROBE 未找到。请确保已安装它们，", file=sys.stderr)
        print("并将其添加至系统的 PATH 环境变量。", file=sys.stderr)
        print("下载地址: https://ffmpeg.org/download.html", file=sys.stderr)
        print(
            "--------------------------------------------------------------------",
            file=sys.stderr,
        )
        raise RuntimeError("FFmpeg/ffprobe 不可用。")


def discover_source_videos(source_dir, edited_video_path=None):
    """递归查找 source 视频，并验证不带扩展名的文件名唯一。"""
    if not source_dir or not os.path.isdir(source_dir):
        raise FileNotFoundError(f"源视频目录不存在: {source_dir}")
    edited_normalized = (
        os.path.normcase(os.path.realpath(edited_video_path))
        if edited_video_path
        else None
    )
    videos = []
    for root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() not in VIDEO_EXTENSIONS:
                continue
            path = os.path.join(root, filename)
            if edited_normalized and os.path.normcase(os.path.realpath(path)) == edited_normalized:
                continue
            videos.append(os.path.abspath(path))
    videos.sort(key=lambda value: os.path.normcase(os.path.normpath(value)))
    if not videos:
        raise RuntimeError(f"源视频目录中没有受支持的视频: {source_dir}")

    stem_map = {}
    for path in videos:
        stem = os.path.splitext(os.path.basename(path))[0]
        stem_map.setdefault(stem, []).append(path)
    if edited_video_path:
        edited_stem = os.path.splitext(os.path.basename(edited_video_path))[0]
        if edited_stem in stem_map:
            stem_map[edited_stem].append(os.path.abspath(edited_video_path))
    conflicts = {stem: paths for stem, paths in stem_map.items() if len(paths) > 1}
    if conflicts:
        details = []
        for stem, paths in sorted(conflicts.items()):
            details.append(f"  {stem}:\n    " + "\n    ".join(paths))
        raise ValueError("视频文件名（不含扩展名）必须唯一:\n" + "\n".join(details))
    return videos


# ==============================================================================
# 3. 通用辅助函数
# ==============================================================================
def run_command(command_list):
    """
    执行外部命令并返回结果，包括进程对象以便检查return code
    """
    try:
        # Uncomment for debugging ffmpeg commands
        # print(f"DEBUG CMD: {' '.join(command_list)}")
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr = process.communicate()
        # Rreturn the process object as well, so the caller can check process.returncode
        return stdout, stderr, process
    except FileNotFoundError:
        print(
            f"Error: Command {command_list[0]} not found. "
            "Please ensure it is installed and available in your PATH."
        )
        raise
    except Exception as e:
        print(
            f"An unexpected error occurred with command {' '.join(command_list)}: {e}"
        )
        raise


def get_video_metadata(video_path):
    """获取视频的元数据，包括宽度、高度和平均帧率"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration:format=duration",
        "-of",
        "json",
        video_path,
    ]
    stdout, _, process = run_command(cmd)

    if process.returncode != 0 and not stdout:
        # If error and no stdout, likely critical
        print(
            f"ffprobe failed to get metadata for {os.path.basename(video_path)} "
            f"(return code {process.returncode})"
        )
        return None

    if stdout:
        try:
            data = json.loads(stdout)
            if not data.get("streams"):
                print(
                    f"Warning: ffprobe returned no streams for metadata of "
                    f"{os.path.basename(video_path)}. Output: {stdout[:200]}"
                )
                return None
            metadata = data["streams"][0]
            duration = metadata.get("duration", data.get("format", {}).get("duration"))
            if duration not in (None, "N/A"):
                metadata["duration_seconds"] = float(duration)
            if (
                isinstance(metadata.get("avg_frame_rate"), str)
                and "/" in metadata["avg_frame_rate"]
            ):
                num, den = map(int, metadata["avg_frame_rate"].split("/"))
                metadata["avg_frame_rate"] = num / den if den != 0 else 0
            else:
                metadata["avg_frame_rate"] = float(metadata.get("avg_frame_rate", 0))
            return metadata
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(
                f"Error parsing ffprobe JSON for metadata of "
                f"{os.path.basename(video_path)}: {e}. "
                f"Output: {stdout[:200]}"
            )
            return None
    return None


def get_frame_timestamps_map_json(video_path):
    """获取视频帧的时间戳映射，返回一个字典，键为帧索引，值为对应的时间戳（秒）。
    使用ffprobe获取视频帧的时间戳信息，解析JSON格式的输出
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pts_time,media_type",
        "-of",
        "json",
        video_path,
    ]
    stdout, _, process = run_command(cmd)
    if process.returncode != 0 and not stdout:
        print(
            f"ffprobe failed to get timestamps for {os.path.basename(video_path)} "
            f"(return code {process.returncode})"
        )
        return {}

    timestamps_map = {}
    if not stdout:
        print(
            f"  Could not get frame timestamps (stdout empty) for "
            f"{os.path.basename(video_path)}"
        )
        return {}
    try:
        data = json.loads(stdout)
        frames_data = data.get("frames", [])
        parsed_count = 0
        for frame_index, frame_info in enumerate(frames_data):
            if frame_info.get("media_type") == "video":
                ts_str = frame_info.get(
                    "best_effort_timestamp_time", frame_info.get("pts_time")
                )
                if ts_str is not None:
                    try:
                        timestamps_map[frame_index] = float(ts_str)
                        parsed_count += 1
                    except (ValueError, TypeError):
                        pass
        # if parsed_count == 0 and len(frames_data) > 0:
        # print(
        #     f"  Warning: No timestamps parsed for {os.path.basename(video_path)} "
        #     f"from {len(frames_data)} ffprobe entries."
        # )
    except json.JSONDecodeError as e:
        print(
            f"  Error decoding ffprobe JSON for timestamps of "
            f"{os.path.basename(video_path)}: {e}. Output: {stdout[:200]}"
        )
        return {}
    except Exception as e:  # Catch any other unexpected error
        print(
            f"  Unexpected error processing ffprobe JSON for "
            f"{os.path.basename(video_path)}: {e}"
        )
        return {}
    return timestamps_map
