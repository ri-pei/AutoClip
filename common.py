import os
import sys
import json
import re
import shutil
import subprocess
import cv2
import config


# ==============================================================================
# 1. 全局设置与路径解析
# ==============================================================================

ABS_WORKING_DIR = os.path.abspath(config.WORKING_DIR or ".")


def _resolve_path(path_str):
    """
    辅助函数：如果path_str是绝对路径则直接返回，否则与工作目录合并。
    """
    if not path_str:
        return None  # 处理空或None的情况
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(ABS_WORKING_DIR, path_str)


# 预处理所有路径配置
ABS_EDITED_VIDEO_PATH = _resolve_path(config.EDITED_VIDEO_FILENAME)
ABS_OUTPUT_DIR = _resolve_path(config.OUTPUT_DIR)
ABS_SOURCE_VIDEO_FOLDER = _resolve_path(config.SOURCE_VIDEO_FOLDER)
ABS_REF_ORIGINAL_FRAME_PATH = _resolve_path(config.REF_ORIGINAL_FRAME_PATH)
ABS_REF_EDITED_FRAME_PATH = _resolve_path(config.REF_EDITED_FRAME_PATH)
ABS_USER_COLOR_LUT_PATH = _resolve_path(config.USER_COLOR_LUT_PATH)


# ==============================================================================
# 2. 启动时环境检查
# ==============================================================================


def _check_dependencies():
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
        sys.exit(1)


print("正在检查项目依赖...")
_check_dependencies()
print("依赖检查通过。")


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
        "stream=width,height,avg_frame_rate",
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


def format_timestamp_from_seconds(ts_seconds):
    # 将秒数格式化为 HH-MM-SS-SSS 格式的时间戳字符串
    if ts_seconds < 0:
        ts_seconds = 0
    hours = int(ts_seconds / 3600)
    minutes = int((ts_seconds % 3600) / 60)
    seconds = int(ts_seconds % 60)
    milliseconds = int((ts_seconds - int(ts_seconds)) * 1000)
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}-{milliseconds:03d}"


def parse_frame_filename(filename):
    """
    解析帧文件名以提取视频名称前缀、帧编号和时间字符串。
    示例: <video_filename_no_ext>_frame_0000000_time_HH-MM-SS-mmm.png
    返回: (video_prefix, frame_number, time_str) 或 (None, None, None) 如果没有匹配。
    """
    # Regex to capture:
    # 1. Video name prefix (anything before _frame_)
    # 2. Frame number (digits)
    # 3. Time string (HH-MM-SS-mmm)
    match = re.match(r"(.+)_frame_(\d+)_time_(\d{2}-\d{2}-\d{2}-\d{3})\.png", filename)
    if match:
        video_prefix = match.group(1)
        frame_number = int(match.group(2))
        time_str = match.group(3)  # HH-MM-SS-mmm
        return video_prefix, frame_number, time_str
    return None, None, None


def time_str_to_milliseconds(time_str):
    """
    Converts HH-MM-SS-mmm time string to total milliseconds.
    """
    if not time_str:
        return 0
    parts = time_str.split("-")
    if len(parts) == 4:
        h, m, s, ms = map(int, parts)
        return (h * 3600 + m * 60 + s) * 1000 + ms
    return 0


def stitch_images_vertically(img_path1, img_path2, output_path):
    """
    Reads two images, stitches them vertically (img1 on top of img2),
    and saves the result. Paths are assumed to be absolute or resolvable.
    """
    try:
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        if img1 is None:
            print(f"Warning: Could not read image for stitching: {img_path1}")
            return False
        if img2 is None:
            print(f"Warning: Could not read image for stitching: {img_path2}")
            return False

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if w1 != w2:
            print(
                f"Warning: Image widths differ for stitching ({w1} vs {w2}) for "
                f"{os.path.basename(img_path1)} and {os.path.basename(img_path2)}. "
                f"Resizing second image to width {w1}."
            )
            target_height_img2 = int(h2 * (w1 / w2))
            img2_resized = cv2.resize(
                img2, (w1, target_height_img2), interpolation=cv2.INTER_AREA
            )
            stitched_image = cv2.vconcat([img1, img2_resized])
        else:
            stitched_image = cv2.vconcat([img1, img2])

        cv2.imwrite(output_path, stitched_image)
        return True
    except Exception as e:
        print(
            f"Error stitching images: {e}"
            f"   {os.path.basename(img_path1)}"
            f"    and"
            f"   {os.path.basename(img_path2)}"
        )
        return False
