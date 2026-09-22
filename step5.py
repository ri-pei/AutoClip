import os
import math
from fractions import Fraction
import csv
import json
from urllib.parse import quote
import xml.etree.ElementTree as ET
from xml.dom import minidom  # 用于格式化输出（pretty printing）
from common import (
    discover_source_videos,
    get_video_metadata,
    run_command,
)
from time_mapping import parse_time_map



def get_fcpxml_time_params(frame_rate_float):
    """
    根据浮点数帧率计算FCPXML所需的时间参数。
    这是FCPXML格式的关键部分，用于所有时间码计算。

    Args:
        frame_rate_float (float): 视频的平均帧率 (例如, 23.976, 25.0, 29.97)。

    Returns:
        tuple: 包含 (frame_duration_str, numerator, denominator) 的元组。
               例如 ( "1001/24000s", 1001, 24000 )。
               如果无法识别帧率，则返回 (None, None, None)。
    """
    if not math.isfinite(frame_rate_float) or frame_rate_float <= 0:
        raise ValueError("Frame rate must be finite and positive")
    # 0.1% tolerance makes 59.9995 incorrectly match 59.94.  Pick the nearest
    # standard first and use a tolerance narrower than their separation.
    standards = [(1, fps) for fps in (24, 25, 30, 48, 50, 60, 120)]
    standards += [(1001, fps) for fps in (24000, 30000, 48000, 60000, 120000)]
    numerator, denominator = min(standards, key=lambda pair: abs(pair[1] / pair[0] - frame_rate_float))
    if not math.isclose(frame_rate_float, denominator / numerator, rel_tol=1e-4):
        duration = (1 / Fraction(str(frame_rate_float))).limit_denominator(1_000_000)
        numerator, denominator = duration.numerator, duration.denominator
    return f"{numerator}/{denominator}s", numerator, denominator


def format_time_value(frames, fd_numerator, fd_denominator):
    """将帧数格式化为FCPXML的时间值字符串 'value/denominator s'。"""
    # 此处的 'value' 是 帧数 * 帧持续时间的分子 (frames * fd_numerator)
    return f"{int(frames * fd_numerator)}/{fd_denominator}s"


def format_fcpxml_path(media_path):
    """将本地文件路径格式化为FCPXML所需的 'file://' URI。"""
    # 使用os.path.join确保路径分隔符正确
    # 转换为绝对路径以确保URI的有效性
    abs_path = os.path.abspath(media_path)

    # 将Windows路径的反斜杠'\'替换为正斜杠'/'
    posix_path = abs_path.replace(os.path.sep, "/")

    # 为Windows驱动器号路径（如 C:/...）添加前导斜杠
    if ":" in posix_path and posix_path[1] == ":":  # 例如 C:/...
        posix_path = "/" + posix_path

    # FCPXML期望的格式是 file://localhost/path/to/file
    # Resolve等软件通常也能处理 file:///path/to/file 格式
    return f"file://localhost{quote(posix_path, safe='/:')}"


def get_audio_metadata(video_path):
    """读取第一个音频流以及媒体总时长。"""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,channel_layout:format=duration",
        "-of",
        "json",
        video_path,
    ]
    stdout, stderr, process = run_command(command)
    if process.returncode != 0:
        raise RuntimeError(f"ffprobe 读取音频失败: {stderr[:1000]}")
    data = json.loads(stdout)
    streams = data.get("streams", [])
    if not streams:
        return None
    stream = streams[0]
    duration_value = data.get("format", {}).get("duration")
    return {
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "channel_layout": stream.get("channel_layout"),
        "duration_seconds": float(duration_value) if duration_value else None,
    }


def csv_to_fcpxml(settings):
    """
    将包含视频片段数据的CSV文件转换为FCPXML文件。
    """
    ABS_EDITED_VIDEO_PATH = settings["edited_video_path"]
    ABS_SOURCE_VIDEO_FOLDER = settings["source_dir"]
    ABS_OUTPUT_DIR = settings["output_dir"]
    FINAL_SEGMENTS_CSV_FILENAME = settings["final_segments_csv"]
    FCPXML_PROJECT_NAME = settings["project_name"]
    FCPXML_EVENT_NAME = settings["event_name"]
    FRAME_RATE_FLOAT = settings["frame_rate"]

    # 输入CSV文件路径（位于输出目录中）
    csv_filepath = os.path.join(ABS_OUTPUT_DIR, FINAL_SEGMENTS_CSV_FILENAME)

    # 基于FCPXML_PROJECT_NAME生成输出FCPXML文件名
    output_fcpxml_filepath = os.path.join(
        ABS_OUTPUT_DIR, f"{FCPXML_PROJECT_NAME}.fcpxml"
    )

    # 验证必要的路径是否存在
    if not os.path.isfile(ABS_EDITED_VIDEO_PATH):
        raise FileNotFoundError(f"剪辑后的视频文件未找到 '{ABS_EDITED_VIDEO_PATH}'。")
    elif not os.path.isdir(ABS_SOURCE_VIDEO_FOLDER):
        raise FileNotFoundError(f"源视频文件夹 '{ABS_SOURCE_VIDEO_FOLDER}' 不存在。")

    print(f"开始将 '{os.path.basename(csv_filepath)}' 转换为FCPXML...")

    # --- 1. 从剪辑好的视频中动态获取元数据 ---
    print(f"正在从 '{os.path.basename(ABS_EDITED_VIDEO_PATH)}' 获取视频元数据...")
    metadata = get_video_metadata(ABS_EDITED_VIDEO_PATH)
    if not metadata:
        raise RuntimeError("无法获取视频元数据，无法继续生成FCPXML。")

    video_width = metadata.get("width")
    video_height = metadata.get("height")
    if not FRAME_RATE_FLOAT:
        # 如果配置中没有指定帧率，则使用元数据中的平均帧率
        frame_rate_float = metadata.get("avg_frame_rate", 0)
    else:
        # 使用配置中的帧率
        frame_rate_float = FRAME_RATE_FLOAT
        print(f"使用配置中的帧率: {frame_rate_float} fps")

    if not all([video_width, video_height, frame_rate_float]):
        raise RuntimeError(f"获取的视频元数据不完整: {metadata}。")

    frame_duration_str, fd_numerator, fd_denominator = get_fcpxml_time_params(
        frame_rate_float
    )
    if not frame_duration_str:
        raise RuntimeError(f"不支持的帧率 {frame_rate_float}，无法生成FCPXML。")

    print("检测到的视频参数：")
    print(f"  - 分辨率: {video_width}x{video_height}")
    print(f"  - 帧率: {frame_rate_float:.3f} fps")
    print(f"  - FCPXML帧持续时间: {frame_duration_str}")

    # --- 2. 读取并解析CSV文件 ---
    segments = []
    original_video_names = set()
    max_edited_end_frame = 0

    try:
        with open(csv_filepath, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                try:
                    segment = {
                        "edited_start_frame": int(row["edited_start_frame"]),
                        "edited_end_frame": int(row["edited_end_frame"]),
                        "original_video_name": row["original_video_name"],
                    }
                    if not row.get("speed"):
                        raise ValueError("Obsolete segment CSV; rerun Step 4 to generate source time mappings")
                    segment["speed"] = float(row["speed"])
                    segment["source_start_time_ms"] = float(row["source_start_time_ms"])
                    segment["source_end_time_ms"] = float(row["source_end_time_ms"])
                    segment["edited_total_frames"] = int(row.get("edited_total_frames") or 0)
                    segment["time_map"] = parse_time_map(
                        row.get("time_map"), segment["edited_end_frame"] - segment["edited_start_frame"] + 1)
                    segment["frame_sampling"] = row.get("frame_sampling") or "floor"
                    if segment["frame_sampling"] not in ("floor", "frame-blending"):
                        raise ValueError("Unsupported frame sampling")
                    if (not all(math.isfinite(segment[key]) for key in
                                ("speed", "source_start_time_ms", "source_end_time_ms"))
                            or segment["speed"] < 0 or segment["source_start_time_ms"] < 0
                            or segment["source_end_time_ms"] < segment["source_start_time_ms"]):
                        raise ValueError("Invalid retiming metadata")
                    segments.append(segment)
                    original_video_names.add(segment["original_video_name"])
                    if segment["edited_end_frame"] > max_edited_end_frame:
                        max_edited_end_frame = segment["edited_end_frame"]
                except KeyError as e:
                    raise ValueError(f"CSV第 {i + 2} 行缺少预期的列: {e}") from e
                except ValueError as e:
                    raise ValueError(f"CSV第 {i + 2} 行存在无效的片段数据: {e}") from e
    except FileNotFoundError as error:
        raise FileNotFoundError(f"输入CSV文件未找到 '{csv_filepath}'") from error
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"读取CSV文件 '{csv_filepath}' 时出错: {e}") from e

    if not segments:
        raise RuntimeError("CSV文件中没有找到任何片段数据。")

    segments.sort(key=lambda seg: seg["edited_start_frame"])
    previous_end = -1
    for seg in segments:
        if seg["edited_start_frame"] <= previous_end or seg["edited_end_frame"] < seg["edited_start_frame"]:
            raise ValueError("Overlapping or invalid edited frame ranges")
        previous_end = seg["edited_end_frame"]
    seq_total_frames = max(max_edited_end_frame + 1,
                           max(seg.get("edited_total_frames", 0) for seg in segments))
    seq_duration_str = format_time_value(seq_total_frames, fd_numerator, fd_denominator)

    source_files = discover_source_videos(
        ABS_SOURCE_VIDEO_FOLDER, ABS_EDITED_VIDEO_PATH
    )
    source_path_map = {
        os.path.splitext(os.path.basename(path))[0]: path for path in source_files
    }
    missing_sources = sorted(original_video_names - set(source_path_map))
    if missing_sources:
        raise FileNotFoundError(
            "最终片段引用了 source 目录中不存在的视频: "
            + ", ".join(missing_sources)
        )

    audio_metadata = None
    if settings["include_edited_audio"]:
        audio_metadata = get_audio_metadata(ABS_EDITED_VIDEO_PATH)
        if not audio_metadata:
            raise RuntimeError("include_edited_audio=true，但待分析 MV 没有音频流。")
        sequence_seconds = seq_total_frames * fd_numerator / fd_denominator
        audio_duration = audio_metadata["duration_seconds"]
        frame_seconds = fd_numerator / fd_denominator
        if audio_duration is not None and audio_duration + frame_seconds < sequence_seconds:
            raise RuntimeError(
                f"MV 音频时长 {audio_duration:.3f}s 短于时间线 "
                f"{sequence_seconds:.3f}s 超过一帧。"
            )

    # --- 3. 构建FCPXML结构 ---
    fcpxml = ET.Element("fcpxml", version="1.9")

    # ** 资源 (Resources) **
    resources = ET.SubElement(fcpxml, "resources")

    # 格式定义 (共享)
    fcpxml_format_name = (
        f"FFVideoFormat{video_height}p{str(frame_rate_float).replace('.', '')}"
    )
    format_id = "r0"
    ET.SubElement(
        resources,
        "format",
        id=format_id,
        name=fcpxml_format_name,
        width=str(video_width),
        height=str(video_height),
        frameDuration=frame_duration_str,
    )

    # 资产定义 (Assets)
    asset_map = {}  # 用于映射 original_video_name 到 asset_id (r1, r2, ...)
    asset_id_counter = 1
    for name_key in sorted(list(original_video_names)):  # 排序以确保rX ID的一致性
        asset_id = f"r{asset_id_counter}"
        asset_map[name_key] = asset_id

        source_path = source_path_map[name_key]
        asset_filename = os.path.basename(source_path)
        asset_src_path = format_fcpxml_path(source_path)

        source_meta = get_video_metadata(source_path)
        if not source_meta:
            raise RuntimeError(f"Cannot read source metadata: {source_path}")
        source_fd, source_num, source_den = get_fcpxml_time_params(source_meta["avg_frame_rate"])
        source_format = f"format_source_{asset_id_counter}"
        ET.SubElement(resources, "format", id=source_format,
                      width=str(source_meta["width"]), height=str(source_meta["height"]),
                      frameDuration=source_fd)
        duration_seconds = source_meta.get("duration_seconds")
        if not duration_seconds:
            # If the container has no duration, bound the used source range.
            duration_seconds = max((seg["time_map"][-1, 1] if seg["time_map"] is not None
                                    else seg["source_end_time_ms"]) for seg in segments
                                   if seg["original_video_name"] == name_key) / 1000 + source_num / source_den
        duration = Fraction(str(duration_seconds)).limit_denominator(1_000_000)
        asset_duration_str = f"{duration.numerator}/{duration.denominator}s"
        asset = ET.SubElement(
            resources,
            "asset",
            id=asset_id,
            name=asset_filename,
            start=f"0/{fd_denominator}s",  # 资产通常从0开始
            duration=asset_duration_str,
            hasVideo="1",
            format=source_format,
        )
        ET.SubElement(asset, "media-rep", kind="original-media", src=asset_src_path)
        asset_id_counter += 1

    edited_audio_asset_id = None
    if audio_metadata:
        edited_audio_asset_id = f"r{asset_id_counter}"
        edited_filename = os.path.basename(ABS_EDITED_VIDEO_PATH)
        edited_asset = ET.SubElement(
            resources,
            "asset",
            id=edited_audio_asset_id,
            name=edited_filename,
            start=f"0/{fd_denominator}s",
            duration=seq_duration_str,
            hasVideo="1",
            hasAudio="1",
            format=format_id,
            audioSources="1",
            audioChannels=str(audio_metadata["channels"]),
            audioRate=str(audio_metadata["sample_rate"]),
        )
        ET.SubElement(
            edited_asset,
            "media-rep",
            kind="original-media",
            src=format_fcpxml_path(ABS_EDITED_VIDEO_PATH),
        )

    # ** 库 (Library) **
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name=FCPXML_EVENT_NAME)
    project = ET.SubElement(event, "project", name=FCPXML_PROJECT_NAME)

    # ** 序列 (Sequence) **
    sequence = ET.SubElement(
        project,
        "sequence",
        tcStart=f"0/{fd_denominator}s",
        duration=seq_duration_str,
        tcFormat="NDF",  # Non-Drop Frame timecode
        format=format_id,  # 引用格式定义 "r0"
    )

    spine = ET.SubElement(sequence, "spine")

    # ** 从CSV片段数据创建资产片段 (Asset-Clips) **
    first_asset_clip = None
    next_frame = 0
    def rational_seconds(milliseconds):
        value = (Fraction(str(milliseconds)) / 1000).limit_denominator(1_000_000_000)
        return f"{value.numerator}/{value.denominator}s"

    for i, seg_data in enumerate(segments):
        clip_name = f"clip{i + 1:04d}"  # 例如: clip0001, clip0002

        # offset: 片段在时间线上的起始位置（单位：帧）
        offset_str = format_time_value(
            seg_data["edited_start_frame"], fd_numerator, fd_denominator
        )

        # duration: 片段自身的时长（单位：帧）
        clip_frame_duration = (
            seg_data["edited_end_frame"] - seg_data["edited_start_frame"] + 1
        )
        duration_str = format_time_value(
            clip_frame_duration, fd_numerator, fd_denominator
        )

        start_str = "0s"
        if seg_data["edited_start_frame"] > next_frame:
            ET.SubElement(spine, "gap", name="Unmatched", start="0s",
                          offset=format_time_value(next_frame, fd_numerator, fd_denominator),
                          duration=format_time_value(seg_data["edited_start_frame"] - next_frame,
                                                     fd_numerator, fd_denominator))
        next_frame = seg_data["edited_end_frame"] + 1
        asset_ref_id = asset_map[seg_data["original_video_name"]]

        # Anchor music to an unretimed container.  A timeMap also remaps the
        # offsets of its anchored children (Apple's timeMap specification).
        parent = spine
        audio_anchor = None
        if edited_audio_asset_id and first_asset_clip is None:
            audio_anchor = ET.SubElement(spine, "clip", name=clip_name, offset=offset_str,
                                         start="0s", duration=duration_str, format=format_id)
            parent, offset_str = audio_anchor, "0s"
        asset_clip = ET.SubElement(
            parent,
            "asset-clip",
            name=clip_name,
            ref=asset_ref_id,  # 引用资源ID
            offset=offset_str,
            duration=duration_str,
            start=start_str,
            tcFormat="NDF",
            format=format_id,  # 引用格式定义 "r0"
            enabled="1",
            srcEnable="video",
        )
        time_map = ET.SubElement(asset_clip, "timeMap", frameSampling=seg_data["frame_sampling"])
        points = seg_data.get("time_map")
        if points is None:
            points = [(0, seg_data["source_start_time_ms"]),
                      (clip_frame_duration, seg_data["source_end_time_ms"])]
        for local_frame, source_ms in points:
            ET.SubElement(time_map, "timept",
                          time=format_time_value(int(local_frame), fd_numerator, fd_denominator),
                          value=rational_seconds(float(source_ms)), interp="linear")
        if first_asset_clip is None:
            first_asset_clip = audio_anchor if audio_anchor is not None else asset_clip

        # 添加变换调整信息（通常保持默认值）
        ET.SubElement(
            asset_clip, "adjust-transform", scale="1 1", anchor="0 0", position="0 0"
        )

    if next_frame < seq_total_frames:
        ET.SubElement(spine, "gap", name="Unmatched", start="0s",
                      offset=format_time_value(next_frame, fd_numerator, fd_denominator),
                      duration=format_time_value(seq_total_frames - next_frame, fd_numerator, fd_denominator))

    if edited_audio_asset_id and first_asset_clip is not None:
        ET.SubElement(
            first_asset_clip,
            "asset-clip",
            name=f"{os.path.splitext(os.path.basename(ABS_EDITED_VIDEO_PATH))[0]} Audio",
            ref=edited_audio_asset_id,
            lane="-1",
            offset=format_time_value(-segments[0]["edited_start_frame"], fd_numerator, fd_denominator),
            start=f"0/{fd_denominator}s",
            duration=seq_duration_str,
            tcFormat="NDF",
            format=format_id,
            enabled="1",
            srcEnable="audio",
            audioRole="music",
        )

    # --- 4. 输出FCPXML文件 ---
    # 添加 FCPXML 的 DOCTYPE 声明
    doctype_str = "<!DOCTYPE fcpxml>\n"

    # 使用minidom进行格式化（pretty print），以获得带缩进的易读XML
    rough_string = ET.tostring(fcpxml, encoding="utf-8", method="xml")
    reparsed = minidom.parseString(rough_string)
    pretty_xml_str = reparsed.toprettyxml(indent="    ", encoding="UTF-8").decode(
        "utf-8"
    )

    # minidom会默认添加一个XML声明，我们移除它，以便使用我们自己的
    if pretty_xml_str.startswith("<?xml"):
        pretty_xml_str = pretty_xml_str.split("?>", 1)[1].lstrip()

    final_xml_content = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n{doctype_str}{pretty_xml_str}'
    )

    try:
        with open(output_fcpxml_filepath, "w", encoding="utf-8") as f:
            f.write(final_xml_content)
        print(f"成功生成FCPXML文件: '{output_fcpxml_filepath}'")
    except IOError as e:
        raise RuntimeError(
            f"写入FCPXML文件 '{output_fcpxml_filepath}' 时出错: {e}"
        ) from e


if __name__ == "__main__":
    from settings import load_settings

    csv_to_fcpxml(load_settings("config.toml"))
