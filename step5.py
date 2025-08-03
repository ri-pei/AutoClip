import os
import math
import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom  # 用于格式化输出（pretty printing）
from common import (
    get_video_metadata,
    ABS_EDITED_VIDEO_PATH,
    ABS_SOURCE_VIDEO_FOLDER,
    ABS_OUTPUT_DIR,
)
from config import (
    FINAL_SEGMENTS_CSV_FILENAME,
    FCPXML_PROJECT_NAME,
    FCPXML_EVENT_NAME,
    FRAME_RATE_FLOAT,
)

# --- Step 4 Specific Configuration ---
# 该脚本假设所有ABS_SOURCE_VIDEO_FOLDER中的视频文件扩展名为.mp4
ASSET_PLACEHOLDER_DURATION_HOURS = 2  # 2 hours，应长于最终生成视频
# --- END Configuration ---


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
    # 针对标准NTSC和PAL帧率进行精确匹配
    if math.isclose(frame_rate_float, 23.976, rel_tol=1e-3):
        return "1001/24000s", 1001, 24000  # 24000 / 1.001
    elif math.isclose(frame_rate_float, 29.97, rel_tol=1e-3):
        return "1001/30000s", 1001, 30000  # 30000 / 1.001
    elif math.isclose(frame_rate_float, 59.94, rel_tol=1e-3):
        return "1001/60000s", 1001, 60000  # 60000 / 1.001
    # 针对整数帧率
    elif math.isclose(frame_rate_float, 24.0):
        return "1/24s", 1, 24
    elif math.isclose(frame_rate_float, 25.0):
        return "1/25s", 1, 25
    elif math.isclose(frame_rate_float, 30.0):
        return "1/30s", 1, 30
    elif math.isclose(frame_rate_float, 50.0):
        return "1/50s", 1, 50
    elif math.isclose(frame_rate_float, 60.0):
        return "1/60s", 1, 60
    else:
        # 对于不常见的帧率，尝试使用其分数表示
        # 注意：这可能不被所有编辑器完美支持
        fr_fraction = frame_rate_float.as_integer_ratio()
        print(
            f"警告：检测到非标准帧率 {frame_rate_float}。"
            f"将使用分数表示 {fr_fraction[1]}/{fr_fraction[0]}s，兼容性未知。"
        )
        return f"{fr_fraction[1]}/{fr_fraction[0]}s", fr_fraction[1], fr_fraction[0]


def format_time_value(frames, fd_numerator, fd_denominator):
    """将帧数格式化为FCPXML的时间值字符串 'value/denominator s'。"""
    # 此处的 'value' 是 帧数 * 帧持续时间的分子 (frames * fd_numerator)
    return f"{int(frames * fd_numerator)}/{fd_denominator}s"


def format_fcpxml_path(source_folder_path, filename):
    """将本地文件路径格式化为FCPXML所需的 'file://' URI。"""
    # 使用os.path.join确保路径分隔符正确
    full_path_str = os.path.join(source_folder_path, filename)

    # 转换为绝对路径以确保URI的有效性
    abs_path = os.path.abspath(full_path_str)

    # 将Windows路径的反斜杠'\'替换为正斜杠'/'
    posix_path = abs_path.replace(os.path.sep, "/")

    # 为Windows驱动器号路径（如 C:/...）添加前导斜杠
    if ":" in posix_path and posix_path[1] == ":":  # 例如 C:/...
        posix_path = "/" + posix_path

    # FCPXML期望的格式是 file://localhost/path/to/file
    # Resolve等软件通常也能处理 file:///path/to/file 格式
    return f"file://localhost{posix_path}"


def csv_to_fcpxml():
    """
    将包含视频片段数据的CSV文件转换为FCPXML文件。
    """
    # 输入CSV文件路径（位于输出目录中）
    csv_filepath = os.path.join(ABS_OUTPUT_DIR, FINAL_SEGMENTS_CSV_FILENAME)

    # 基于FCPXML_PROJECT_NAME生成输出FCPXML文件名
    output_fcpxml_filepath = os.path.join(
        ABS_OUTPUT_DIR, f"{FCPXML_PROJECT_NAME}.fcpxml"
    )

    # 验证必要的路径是否存在
    if not os.path.isfile(ABS_EDITED_VIDEO_PATH):
        print(f"错误: 剪辑后的视频文件未找到 '{ABS_EDITED_VIDEO_PATH}'。")
        print("请检查 config.py 中的 EDITED_VIDEO_FILENAME 配置。")
        return
    elif not os.path.isdir(ABS_SOURCE_VIDEO_FOLDER):
        print(f"错误: 源视频文件夹 '{ABS_SOURCE_VIDEO_FOLDER}' 不存在。")
        print("请检查 config.py 中的 SOURCE_VIDEO_FOLDER 配置。")
        return

    print(f"开始将 '{os.path.basename(csv_filepath)}' 转换为FCPXML...")

    # --- 1. 从剪辑好的视频中动态获取元数据 ---
    print(f"正在从 '{os.path.basename(ABS_EDITED_VIDEO_PATH)}' 获取视频元数据...")
    metadata = get_video_metadata(ABS_EDITED_VIDEO_PATH)
    if not metadata:
        print("错误：无法获取视频元数据，无法继续生成FCPXML。")
        return

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
        print(f"错误：获取的元数据不完整: {metadata}。请检查视频文件。")
        return

    frame_duration_str, fd_numerator, fd_denominator = get_fcpxml_time_params(
        frame_rate_float
    )
    if not frame_duration_str:
        print(f"错误：不支持的帧率 {frame_rate_float}，无法生成FCPXML。")
        return

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
                        "original_start_frame": int(row["original_start_frame"]),
                    }
                    segments.append(segment)
                    original_video_names.add(segment["original_video_name"])
                    if segment["edited_end_frame"] > max_edited_end_frame:
                        max_edited_end_frame = segment["edited_end_frame"]
                except KeyError as e:
                    print(f"CSV错误：第 {i + 2} 行缺少预期的列: {e}")
                    return
                except ValueError as e:
                    print(f"CSV错误：第 {i + 2} 行存在无效的整数值: {e}")
                    return
    except FileNotFoundError:
        print(f"错误：输入CSV文件未找到 '{csv_filepath}'")
        return
    except Exception as e:
        print(f"读取CSV文件 '{csv_filepath}' 时出错: {e}")
        return

    if not segments:
        print("CSV文件中没有找到任何片段数据，操作中止。")
        return

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
    # 基于配置计算占位时长的分子乘积
    # 总帧数 = 时长(秒) * 帧率 = (小时 * 3600) * (分母/分子) -- 这是错的
    # 总帧数 = 时长(秒) * 帧率(fps) = (小时 * 3600) * frame_rate_float
    total_placeholder_frames = (
        ASSET_PLACEHOLDER_DURATION_HOURS * 3600 * frame_rate_float
    )
    asset_duration_numerator_product = int(total_placeholder_frames * fd_numerator)

    for name_key in sorted(list(original_video_names)):  # 排序以确保rX ID的一致性
        asset_id = f"r{asset_id_counter}"
        asset_map[name_key] = asset_id

        asset_filename = f"{name_key}.mp4"  # 假设扩展名为.mp4
        asset_src_path = format_fcpxml_path(ABS_SOURCE_VIDEO_FOLDER, asset_filename)

        # 资产时长：使用基于配置计算出的占位符
        asset_duration_str = f"{asset_duration_numerator_product}/{fd_denominator}s"

        asset = ET.SubElement(
            resources,
            "asset",
            id=asset_id,
            name=asset_filename,
            start=f"0/{fd_denominator}s",  # 资产通常从0开始
            duration=asset_duration_str,
            hasVideo="1",
            format=format_id,
        )
        ET.SubElement(asset, "media-rep", kind="original-media", src=asset_src_path)
        asset_id_counter += 1

    # ** 库 (Library) **
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name=FCPXML_EVENT_NAME)
    project = ET.SubElement(event, "project", name=FCPXML_PROJECT_NAME)

    # ** 序列 (Sequence) **
    # 序列时长: (最后一个片段的结束帧 + 1) * 帧时长
    seq_total_frames = max_edited_end_frame + 1
    seq_duration_str = format_time_value(seq_total_frames, fd_numerator, fd_denominator)

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

        # start: 片段在源素材中的入点（单位：帧）
        start_str = format_time_value(
            seg_data["original_start_frame"], fd_numerator, fd_denominator
        )

        asset_ref_id = asset_map[seg_data["original_video_name"]]

        asset_clip = ET.SubElement(
            spine,
            "asset-clip",
            name=clip_name,
            ref=asset_ref_id,  # 引用资源ID
            offset=offset_str,
            duration=duration_str,
            start=start_str,
            tcFormat="NDF",
            format=format_id,  # 引用格式定义 "r0"
            enabled="1",
        )

        # 添加变换调整信息（通常保持默认值）
        ET.SubElement(
            asset_clip, "adjust-transform", scale="1 1", anchor="0 0", position="0 0"
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
        print(f"写入FCPXML文件 '{output_fcpxml_filepath}' 时出错: {e}")


if __name__ == "__main__":

    csv_to_fcpxml()
