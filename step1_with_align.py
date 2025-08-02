import os
import glob
import shutil
import cv2  # OpenCV for image processing and transform estimation
import numpy as np  # For numerical operations with OpenCV
from config import (
    EDITED_VIDEO_FILENAME,
    SOURCE_VIDEO_FOLDER,
    REF_ORIGINAL_FRAME_PATH,
    REF_EDITED_FRAME_PATH,
    MASK_RECT,
)
from common import (
    get_video_metadata,
    run_command,
    format_timestamp_from_seconds,
    get_frame_timestamps_map_json,
    ABS_WORKING_DIR,
    ABS_EDITED_VIDEO_PATH,
    ABS_OUTPUT_DIR,
    ABS_SOURCE_VIDEO_FOLDER,
    ABS_REF_ORIGINAL_FRAME_PATH,
    ABS_REF_EDITED_FRAME_PATH,
    ABS_USER_COLOR_LUT_PATH,
)


# --- 内部常量 ---
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv")
MIN_MATCH_COUNT_GEO = 10  # SIFT/ORB匹配的最小特征点数
# --- END 内部常量 ---

# 全局变量存储计算出的变换参数
GEOMETRIC_TRANSFORM_PARAMS = None  # Calculated once


def estimate_geometric_transform_from_refs(ref_original_path, ref_edited_path):
    """
    从参考原始帧和参考剪辑帧估算几何变换（裁剪和缩放）。
    使用特征点匹配和单应性矩阵，返回适用于ffmpeg crop/scale的参数字典，失败时返回None。
    """
    print(
        f"  Estimating geometric transform: '{os.path.basename(ref_original_path)}' "
        f"vs '{os.path.basename(ref_edited_path)}'"
    )
    img_orig = cv2.imread(ref_original_path)
    img_edit = cv2.imread(ref_edited_path)

    if img_orig is None:
        print(
            f"    Error: Could not read original reference image: {ref_original_path}"
        )
        return None
    if img_edit is None:
        print(f"    Error: Could not read edited reference image: {ref_edited_path}")
        return None

    h_orig, w_orig = img_orig.shape[:2]
    h_edit, w_edit = img_edit.shape[:2]

    try:  # Try SIFT first (more robust to scale)
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
        print("    Using SIFT for feature detection.")
    except AttributeError:
        print(
            (
                "    SIFT not available (try 'pip install opencv-contrib-python'). "
                "Falling back to ORB."
            )
        )
        detector = cv2.ORB_create(
            nfeatures=2000
        )  # More features for potentially better matching
        norm_type = cv2.NORM_HAMMING

    kp_orig, des_orig = detector.detectAndCompute(img_orig, None)
    kp_edit, des_edit = detector.detectAndCompute(img_edit, None)

    if des_orig is None or len(kp_orig) < MIN_MATCH_COUNT_GEO:
        print(
            (
                "    Error: Not enough keypoints/descriptors in original reference "
                f"({len(kp_orig) if kp_orig is not None else 0})."
            )
        )
        return None
    if des_edit is None or len(kp_edit) < MIN_MATCH_COUNT_GEO:
        print(
            (
                "    Error: Not enough keypoints/descriptors in edited reference "
                f"({len(kp_edit) if kp_edit is not None else 0})."
            )
        )
        return None

    # Match descriptors
    if norm_type == cv2.NORM_L2:  # SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:  # ORB
        matcher = cv2.BFMatcher(
            norm_type, crossCheck=False
        )  # Use knnMatch, so crossCheck=False

    matches = matcher.knnMatch(des_orig, des_edit, k=2)

    good_matches = []
    if matches:
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                    good_matches.append(m)

    print(
        f"    Found {len(good_matches)} good matches "
        f"(min required: {MIN_MATCH_COUNT_GEO})."
    )
    if len(good_matches) < MIN_MATCH_COUNT_GEO:
        print("    Not enough good matches to estimate transform reliably.")
        # 如果需要调试，可以在这里绘制匹配结果
        # img_matches = cv2.drawMatches(
        #     img_orig, kp_orig, img_edit, kp_edit, good_matches, None,
        #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        # )
        # cv2.imwrite(
        #     os.path.join(
        #         os.path.dirname(ref_original_path),
        #         "debug_matches.png"
        #     ),
        #     img_matches
        # )
        # print("    (已保存 debug_matches.png，可用于检查匹配效果)")
        return None

    src_pts = np.float32([kp_orig[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([kp_edit[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )

    # Estimate Homography (more general than Affine, can handle some perspective)
    M_homo, mask_homo = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0
    )

    if M_homo is None:
        print("    Could not estimate Homography matrix.")
        return None

    # Now, we need to figure out the crop window in the original image that corresponds
    # to the entire edited image. We do this by projecting the corners of the
    # *edited image* back into the *original image's coordinate space* using the
    # inverse of the homography matrix.
    try:
        M_inv_homo = np.linalg.inv(M_homo)
    except np.linalg.LinAlgError:
        print("    Error: Homography matrix is singular, cannot invert.")
        return None

    corners_edited_frame = np.float32(
        [[0, 0], [w_edit - 1, 0], [w_edit - 1, h_edit - 1], [0, h_edit - 1]]
    ).reshape(-1, 1, 2)

    projected_corners_in_orig = cv2.perspectiveTransform(
        corners_edited_frame, M_inv_homo
    )
    if projected_corners_in_orig is None:
        print("    Error during perspectiveTransform of corners.")
        return None

    # Get the bounding box of these projected corners
    x_coords = projected_corners_in_orig[:, 0, 0]
    y_coords = projected_corners_in_orig[:, 0, 1]

    crop_x = int(round(np.min(x_coords)))
    crop_y = int(round(np.min(y_coords)))
    crop_w = int(round(np.max(x_coords) - crop_x))
    crop_h = int(round(np.max(y_coords) - crop_y))

    # Sanity checks and clamping for crop parameters
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)
    if crop_x + crop_w > w_orig:
        crop_w = w_orig - crop_x
    if crop_y + crop_h > h_orig:
        crop_h = h_orig - crop_y

    if crop_w <= 0 or crop_h <= 0:
        print(f"    错误：估算的裁剪尺寸无效 (w={crop_w}, h={crop_h})。")
        return None

    # The ffmpeg crop filter takes: crop=width:height:x:y
    # The ffmpeg scale filter will then scale this cropped region to the dimensions
    # of the edited video. (which is h_edit, w_edit)
    transform_params = {
        "crop_w": crop_w,
        "crop_h": crop_h,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "scale_target_w": w_edit,  # The cropped part should be scaled to this width
        "scale_target_h": h_edit,  # and this height
    }
    print("    Estimated geometric transform for ffmpeg:")
    print(
        f"      Crop: w={crop_w}, h={crop_h}, x={crop_x}, y={crop_y} "
        f"(from original {w_orig}x{h_orig})"
    )
    print(f"      Scale to: w={w_edit}, h={h_edit}")
    return transform_params


def process_video_for_frames(
    video_path,
    video_name_no_ext,
    main_output_folder,
    is_edited_video_flag,
    final_output_resolution,
    mask_rect_config,
    geom_transform_to_apply,
    color_lut_to_apply,
):
    """
    处理视频以提取帧，并应用必要的几何变换、颜色校正和遮罩。
    video_path: 视频文件的完整路径
    video_name_no_ext: 视频文件名（不带扩展名）
    main_output_folder: 主输出文件夹路径
    is_edited_video_flag: 布尔值，指示是否为编辑后的视频
    final_output_resolution: 字典，包含编辑后视频的分辨率（width, height）
    mask_rect_config: 遮罩矩形配置，格式为 (x, y, w, h) 或 None
    geom_transform_to_apply: 几何变换参数
    color_lut_to_apply: 颜色LUT文件路径或None
    """
    video_specific_output_dir = os.path.join(main_output_folder, video_name_no_ext)
    frames_output_dir = os.path.join(video_specific_output_dir, "frames")

    if os.path.exists(frames_output_dir):
        final_name_pattern = os.path.join(
            frames_output_dir, f"{video_name_no_ext}_frame_*_time_*.png"
        )
        if glob.glob(final_name_pattern):
            print(f"  Frames for '{video_name_no_ext}' seem to exist. Skipping.")
            return True
        else:
            print(
                f"  Frame dir '{frames_output_dir}' exists but no final files. 清理中"
            )
            shutil.rmtree(frames_output_dir)

    os.makedirs(frames_output_dir, exist_ok=True)
    temp_ffmpeg_output_dir = os.path.join(
        video_specific_output_dir, "temp_ffmpeg_frames"
    )
    if os.path.exists(temp_ffmpeg_output_dir):
        shutil.rmtree(temp_ffmpeg_output_dir)
    os.makedirs(temp_ffmpeg_output_dir, exist_ok=True)

    # print(f"  Processing video '{video_name_no_ext}':")
    vf_options = []

    # --- 1. Apply Geometric Transform (Crop and Scale) - ONLY FOR ORIGINAL VIDEOS ---
    if not is_edited_video_flag and geom_transform_to_apply:
        gt = geom_transform_to_apply
        vf_options.append(
            f"crop={gt['crop_w']}:{gt['crop_h']}:{gt['crop_x']}:{gt['crop_y']}"
        )
        # Scale the (now cropped) video to match the edited video's resolution
        vf_options.append(f"scale={gt['scale_target_w']}:{gt['scale_target_h']}")
    elif is_edited_video_flag:
        # For the EDITED video itself, we only scale it if its resolution differs
        # from its own reported metadata (which defines final_output_resolution).
        # This situation should be rare if final_output_resolution is derived from
        # edited video's metadata.
        current_meta = get_video_metadata(
            video_path
        )  # Get actual current res of this file
        if current_meta and (
            current_meta["width"] != final_output_resolution["width"]
            or current_meta["height"] != final_output_resolution["height"]
        ):
            print(
                f"    Scaling edited video '{video_name_no_ext}' "
                f"to its defined final output resolution."
            )
            vf_options.append(
                f"scale={final_output_resolution['width']}:"
                f"{final_output_resolution['height']}"
            )

    # --- 2. Apply Color Transformation - ONLY FOR ORIGINAL VIDEOS ---
    if not is_edited_video_flag:
        if color_lut_to_apply and os.path.exists(color_lut_to_apply):
            # Ensure path is suitable for ffmpeg (e.g., escape special chars if any -
            # simplified here)
            lut_path_escaped = color_lut_to_apply.replace(
                "\\", "/"
            )  # Basic path normalization
            vf_options.append(f"lut3d=file='{lut_path_escaped}'")
            # print(f"    Applying 3D LUT: {color_lut_to_apply}")
            # else if ref_frame_for_histmatch:
            # Direct histmatch is complex for ffmpeg CLI here
            # print(
            #     "    (Skipping histmatch for now - direct use with image ref is "
            #     "complex in simple -vf chain)"
            # )
            pass

    # --- 3. Apply Masking (for ALL videos, after all other transforms) ---
    # The frame dimensions at this point should be `final_output_resolution`.
    if mask_rect_config:
        x, y, w, h = mask_rect_config
        # Mask coordinates are relative to the frame *after* all previous transforms.
        img_w_mask, img_h_mask = (
            final_output_resolution["width"],
            final_output_resolution["height"],
        )

        valid_mask = False
        if w > 0 and h > 0:
            x1_m, y1_m = max(0, x), max(0, y)
            eff_w_m, eff_h_m = min(img_w_mask - x1_m, w), min(img_h_mask - y1_m, h)
            if eff_w_m > 0 and eff_h_m > 0:
                vf_options.append(
                    (
                        f"drawbox=x={x1_m}:y={y1_m}:w={eff_w_m}:h={eff_h_m}:"
                        f"color=black:t=fill"
                    )
                )
                valid_mask = True
        if not valid_mask and mask_rect_config:
            print(
                (
                    f"    Warning: Mask {mask_rect_config} is invalid for res "
                    f"{img_w_mask}x{img_h_mask}. No mask applied."
                )
            )

    # --- Construct and Run FFmpeg Command ---
    ffmpeg_filter_string = ",".join(vf_options)
    temp_frame_pattern = os.path.join(temp_ffmpeg_output_dir, "ffmpeg_frame_%07d.png")

    extract_cmd = ["ffmpeg", "-y", "-i", video_path, "-an", "-vsync", "vfr"]
    if ffmpeg_filter_string:
        extract_cmd.extend(["-vf", ffmpeg_filter_string])
    extract_cmd.extend(["-q:v", "2", "-start_number", "0", temp_frame_pattern])

    # print(f"    Running ffmpeg for '{video_name_no_ext}'...")
    # if ffmpeg_filter_string:
    #     filter_str = ffmpeg_filter_string[:200]
    #     ellipsis = '...' if len(ffmpeg_filter_string) > 200 else ''
    #     print(f"      Filter: {filter_str}{ellipsis}")

    _, stderr_ffmpeg, process_ffmpeg = run_command(extract_cmd)

    extracted_temp_files = sorted(
        glob.glob(os.path.join(temp_ffmpeg_output_dir, "ffmpeg_frame_*.png"))
    )

    if process_ffmpeg.returncode != 0 or not extracted_temp_files:
        print(
            (
                f"    FFmpeg failed or produced no frames for {video_name_no_ext} "
                f"(Code: {process_ffmpeg.returncode})."
            )
        )
        if stderr_ffmpeg:
            print(
                f"      FFmpeg STDERR:\n{stderr_ffmpeg[:1000]}\n"
            )  # Print a good chunk of stderr
        if os.path.exists(temp_ffmpeg_output_dir):
            shutil.rmtree(temp_ffmpeg_output_dir)
        return False
    # print(f"    FFmpeg extracted {len(extracted_temp_files)} frames.")

    # --- Rename frames ---
    # print(f"    Fetching timestamps & renaming {len(extracted_temp_files)} frames...")
    frame_ts_map = get_frame_timestamps_map_json(video_path)
    renamed_count = 0
    # 在这里,遍历所有提取的临时文件并重命名它们
    # 从frame_ts_map获取每一帧的时间戳
    # 将临时文件重命名为带有时间戳信息的最终文件名
    for i, temp_processed_path in enumerate(extracted_temp_files):
        timestamp_sec = frame_ts_map.get(i, 0.0)  # Default to 0.0s if TS not found
        formatted_ts = format_timestamp_from_seconds(timestamp_sec)
        final_frame_name = f"{video_name_no_ext}_frame_{i:07d}_time_{formatted_ts}.png"
        final_frame_path = os.path.join(frames_output_dir, final_frame_name)
        try:
            os.rename(temp_processed_path, final_frame_path)
            renamed_count += 1
        except Exception as e:
            print(
                f"    Error renaming {temp_processed_path} to {final_frame_path}: {e}"
            )
    # print(f"    Renamed {renamed_count} frames.")

    if os.path.exists(temp_ffmpeg_output_dir):
        try:
            if not os.listdir(temp_ffmpeg_output_dir):
                os.rmdir(temp_ffmpeg_output_dir)
            # else:
            # print(f"    Warning: Temp dir '{temp_ffmpeg_output_dir}' not empty.")
        except OSError:
            pass  # Ignore error if dir is not empty or other issue

    if renamed_count == 0 and len(extracted_temp_files) > 0:
        print(
            f"    WARNING: Extracted frames but NONE renamed for {video_name_no_ext}."
        )
        return False
    return True


def main_step1():
    """
    主函数，执行步骤1：提取帧并应用必要的几何变换和颜色校正。
    处理所有视频文件，使用用户指定的编辑后视频作为参考。
    """

    print("开始步骤1: 提取帧和准备(包含变换估计)...")
    # 打印预处理后的路径以供用户确认
    print(f"工作目录: {ABS_WORKING_DIR}")
    print(f"源视频文件夹: {ABS_SOURCE_VIDEO_FOLDER or '未设置'}")
    print(f"输出目录: {ABS_OUTPUT_DIR}")
    print("-" * 30)
    print(f"编辑后的视频文件: {ABS_EDITED_VIDEO_PATH}")
    if MASK_RECT:
        print(f"遮罩矩形配置: {MASK_RECT}")
    else:
        print("遮罩矩形: 未设置.")

    print(f"参考-原始帧: {ABS_REF_ORIGINAL_FRAME_PATH or '未设置'}")
    print(f"参考-编辑后帧: {ABS_REF_EDITED_FRAME_PATH or '未设置'}")
    if ABS_USER_COLOR_LUT_PATH:
        print(f"用户颜色LUT文件: {ABS_USER_COLOR_LUT_PATH}")
    else:
        print("用户颜色LUT文件: 未设置.")
    print("-" * 30)

    global GEOMETRIC_TRANSFORM_PARAMS  # Use the global variable

    # 输出目录已在预处理部分解析，此处确保它存在
    os.makedirs(ABS_OUTPUT_DIR, exist_ok=True)

    # 检查必须的编辑后视频文件是否存在
    if not ABS_EDITED_VIDEO_PATH or not os.path.exists(ABS_EDITED_VIDEO_PATH):
        print(
            f"致命错误: 编辑后的视频 '{EDITED_VIDEO_FILENAME}' "
            f"在解析的路径 '{ABS_EDITED_VIDEO_PATH}' 未找到。"
        )
        return

    print(
        f"正在处理编辑后的视频: {os.path.basename(ABS_EDITED_VIDEO_PATH)} "
        "以确定最终输出分辨率..."
    )
    edited_metadata = get_video_metadata(ABS_EDITED_VIDEO_PATH)
    if (
        not edited_metadata
        or "width" not in edited_metadata
        or "height" not in edited_metadata
    ):
        print(
            f"致命错误: 无法获取编辑后视频 '{os.path.basename(ABS_EDITED_VIDEO_PATH)}' 的元数据。"
        )
        return

    final_output_resolution = {
        "width": edited_metadata["width"],
        "height": edited_metadata["height"],
    }
    print(
        f"最终输出分辨率 (来自编辑后视频): "
        f"{final_output_resolution['width']}x{final_output_resolution['height']}"
    )

    # --- 估算几何变换（仅执行一次）---
    # 路径已被预处理为绝对路径
    if not (
        ABS_REF_ORIGINAL_FRAME_PATH
        and os.path.exists(ABS_REF_ORIGINAL_FRAME_PATH)
        and ABS_REF_EDITED_FRAME_PATH
        and os.path.exists(ABS_REF_EDITED_FRAME_PATH)
    ):
        print("警告: 用于几何变换的参考帧未找到:")
        if not ABS_REF_ORIGINAL_FRAME_PATH or not os.path.exists(
            ABS_REF_ORIGINAL_FRAME_PATH
        ):
            print(
                f"  缺失: {ABS_REF_ORIGINAL_FRAME_PATH} (来自设置: {REF_ORIGINAL_FRAME_PATH})"
            )
        if not ABS_REF_EDITED_FRAME_PATH or not os.path.exists(
            ABS_REF_EDITED_FRAME_PATH
        ):
            print(
                f"  缺失: {ABS_REF_EDITED_FRAME_PATH} (来自设置: {REF_EDITED_FRAME_PATH})"
            )
        print("  将跳过对源素材的几何变换（裁切/缩放）。")
        GEOMETRIC_TRANSFORM_PARAMS = None
    else:
        GEOMETRIC_TRANSFORM_PARAMS = estimate_geometric_transform_from_refs(
            ABS_REF_ORIGINAL_FRAME_PATH, ABS_REF_EDITED_FRAME_PATH
        )
        if not GEOMETRIC_TRANSFORM_PARAMS:
            print("  几何变换估算失败，将不会对源素材进行特定的裁切/缩放。")

    # --- 检查用户提供的颜色LUT文件 ---
    # 路径已被预处理为绝对路径，只需检查文件是否存在
    final_color_lut_path = None
    if ABS_USER_COLOR_LUT_PATH:
        if os.path.exists(ABS_USER_COLOR_LUT_PATH):
            print(f"  将使用颜色校正LUT文件: {ABS_USER_COLOR_LUT_PATH}")
            final_color_lut_path = ABS_USER_COLOR_LUT_PATH
        else:
            print(f"警告: 用户指定的颜色LUT文件未找到: " f"{ABS_USER_COLOR_LUT_PATH}")
            print("  将跳过颜色LUT的应用。")

    # --- 查找所有要处理的视频文件 (已修正逻辑) ---
    source_video_files = []
    if ABS_SOURCE_VIDEO_FOLDER and os.path.isdir(ABS_SOURCE_VIDEO_FOLDER):
        print(f"\n正在源视频文件夹中搜索视频: {ABS_SOURCE_VIDEO_FOLDER}")
        for ext in VIDEO_EXTENSIONS:
            # 使用 recursive=True 在子文件夹中查找
            pattern = os.path.join(ABS_SOURCE_VIDEO_FOLDER, "**", f"*{ext.lower()}")
            source_video_files.extend(glob.glob(pattern, recursive=True))
            # 兼容大写扩展名
            pattern_upper = os.path.join(
                ABS_SOURCE_VIDEO_FOLDER, "**", f"*{ext.upper()}"
            )
            source_video_files.extend(glob.glob(pattern_upper, recursive=True))
    elif SOURCE_VIDEO_FOLDER:
        print(f"警告: 源视频文件夹 '{SOURCE_VIDEO_FOLDER}' 未找到或不是一个目录。")

    # 最终处理列表 = 编辑后的视频 + 所有源视频 (去重)
    all_videos_to_process = {os.path.normpath(ABS_EDITED_VIDEO_PATH)}
    all_videos_to_process.update([os.path.normpath(p) for p in source_video_files])

    # 转换为排序后的列表以保证处理顺序一致
    sorted_videos_list = sorted(list(all_videos_to_process))

    if not sorted_videos_list:
        print("No video files found.")
        return

    print(f"\n总共找到 {len(sorted_videos_list)} 个视频文件进行处理。")

    for video_full_path in sorted_videos_list:
        video_file_name_with_ext = os.path.basename(video_full_path)
        video_name_no_ext, _ = os.path.splitext(video_file_name_with_ext)

        print(f"\n--- 正在处理: {video_file_name_with_ext} ---")
        # 使用 os.path.normpath 确保跨平台路径比较的可靠性
        is_edited = os.path.normpath(video_full_path) == os.path.normpath(
            ABS_EDITED_VIDEO_PATH
        )

        current_geom_transform = None
        current_color_lut = None
        if not is_edited:  # 仅对源素材应用特殊变换
            current_geom_transform = GEOMETRIC_TRANSFORM_PARAMS
            current_color_lut = final_color_lut_path

        process_video_for_frames(
            video_path=video_full_path,
            video_name_no_ext=video_name_no_ext,
            main_output_folder=ABS_OUTPUT_DIR,
            is_edited_video_flag=is_edited,
            final_output_resolution=final_output_resolution,
            mask_rect_config=MASK_RECT,
            geom_transform_to_apply=current_geom_transform,
            color_lut_to_apply=current_color_lut,
        )

    print("\n\n步骤 1 (帧提取与变换) 已完成。")
    print(f"所有输出的帧位于以下目录的子文件夹中: {ABS_OUTPUT_DIR}")


if __name__ == "__main__":

    try:
        # Ensure OpenCV Contrib is installed if SIFT is desired.
        # `pip install opencv-python opencv-contrib-python numpy`
        main_step1()
    except FileNotFoundError as e:  # More specific for ffmpeg/ffprobe
        if "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower():
            print(
                "--------------------------------------------------------------------"
            )
            print("错误: FFMPEG/FFPROBE 未找到。请确保已安装它们，")
            print("并将其添加至系统的 PATH 环境变量。")
            print("下载地址: https://ffmpeg.org/download.html")
            print(
                "--------------------------------------------------------------------"
            )
        else:
            print(f"A FileNotFoundError occurred: {e}")  # Other file not found
    except Exception as e:
        print(f"在步骤1执行期间发生严重错误: {e}")
        import traceback

        traceback.print_exc()
