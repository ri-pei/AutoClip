import os
import cv2  # OpenCV for image processing and transform estimation
import numpy as np  # For numerical operations with OpenCV
from common import (
    discover_source_videos,
    get_video_metadata,
)


# --- 内部常量 ---
MIN_MATCH_COUNT_GEO = 10  # SIFT/ORB匹配的最小特征点数
# --- END 内部常量 ---


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

    # Compact matching uses mask coordinates in the target resolution.  Without
    # reference geometry, differently sized sources still need this normalization
    # before drawbox; otherwise the same watermark rectangle covers another area.
    if not is_edited_video_flag and not geom_transform_to_apply:
        source_meta = get_video_metadata(video_path)
        if not source_meta:
            raise RuntimeError(f"Cannot read source dimensions: {video_path}")
        if (source_meta["width"], source_meta["height"]) != (final_output_resolution["width"], final_output_resolution["height"]):
            vf_options.append(f"scale={final_output_resolution['width']}:{final_output_resolution['height']}")

    # --- 2. Apply Color Transformation - ONLY FOR ORIGINAL VIDEOS ---
    if not is_edited_video_flag:
        if color_lut_to_apply and os.path.exists(color_lut_to_apply):
            # Ensure path is suitable for ffmpeg (e.g., escape special chars if any -
            # simplified here)
            lut_path_escaped = color_lut_to_apply.replace(
                "\\", "/"
            )  # Basic path normalization
            vf_options.append(f"lut3d=file='{lut_path_escaped}'")

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

    from compact_frames import extract_compact
    return extract_compact(video_path, video_specific_output_dir, vf_options)


def main_step1(settings):
    """
    主函数，执行步骤1：提取帧并应用必要的几何变换和颜色校正。
    处理所有视频文件，使用用户指定的编辑后视频作为参考。
    """

    abs_working_dir = settings["working_dir"]
    abs_edited_video_path = settings["edited_video_path"]
    abs_output_dir = settings["output_dir"]
    abs_source_video_folder = settings["source_dir"]
    abs_ref_original_frame_path = settings["reference_original_path"]
    abs_ref_edited_frame_path = settings["reference_edited_path"]
    abs_user_color_lut_path = settings["color_lut_path"]
    mask_rect = settings["mask_rect"]
    edited_video_filename = os.path.basename(abs_edited_video_path)

    print("开始步骤1: 提取帧和准备(包含变换估计)...")
    # 打印预处理后的路径以供用户确认
    print(f"工作目录: {abs_working_dir}")
    print(f"源视频文件夹: {abs_source_video_folder or '未设置'}")
    print(f"输出目录: {abs_output_dir}")
    print("-" * 30)
    print(f"编辑后的视频文件: {abs_edited_video_path}")
    if mask_rect:
        print(f"遮罩矩形配置: {mask_rect}")
    else:
        print("遮罩矩形: 未设置.")

    print(f"参考-原始帧: {abs_ref_original_frame_path or '未设置'}")
    print(f"参考-编辑后帧: {abs_ref_edited_frame_path or '未设置'}")
    if abs_user_color_lut_path:
        print(f"用户颜色LUT文件: {abs_user_color_lut_path}")
    else:
        print("用户颜色LUT文件: 未设置.")
    print("-" * 30)


    # 输出目录已在预处理部分解析，此处确保它存在
    os.makedirs(abs_output_dir, exist_ok=True)

    # 检查必须的编辑后视频文件是否存在
    if not abs_edited_video_path or not os.path.exists(abs_edited_video_path):
        raise FileNotFoundError(
            f"编辑后的视频 '{edited_video_filename}' "
            f"在解析的路径 '{abs_edited_video_path}' 未找到。"
        )

    print(
        f"正在处理编辑后的视频: {os.path.basename(abs_edited_video_path)} "
        "以确定最终输出分辨率..."
    )
    edited_metadata = get_video_metadata(abs_edited_video_path)
    if (
        not edited_metadata
        or "width" not in edited_metadata
        or "height" not in edited_metadata
    ):
        raise RuntimeError(
            f"无法获取编辑后视频 '{os.path.basename(abs_edited_video_path)}' 的元数据。"
        )

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
        abs_ref_original_frame_path
        and os.path.exists(abs_ref_original_frame_path)
        and abs_ref_edited_frame_path
        and os.path.exists(abs_ref_edited_frame_path)
    ):
        print("警告: 用于几何变换的参考帧未找到:")
        if not abs_ref_original_frame_path or not os.path.exists(
            abs_ref_original_frame_path
        ):
            print(
                f"  缺失: {abs_ref_original_frame_path}"
            )
        if not abs_ref_edited_frame_path or not os.path.exists(
            abs_ref_edited_frame_path
        ):
            print(
                f"  缺失: {abs_ref_edited_frame_path}"
            )
        print("  将跳过对源素材的几何变换（裁切/缩放）。")
        geometric_transform = None
    else:
        geometric_transform = estimate_geometric_transform_from_refs(
            abs_ref_original_frame_path, abs_ref_edited_frame_path
        )
        if not geometric_transform:
            print("  几何变换估算失败，将不会对源素材进行特定的裁切/缩放。")

    # --- 检查用户提供的颜色LUT文件 ---
    # 路径已被预处理为绝对路径，只需检查文件是否存在
    final_color_lut_path = None
    if abs_user_color_lut_path:
        if os.path.exists(abs_user_color_lut_path):
            print(f"  将使用颜色校正LUT文件: {abs_user_color_lut_path}")
            final_color_lut_path = abs_user_color_lut_path
        else:
            print(f"警告: 用户指定的颜色LUT文件未找到: {abs_user_color_lut_path}")
            print("  将跳过颜色LUT的应用。")

    print(f"\n正在源视频文件夹中搜索视频: {abs_source_video_folder}")
    source_video_files = discover_source_videos(
        abs_source_video_folder, abs_edited_video_path
    )

    # 最终处理列表 = 编辑后的视频 + 所有源视频 (去重)
    all_videos_to_process = {os.path.normpath(abs_edited_video_path)}
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
            abs_edited_video_path
        )

        current_geom_transform = None
        current_color_lut = None
        if not is_edited:  # 仅对源素材应用特殊变换
            current_geom_transform = geometric_transform
            current_color_lut = final_color_lut_path

        success = process_video_for_frames(
            video_path=video_full_path,
            video_name_no_ext=video_name_no_ext,
            main_output_folder=abs_output_dir,
            is_edited_video_flag=is_edited,
            final_output_resolution=final_output_resolution,
            mask_rect_config=mask_rect,
            geom_transform_to_apply=current_geom_transform,
            color_lut_to_apply=current_color_lut,
        )
        if not success:
            raise RuntimeError(f"视频逐帧处理失败: {video_full_path}")

    print("\n\n步骤 1 (帧提取与变换) 已完成。")
    print(f"所有输出的帧位于以下目录的子文件夹中: {abs_output_dir}")


if __name__ == "__main__":
    from settings import load_settings

    main_step1(load_settings("config.toml"))
