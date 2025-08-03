import os
import glob
import json  # For storing list of dicts in CSV cell
import numpy as np
import pandas as pd
import imagehash  # For imagehash.hex_to_hash
from tqdm import tqdm
from sklearn.neighbors import BallTree
from common import (
    ABS_EDITED_VIDEO_PATH,
    ABS_OUTPUT_DIR,
    stitch_images_vertically,
    run_command,
    get_video_metadata,
)
from config import (
    COARSE_MATCH_CSV_FILENAME,
    COARSE_MATCH_OUTPUT_SUBDIR_NAME,
    IF_STITCHED_IMAGES,
    IF_RECONSTRUCTED_VIDEO,
)


# --- Step 3 Specific Configuration ---
EDITED_VIDEO_FILENAME = os.path.basename(ABS_EDITED_VIDEO_PATH)
TOP_N_CANDIDATES = 3  # Number of top candidate matches to find for each edited frame
HASH_BITS = 256  # Based on hash_size=16 for pHash (16*16=256 bits)
# --- END Configuration ---


def hex_to_binary_array(hex_hash_str):
    """
    将十六进制的pHash字符串转换为一维numpy布尔数组。
    假设哈希字符串的长度与 HASH_BITS 设置匹配。
    """
    try:
        img_hash_obj = imagehash.hex_to_hash(hex_hash_str)
        # .hash is usually a 2D boolean array (e.g., 16x16 for 256 bits)
        binary_array = img_hash_obj.hash.flatten()
        if len(binary_array) != HASH_BITS:
            # 理想情况下，如果CSV正确且HASH_BITS设置正确，就不应该发生这种情况。
            print(
                f"警告: pHash字符串 {hex_hash_str} 未能生成 {HASH_BITS} 位。 "
                f"实际得到 {len(binary_array)} 位。请检查 HASH_BITS 设置或pHash生成过程。"
            )
            # 填充或截断？目前让其通过，如果度量允许，BallTree 可能能处理不同长度，
            # 否则如果所有输入维度不一致，稍后会报错。
            # 最好在第二步就确保 pHash 长度保持一致。
    except ValueError as e:
        print(
            f"错误: 转换十六进制字符串 '{hex_hash_str}' 到二进制数组失败: {e}。返回 None。"
        )
        return None
    return binary_array.astype(
        bool
    )  # Ensure boolean type for BallTree 'hamming' metric


def load_phash_data_from_csv(csv_path):
    """从CSV文件中加载pHash数据到Pandas DataFrame。"""
    try:
        df = pd.read_csv(csv_path)
        required_cols = [
            "video_name",
            "image_filename",
            "frame_number",
            "timestamp_ms",
            "phash",
            "image_path",
        ]
        if not all(col in df.columns for col in required_cols):
            print(
                f"警告: CSV文件 {os.path.basename(csv_path)} 缺少一个或多个必需列。"
                f"应包含: {', '.join(required_cols)}."
            )
            return None
        if df.empty:
            return df

        df["phash"] = df["phash"].astype(str)
        df["frame_number"] = df["frame_number"].astype(int)
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        df.dropna(
            subset=["timestamp_ms", "phash"], inplace=True
        )  # Drop rows where essential data became NaN
        df = df[
            df["phash"].apply(lambda x: len(x) == HASH_BITS // 4)
        ]  # 确保十六进制字符串的长度与 HASH_BITS 设置匹配

        return df
    except FileNotFoundError:
        print(f"错误: 在 {csv_path} 未找到pHash CSV文件")
        return None
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        print(f"错误: 加载CSV文件 {os.path.basename(csv_path)} 失败: {e}")
        return None


def stitch_top_match():
    """
    读取粗匹配结果CSV文件，为每个待比较帧与其最佳匹配帧创建并保存拼接图。
    该函数在main_step3成功生成CSV后被调用。
    """
    print("\n--- Running Optional Step: Stitching Top Match Images ---")

    # 1. 定义路径
    coarse_match_results_csv_path = os.path.join(
        ABS_OUTPUT_DIR, COARSE_MATCH_CSV_FILENAME
    )
    coarse_match_visual_output_dir = os.path.join(
        ABS_OUTPUT_DIR, COARSE_MATCH_OUTPUT_SUBDIR_NAME
    )

    # 2. 检查并加载CSV结果文件
    if not os.path.exists(coarse_match_results_csv_path):
        print(
            f"Error: Coarse match results CSV not found at "
            f"'{coarse_match_results_csv_path}'."
        )
        print("Cannot proceed with stitching images. Please run Step 3 first.")
        return

    try:
        df_results = pd.read_csv(coarse_match_results_csv_path)
        if df_results.empty:
            print("Warning: The coarse match results CSV is empty. Nothing to stitch.")
            return
    except Exception as e:
        print(f"Error loading coarse match results CSV: {e}")
        return

    # 3. 创建输出目录
    os.makedirs(coarse_match_visual_output_dir, exist_ok=True)
    print(f"Stitched images will be saved in: {coarse_match_visual_output_dir}")

    # 4. 遍历结果，生成拼接图
    for _, row in tqdm(
        df_results.iterrows(), total=len(df_results), desc="Stitching Images"
    ):
        try:
            # 解析JSON字符串获取匹配列表
            top_n_matches = json.loads(row["top_n_matches"])

            if not top_n_matches:
                continue  # 如果没有匹配项，则跳过

            # 获取最佳匹配（列表中的第一个）
            best_match = top_n_matches[0]

            # 获取待比较帧和源帧的相对路径
            edited_image_path_relative = row["edited_image_path"]
            original_image_path_relative = best_match["original_image_path"]

            # 定义拼接图的输出路径
            stitched_image_filename = row["edited_frame_filename"]
            stitched_image_output_path = os.path.join(
                coarse_match_visual_output_dir, stitched_image_filename
            )

            # 如果拼接图已存在，则跳过，以避免重复工作
            if os.path.exists(stitched_image_output_path):
                continue

            # 构建源图片的完整绝对路径
            # 注意：所有相对路径都是基于 ABS_OUTPUT_DIR
            full_edited_img_path = os.path.join(
                ABS_OUTPUT_DIR, edited_image_path_relative
            )
            full_original_img_path = os.path.join(
                ABS_OUTPUT_DIR, original_image_path_relative
            )

            # 检查图片文件是否存在
            if not os.path.exists(full_edited_img_path):
                print(
                    f"\nWarning: Edited frame image not found at "
                    f"{full_edited_img_path}. Skipping stitch."
                )
                continue
            if not os.path.exists(full_original_img_path):
                print(
                    f"\nWarning: Original candidate frame image not found at "
                    f"{full_original_img_path}. Skipping stitch."
                )
                continue

            # 调用拼接函数
            stitch_images_vertically(
                full_edited_img_path,
                full_original_img_path,
                stitched_image_output_path,
            )

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # 捕获解析JSON或访问字典/列表时可能发生的错误
            print(
                f"\nWarning: Could not process row for "
                f"{row['edited_frame_filename']} due to data error: {e}. "
                f"Skipping."
            )
            continue

    print("\nImage stitching process completed.")
    print(
        "Stitched comparison images (if any) saved in: "
        f"{coarse_match_visual_output_dir}"
    )
    print("\n--- Stitching Step finished. ---")


# 将此函数替换掉您脚本中原有的 generate_reconstructed_video 函数


def generate_reconstructed_video():
    """
    使用粗匹配结果生成一个最终的上下拼接对比视频。
    1. 首先根据最佳匹配帧生成一个临时的“重建”视频。
    2. 然后将原始视频（上）与重建视频（下）垂直拼接。
    3. 新视频的帧率与原始视频一致，并直接复制其音轨。
    4. 最后清理所有临时文件。
    """
    print("\n--- Running Optional Step: Generating Stacked Comparison Video ---")

    # 1. 定义所有需要的路径
    coarse_match_results_csv_path = os.path.join(
        ABS_OUTPUT_DIR, COARSE_MATCH_CSV_FILENAME
    )

    # 临时重建视频的路径
    reconstructed_video_filename = f"reconstructed_intermediate_{EDITED_VIDEO_FILENAME}"
    reconstructed_video_path = os.path.join(
        ABS_OUTPUT_DIR, reconstructed_video_filename
    )

    # 最终输出的上下拼接视频的路径
    stacked_video_filename = f"comparison_stacked_{EDITED_VIDEO_FILENAME}"
    stacked_video_path = os.path.join(ABS_OUTPUT_DIR, stacked_video_filename)

    # FFmpeg需要一个临时的文本文件来列出所有输入图片
    ffmpeg_input_list_path = os.path.join(ABS_OUTPUT_DIR, "ffmpeg_input_list.txt")

    # 2. 检查并加载CSV结果文件
    if not os.path.exists(coarse_match_results_csv_path):
        print(
            "Error: Coarse match results CSV not found at "
            f"'{coarse_match_results_csv_path}'."
        )
        print("Cannot proceed with video generation. Please run Step 3 first.")
        return

    try:
        df_results = pd.read_csv(coarse_match_results_csv_path)
        if df_results.empty:
            print(
                "Warning: The coarse match results CSV is empty. Nothing to generate."
            )
            return
    except Exception as e:
        print(f"Error loading coarse match results CSV: {e}")
        return

    # 3. 获取待比较视频的元数据（主要是帧率）
    print(f"Getting metadata from the original edited video: {EDITED_VIDEO_FILENAME}")
    metadata = get_video_metadata(ABS_EDITED_VIDEO_PATH)
    if not metadata or "avg_frame_rate" not in metadata:
        print(
            "Error: Could not retrieve frame rate from the edited video. "
            "Cannot set output video frame rate."
        )
        print("Aborting video generation.")
        return
    frame_rate = metadata["avg_frame_rate"]
    print(f"Source video frame rate detected: {frame_rate}")

    # 使用 try...finally 确保临时文件最终被删除
    try:
        # --- STAGE 1: Generate the intermediate reconstructed video ---
        print("\n[Stage 1/2] Generating intermediate reconstructed video...")

        with open(ffmpeg_input_list_path, "w", encoding="utf-8") as f_list:
            for _, row in df_results.iterrows():
                try:
                    top_n_matches = json.loads(row["top_n_matches"])
                    if not top_n_matches:
                        print(
                            "Warning: No match found for edited frame "
                            f"{row['edited_frame_filename']}. It will be skipped."
                        )
                        continue

                    best_match = top_n_matches[0]
                    original_image_path_relative = best_match["original_image_path"]
                    full_original_img_path = os.path.join(
                        ABS_OUTPUT_DIR, original_image_path_relative
                    )

                    if not os.path.exists(full_original_img_path):
                        print(
                            "Warning: Best match image not found at "
                            f"{full_original_img_path}. Skipping frame."
                        )
                        continue

                    f_list.write(f"file '{os.path.abspath(full_original_img_path)}'\n")

                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(
                        f"\nWarning: Could not process row for "
                        f"{row['edited_frame_filename']} due to data error: {e}."
                        " Skipping frame."
                    )
                    continue

        # 构建并执行 FFmpeg 命令以创建重建视频
        command_reconstruct = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-r",
            str(frame_rate),
            "-i",
            ffmpeg_input_list_path,
            "-i",
            ABS_EDITED_VIDEO_PATH,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",  # 仍然使用 libx264 编码器
            "-profile:v",
            "main",  # 【关键】使用兼容性非常好的 Main Profile
            "-level",
            "4.1",  # 【关键】设置一个广泛支持的 Level (支持到 720p@30fps)
            "-preset",
            "ultrafast",  # 使用最快的预设
            "-crf",
            "30",  # 降低质量要求 (值越高，质量越差，文件越小)
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-shortest",
            reconstructed_video_path,
        ]

        _, stderr_reconstruct, process_reconstruct = run_command(command_reconstruct)

        if process_reconstruct.returncode != 0:
            print("\n--- FFmpeg command failed during intermediate video creation! ---")
            print(
                "Error generating reconstructed video. Return code: "
                f"{process_reconstruct.returncode}"
            )
            print("FFmpeg stderr:", stderr_reconstruct)
            return  # 失败则终止

        print(f"Successfully created intermediate video: {reconstructed_video_path}")

        # --- STAGE 2: Stack the original and reconstructed videos ---
        print("\n[Stage 2/2] Generating final stacked comparison video...")

        # 构建并执行 FFmpeg 命令以上下拼接视频
        # -i original.mp4 (input 0)
        # -i reconstructed.mp4 (input 1)
        # filter_complex "[0:v][1:v]vstack=inputs=2[v]" -> 将两个视频流垂直拼接
        # -map "[v]" -> 使用拼接后的视频流
        # -map "0:a?" -> 使用第一个输入（原始视频）的音频流
        command_stack = [
            "ffmpeg",
            "-y",
            "-i",
            ABS_EDITED_VIDEO_PATH,
            "-i",
            reconstructed_video_path,
            "-filter_complex",
            "[0:v][1:v]vstack=inputs=2[v]",
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-c:a",
            "copy",
            "-c:v",
            "libx264",  # 仍然使用 libx264 编码器
            "-profile:v",
            "main",  # 【关键】使用兼容性非常好的 Main Profile
            "-level",
            "4.1",  # 【关键】设置一个广泛支持的 Level (支持到 720p@30fps)
            "-preset",
            "ultrafast",  # 使用最快的预设
            "-crf",
            "30",  # 降低质量要求 (值越高，质量越差，文件越小)
            "-pix_fmt",
            "yuv420p",
            stacked_video_path,
        ]

        _, stderr_stack, process_stack = run_command(command_stack)

        if process_stack.returncode != 0:
            print("\n--- FFmpeg command failed during video stacking! ---")
            print(
                "Error generating stacked video. Return code: "
                f"{process_stack.returncode}"
            )
            print("FFmpeg stderr:", stderr_stack)
        else:
            print("\n------------------------------------------------------------")
            print("SUCCESS! Final comparison video generated at:")
            print(f"{stacked_video_path}")
            print("------------------------------------------------------------")

    finally:
        # --- STAGE 3: Cleanup ---
        print("\n[Stage 3/3] Cleaning up temporary files...")
        if os.path.exists(ffmpeg_input_list_path):
            os.remove(ffmpeg_input_list_path)
            print(
                "  - Removed temporary list file: "
                f"{os.path.basename(ffmpeg_input_list_path)}"
            )
        if os.path.exists(reconstructed_video_path):
            os.remove(reconstructed_video_path)
            print(
                f"  - Removed intermediate video: "
                f"{os.path.basename(reconstructed_video_path)}"
            )

    print("\n--- Stacked Comparison Video Generation finished. ---")


# --- Main Step 3 Logic ---
def main_step3():
    """执行步骤3：使用BallTree进行粗略帧匹配。
    该步骤假设已完成步骤1和2，并且已生成所需的pHash数据。
    """

    # 0. Setup paths
    abs_base_output_dir = ABS_OUTPUT_DIR

    if not os.path.exists(abs_base_output_dir):
        print(
            f"Error: Base output directory '{abs_base_output_dir}' not found. "
            f"Please run Step 1 & 2 first."
        )
        return

    edited_video_name_no_ext, _ = os.path.splitext(EDITED_VIDEO_FILENAME)
    edited_video_phash_csv_path = os.path.join(
        abs_base_output_dir,
        edited_video_name_no_ext,
        f"{edited_video_name_no_ext}_phash.csv",
    )

    coarse_match_results_csv_path = os.path.join(
        abs_base_output_dir, COARSE_MATCH_CSV_FILENAME
    )

    print("--- Running Step 3: Coarse Frame Matching (with BallTree) ---")
    print(f"Edited Video Filename (for pHash lookup): {EDITED_VIDEO_FILENAME}")
    print(f"Base Output Directory: {abs_base_output_dir}")
    print(f"Top N Candidates per Edited Frame: {TOP_N_CANDIDATES}")
    print(f"Hash Bits (pHash length): {HASH_BITS}")
    print(f"Coarse Match CSV Filename: {COARSE_MATCH_CSV_FILENAME}")
    print("-" * 30)

    # 1. 加载剪辑后视频的pHash数据
    print(f"加载剪辑后视频的pHash数据: {edited_video_name_no_ext}...")
    if not os.path.exists(edited_video_phash_csv_path):
        print(
            f"致命错误: 在 '{edited_video_phash_csv_path}' 未找到剪辑后视频的pHash CSV。"
            f"请先为该视频运行第二步。"
        )
        return
    df_edited_video = load_phash_data_from_csv(edited_video_phash_csv_path)
    if df_edited_video is None or df_edited_video.empty:
        print(
            f"致命错误: 剪辑后视频 ('{edited_video_name_no_ext}') 的pHash数据为空或无法加载。"
            f"请检查CSV: '{edited_video_phash_csv_path}'"
        )
        return
    print(
        f"已为剪辑后视频 '{edited_video_name_no_ext}' 加载 {len(df_edited_video)} 帧。"
    )

    # 2. 加载所有原始素材视频的pHash数据
    print("\n加载原始素材视频的pHash数据...")
    original_source_phash_dfs = []
    all_phash_csv_files_glob = os.path.join(abs_base_output_dir, "*", "*_phash.csv")

    for csv_file_path in glob.glob(all_phash_csv_files_glob):
        if os.path.normpath(csv_file_path) == os.path.normpath(
            edited_video_phash_csv_path
        ):
            continue

        source_video_name_from_dir = os.path.basename(os.path.dirname(csv_file_path))
        print(
            f"  尝试加载pHash数据: {source_video_name_from_dir} "
            f"from {os.path.basename(csv_file_path)}"
        )
        df_source = load_phash_data_from_csv(csv_file_path)

        if df_source is not None and not df_source.empty:
            if (
                not df_source["video_name"].empty
                and df_source["video_name"].iloc[0] != source_video_name_from_dir
            ):
                print(
                    f"    警告: 视频名称与目录不匹配 ({source_video_name_from_dir})。"
                    f"    CSV 中报告的是 '{df_source['video_name'].iloc[0]}'。"
                    f"    将使用目录名称。"
                )
                df_source["video_name"] = source_video_name_from_dir
            original_source_phash_dfs.append(df_source)
            print(
                f"    成功为 '{source_video_name_from_dir}' 加载 {len(df_source)} 帧。"
            )
        elif df_source is not None and df_source.empty:
            print(
                f"    注意: '{source_video_name_from_dir}' 的pHash数据为空或被过滤。"
                f"    CSV: {os.path.basename(csv_file_path)}"
            )
        else:
            print(
                f"    警告: 加载 '{source_video_name_from_dir}' 的pHash数据失败。"
                f"    CSV: {os.path.basename(csv_file_path)}"
            )

    if not original_source_phash_dfs:
        print(
            "FATAL: No pHash data found for any original source videos. "
            f"Please ensure Step 2 has run for all source videos and their "
            f"pHash CSVs are in '{abs_base_output_dir}/<source_video_name>/'."
        )
        return

    df_all_originals = pd.concat(original_source_phash_dfs, ignore_index=True)
    if df_all_originals.empty:
        print("致命错误: 合并后的原始素材pHash数据为空，无法构建BallTree。")
        return

    print(f"\n准备 {len(df_all_originals)} 个原始素材帧用于BallTree...")

    # 将原始帧的十六进制pHash转换为二进制数组用于BallTree
    original_phashes_binary_list = []
    valid_original_indices = []  # Keep track of indices that yield valid binary arrays
    for idx, hex_hash in enumerate(df_all_originals["phash"]):
        binary_arr = hex_to_binary_array(hex_hash)
        if binary_arr is not None and len(binary_arr) == HASH_BITS:
            original_phashes_binary_list.append(binary_arr)
            valid_original_indices.append(idx)
        else:
            print(
                f"  Skipping original frame (index {idx}, hash {hex_hash}) "
                f"  due to pHash conversion error or length mismatch."
            )

    if not original_phashes_binary_list:
        print("致命错误: 无法从原始素材帧生成有效的二进制pHash，无法构建BallTree。")
        return

    original_phashes_matrix = np.array(original_phashes_binary_list)
    # 仅保留具有有效二进制哈希的帧，过滤df_all_originals
    df_all_originals_valid = df_all_originals.iloc[valid_original_indices].reset_index(
        drop=True
    )

    print(
        f"Building BallTree with {len(original_phashes_matrix)} original source frames "
        f"(metric: hamming)..."
    )
    tree = BallTree(
        original_phashes_matrix.astype(np.int8), metric="hamming"
    )  # BallTree expects numerical 0/1 for hamming

    # 3. Perform coarse matching using BallTree
    coarse_match_results_list = []

    print(
        f"\nPerforming coarse matching for {len(df_edited_video)} edited frames "
        f"(Top {TOP_N_CANDIDATES} candidates using BallTree)..."
    )
    for _, edited_frame_row in tqdm(
        df_edited_video.iterrows(),
        total=len(df_edited_video),
        desc="Matching Edited Frames",
    ):
        edited_phash_hex = str(edited_frame_row["phash"])
        edited_image_path_relative = edited_frame_row["image_path"]

        edited_phash_binary = hex_to_binary_array(edited_phash_hex)
        if edited_phash_binary is None or len(edited_phash_binary) != HASH_BITS:
            print(
                f"  Skipping edited frame {edited_frame_row['image_filename']} "
                f"  due to pHash conversion error or length mismatch."
            )
            # 添加一个没有匹配项的条目，或跳过？现在让我们添加空匹配项
            coarse_match_results_list.append(
                {
                    "edited_video_name": edited_frame_row["video_name"],
                    "edited_frame_filename": edited_frame_row["image_filename"],
                    "edited_frame_number": int(edited_frame_row["frame_number"]),
                    "edited_timestamp_ms": float(edited_frame_row["timestamp_ms"]),
                    "edited_phash": edited_phash_hex,
                    "edited_image_path": edited_image_path_relative,
                    "top_n_matches": json.dumps([]),  # Empty list of matches
                }
            )
            continue

        # Query the tree
        # .reshape(1,-1) to make it a 2D array for query
        # .astype(np.int8) to match tree input type
        distances_prop, indices = tree.query(
            edited_phash_binary.astype(np.int8).reshape(1, -1),
            k=min(TOP_N_CANDIDATES, len(original_phashes_matrix)),
        )

        # distances_prop is fractional Hamming distance (proportion of differing bits)
        # Convert to integer Hamming distance (count of differing bits)
        distances_int = (distances_prop[0] * HASH_BITS).round().astype(int)
        candidate_indices_in_valid_df = indices[0]

        top_n_selected_matches = []
        for i, original_idx in enumerate(candidate_indices_in_valid_df):
            original_frame_record = df_all_originals_valid.iloc[original_idx]
            top_n_selected_matches.append(
                {
                    "original_video_name": original_frame_record["video_name"],
                    "original_frame_filename": original_frame_record["image_filename"],
                    "original_frame_number": int(original_frame_record["frame_number"]),
                    "original_timestamp_ms": float(
                        original_frame_record["timestamp_ms"]
                    ),
                    "original_phash": original_frame_record["phash"],
                    "original_image_path": original_frame_record["image_path"],
                    "phash_distance": int(
                        distances_int[i]
                    ),  # ensure it's python int for JSON
                }
            )

        coarse_match_results_list.append(
            {
                "edited_video_name": edited_frame_row["video_name"],
                "edited_frame_filename": edited_frame_row["image_filename"],
                "edited_frame_number": int(edited_frame_row["frame_number"]),
                "edited_timestamp_ms": float(edited_frame_row["timestamp_ms"]),
                "edited_phash": edited_phash_hex,
                "edited_image_path": edited_image_path_relative,
                "top_n_matches": json.dumps(top_n_selected_matches),
            }
        )

    # 4. Save the coarse match results CSV
    if coarse_match_results_list:
        df_coarse_results = pd.DataFrame(coarse_match_results_list)
        df_coarse_results.to_csv(
            coarse_match_results_csv_path, index=False, lineterminator="\n"
        )
        print(f"\nCoarse match results (CSV) saved to: {coarse_match_results_csv_path}")
    else:
        print("\nNo coarse match results were generated.")
        return

    print("\n--- Step 3 (Coarse Matching with BallTree) completed. ---")

    # 5. Optionally generate stitched video for the best match
    if IF_RECONSTRUCTED_VIDEO:
        generate_reconstructed_video()

    # 6. Optionally stitch images for the best match
    if IF_STITCHED_IMAGES:
        stitch_top_match()


if __name__ == "__main__":

    try:
        main_step3()
    except Exception as e:
        print(f"\nAN UNEXPECTED CRITICAL ERROR occurred during Step 3 execution: {e}")
        import traceback

        traceback.print_exc()
