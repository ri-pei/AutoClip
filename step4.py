import pandas as pd
import json
import numpy as np
import os
from common import ABS_OUTPUT_DIR, ABS_EDITED_VIDEO_PATH
from config import COARSE_MATCH_CSV_FILENAME, FINAL_SEGMENTS_CSV_FILENAME


# --- Step 4 Specific Configuration ---
OFFSET_JUMP_THRESHOLD_MS = 1000  # 用于检测潜在的中断
SUSTAINED_LOOKAHEAD_FRAMES = 3  # 检查多少帧以确认持续性中断
# 用于检查前瞻帧是否与一个*新提出*的片段对齐。
# 可以与 OFFSET_JUMP_THRESHOLD_MS 相同，或稍微宽松一些
OFFSET_JUMP_THRESHOLD_MS_SUSTAINED = 1200
# --- END Configuration ---


# --- 辅助函数 ---
def load_coarse_matches_with_timestamps(filepath):
    """
    从CSV文件加载粗略匹配结果，并确保时间戳和帧号信息完整。
    如果CSV中缺少 'edited_timestamp_ms' 或 'edited_frame_number' 列，
    此函数会尝试从 'edited_frame_filename' 文件名中解析这些信息。
    最后，它会根据编辑后视频的帧号对数据进行排序。
    :param filepath: 粗略匹配结果CSV文件的路径。
    :return: 一个处理过的Pandas DataFrame，包含了时间戳和帧号，并按帧号排序。
    """

    df = pd.read_csv(filepath)
    df["top_n_matches"] = df["top_n_matches"].apply(json.loads)

    # 确保 edited_timestamp_ms 和 edited_frame_number 列存在
    def parse_time_from_filename(filename_str):
        # 用于apply，解析文件名中的时间戳信息。
        if not isinstance(filename_str, str):  # 防止处理 NaN或其他类型
            return None
        try:
            time_str = filename_str.split("_time_")[-1].replace(".png", "")
            parts = time_str.split("-")
            h, m, s, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            return (h * 3600 + m * 60 + s) * 1000 + ms
        except Exception:
            return None

    def get_frame_num_from_filename(filename_str):
        # 用于apply，解析文件名中的帧号信息。
        if not isinstance(filename_str, str):
            return None
        try:
            return int(filename_str.split("_frame_")[-1].split("_time_")[0])
        except Exception:
            return None

    if "edited_timestamp_ms" not in df.columns:
        df["edited_timestamp_ms"] = df["edited_frame_filename"].apply(
            parse_time_from_filename
        )
    if "edited_frame_number" not in df.columns:
        df["edited_frame_number"] = df["edited_frame_filename"].apply(
            get_frame_num_from_filename
        )

    df.dropna(subset=["edited_timestamp_ms", "edited_frame_number"], inplace=True)
    df["edited_frame_number"] = df["edited_frame_number"].astype(int)
    df = df.sort_values(by="edited_frame_number").reset_index(drop=True)
    return df


def get_initial_best_match_from_top_n(top_n_list):
    if not top_n_list:
        return None
    return min(top_n_list, key=lambda x: x["phash_distance"])


def load_all_original_frames_data(phash_csvs_base_dir, edited_video_name_no_ext):
    # 加载所有原始视频的pHash数据。
    # 此函数会遍历指定的目录，查找所有以 "_phash.csv" 结尾的文件，
    # 同时排除与编辑后视频同名的文件。
    # 它会从文件路径推断原始视频的名称，加载数据，重命名列以保持标准，
    # 并将所有原始视频的帧数据合并成一个大的DataFrame。
    # :param phash_csvs_base_dir: 存放pHash CSV文件的基础目录。
    # :param edited_video_name_no_ext: 编辑后视频的无扩展名文件名，用于排除其自身的pHash文件。
    # :return: 一个包含所有原始视频帧数据的合并后的Pandas DataFrame。
    # :raises ValueError: 如果没有加载到任何原始视频pHash数据，则抛出此异常。
    all_original_dfs = []
    # 用于查找原始视频 pHash CSV 的路径模式
    # 假设 pHash CSV 位于以视频命名的子目录中
    # 例如, output/<original_video_name>/<original_video_name>_phash.csv

    # 首先找到所有潜在的 pHash CSV 文件
    phash_files = []
    for root, dirs, files in os.walk(phash_csvs_base_dir):
        for file in files:
            if file.endswith("_phash.csv") and edited_video_name_no_ext not in file:
                phash_files.append(os.path.join(root, file))

    print(f"Found potential original pHash files: {phash_files}")

    for f_path in phash_files:
        try:
            # 从 CSV 文件名或其父目录推断 original_video_name
            # 这部分对您的目录结构和命名方式很敏感
            original_video_name = os.path.basename(
                os.path.dirname(f_path)
            )  # 假设 CSV 文件位于以视频命名的目录中
            if (
                original_video_name == edited_video_name_no_ext
                or not original_video_name
            ):  # 再次检查
                original_video_name = os.path.basename(f_path).replace("_phash.csv", "")
                if (
                    original_video_name == edited_video_name_no_ext
                ):  # 仍然是编辑过的那个？跳过。
                    continue

            df_orig = pd.read_csv(f_path)
            # 确保使用标准列名；如果您的 pHash CSV 不同，请进行调整
            # 我们需要: 'original_video_name', 'original_frame_number',
            # 'original_timestamp_ms', 'original_phash'
            df_orig.rename(
                columns={
                    "video_name": "original_video_name_temp",  # 避免立即覆盖
                    # (如果'original_video_name'已存在）
                    "frame_number": "original_frame_number",
                    "timestamp_ms": "original_timestamp_ms",
                    "phash": "original_phash",
                },
                inplace=True,
                errors="ignore",
            )  # 如果某些列可能不存在，则 errors='ignore'

            # 如果 'original_video_name' 不在 CSV 中，则分配它
            if "original_video_name_temp" not in df_orig.columns:
                df_orig["original_video_name"] = original_video_name
            else:
                df_orig["original_video_name"] = df_orig["original_video_name_temp"]

            # 选择必要的列
            # 检查重命名和分配后所有期望的列是否存在
            required_cols = [
                "original_video_name",
                "original_frame_number",
                "original_timestamp_ms",
                "original_phash",
            ]
            if not all(col in df_orig.columns for col in required_cols):
                print(
                    f"Warning: Skipping {f_path} due to 缺少指定列: "
                    f"{required_cols}. Available: {df_orig.columns.tolist()}"
                )
                continue

            all_original_dfs.append(df_orig[required_cols])
            print(f"Loaded {len(df_orig)} frames from {original_video_name} ({f_path})")

        except Exception as e:
            print(f"Error loading or processing original pHash CSV {f_path}: {e}")

    if not all_original_dfs:
        raise ValueError(
            "No original video pHash data loaded. Check paths and file contents."
        )

    df_combined = pd.concat(all_original_dfs, ignore_index=True)
    # 确保时间戳是数字类型，以便进行排序和计算
    df_combined["original_timestamp_ms"] = pd.to_numeric(
        df_combined["original_timestamp_ms"], errors="coerce"
    )
    df_combined.dropna(subset=["original_timestamp_ms"], inplace=True)
    return df_combined


# --- 主逻辑 ---
def main_step4():
    """
    主函数，执行第四步：从粗略匹配到最终精炼视频片段的生成。
    流程包括：
    1. 加载粗略匹配结果和所有原始视频的帧数据。
    2. 通过一个智能的、带前瞻性检查的算法，将帧匹配初步划分为连续的片段。
       该算法可以处理时间偏移的突变，并尝试从候选匹配中“拯救”连续性。
    3. 对每个初步片段，计算一个中位数时间偏移量。
    4. 使用此偏移量，在完整的原始视频帧数据中为每个编辑帧重新寻找最接近的匹配，
       从而精炼匹配结果，使其在时间上更连贯。
    5. 将精炼后的、连续的帧匹配合并成最终的视频片段。
    6. 将这些最终片段保存为CSV文件，可用于后续的FFmpeg处理。
    :param edited_video_name_no_ext_param: 编辑后视频的无扩展名文件名，用于加载数据时排除自身。
    """
    if not (ABS_EDITED_VIDEO_PATH and os.path.basename(ABS_EDITED_VIDEO_PATH)):
        print("错误: ABS_EDITED_VIDEO_PATH 未在 common.py 中正确定义。")
        print("请确保 config.py 中的 EDITED_VIDEO_FILENAME 配置正确。")
        return  # 提前退出

    edited_video_basename = os.path.basename(ABS_EDITED_VIDEO_PATH)
    edited_video_name_no_ext_param, _ = os.path.splitext(edited_video_basename)
    print(f"步骤4分析的目标视频: '{edited_video_name_no_ext_param}'")

    coarse_match_file = os.path.join(ABS_OUTPUT_DIR, COARSE_MATCH_CSV_FILENAME)
    df_processed_frames = load_coarse_matches_with_timestamps(coarse_match_file)

    if df_processed_frames.empty:
        print("No coarse match data loaded. Exiting.")
        return

    try:
        # 传入编辑后视频的名称以排除其 pHash CSV
        df_all_original_frames = load_all_original_frames_data(
            ABS_OUTPUT_DIR, edited_video_name_no_ext_param
        )
    except ValueError as e:
        print(e)
        return

    print(f"Total original frames loaded: {len(df_all_original_frames)}")
    if df_all_original_frames.empty:
        print("No original frames available for matching. Exiting.")
        return

    preliminary_segments = []
    current_segment_chosen_matches = []  # 存储 (edited_frame_row, chosen_match_dict)

    # 用第一帧的最佳匹配进行初始化
    if len(df_processed_frames) == 0:
        print("df_processed_frames is empty after loading. Cannot proceed.")
        return

    # --- 3. 改进的初步分段 ---
    last_confirmed_match_info = None

    for idx, edited_frame_row in df_processed_frames.iterrows():
        current_edited_ts = edited_frame_row["edited_timestamp_ms"]
        current_top_n = edited_frame_row["top_n_matches"]

        current_initial_best_match = get_initial_best_match_from_top_n(current_top_n)

        if current_initial_best_match is None:
            # 此帧没有匹配项，如果一个片段是打开的，则关闭它
            if current_segment_chosen_matches:
                preliminary_segments.append(list(current_segment_chosen_matches))
                current_segment_chosen_matches = []
            last_confirmed_match_info = None  # 重置
            print(
                f"Warning: Edited frame {edited_frame_row['edited_frame_number']} "
                f"has no coarse matches. Skipping."
            )
            continue

        chosen_match_for_current_frame = current_initial_best_match  # 默认值

        if last_confirmed_match_info is None:  # 视频的第一帧或在一个硬性中断之后
            is_break_decision = True  # 实际上，这是一个新片段的开始
        else:
            current_initial_offset = (
                current_initial_best_match["original_timestamp_ms"] - current_edited_ts
            )
            expected_offset_from_last = (
                last_confirmed_match_info["original_timestamp_ms"]
                - last_confirmed_match_info["edited_timestamp_ms"]
            )

            potential_break = (
                current_initial_best_match["original_video_name"]
                != last_confirmed_match_info["video_name"]
                or abs(current_initial_offset - expected_offset_from_last)
                > OFFSET_JUMP_THRESHOLD_MS
            )
            is_break_decision = potential_break  # 暂定

            if potential_break:
                # 尝试 1：从 top_N 列表中拯救
                rescued = False
                for candidate_match in current_top_n:
                    candidate_offset = (
                        candidate_match["original_timestamp_ms"] - current_edited_ts
                    )
                    if (
                        candidate_match["original_video_name"]
                        == last_confirmed_match_info["video_name"]
                        and abs(candidate_offset - expected_offset_from_last)
                        <= OFFSET_JUMP_THRESHOLD_MS
                    ):
                        chosen_match_for_current_frame = candidate_match
                        is_break_decision = False  # 已拯救，没有中断
                        rescued = True
                        break

                if not rescued:  # 尝试 2：确认持续性中断（如果仍然是中断的话）
                    is_sustained = True  # 初始假设是持续的
                    if idx + SUSTAINED_LOOKAHEAD_FRAMES < len(df_processed_frames):
                        for k_lookahead in range(1, SUSTAINED_LOOKAHEAD_FRAMES + 1):
                            lookahead_idx = idx + k_lookahead
                            lookahead_edited_frame_row = df_processed_frames.iloc[
                                lookahead_idx
                            ]
                            lookahead_edited_ts = lookahead_edited_frame_row[
                                "edited_timestamp_ms"
                            ]
                            lookahead_top_n = lookahead_edited_frame_row[
                                "top_n_matches"
                            ]
                            lookahead_best_match = get_initial_best_match_from_top_n(
                                lookahead_top_n
                            )

                            if lookahead_best_match is None:  # 一个前瞻帧没有匹配
                                is_sustained = False
                                break

                            lookahead_offset = (
                                lookahead_best_match["original_timestamp_ms"]
                                - lookahead_edited_ts
                            )
                            # 检查前瞻帧是否与 current_initial_best_match 提出的 *新* 片段对齐
                            if not (
                                lookahead_best_match["original_video_name"]
                                == current_initial_best_match["original_video_name"]
                                and abs(
                                    lookahead_offset
                                    - (
                                        current_initial_best_match[
                                            "original_timestamp_ms"
                                        ]
                                        - current_edited_ts
                                    )
                                )
                                <= OFFSET_JUMP_THRESHOLD_MS_SUSTAINED
                            ):
                                is_sustained = False
                                break
                    else:  # 没有足够的前瞻帧
                        is_sustained = False  # 无法确认，假设不是持续的

                    if is_sustained:
                        is_break_decision = True  # 确认的中断
                        chosen_match_for_current_frame = (
                            current_initial_best_match  # 此帧开启了新趋势
                        )
                    else:
                        is_break_decision = (
                            False  # 不是持续的，保留在旧片段中（使用其初始最佳匹配）
                        )
                        chosen_match_for_current_frame = current_initial_best_match

        if (
            is_break_decision and current_segment_chosen_matches
        ):  # is_break_decision 为 True 意味着新片段
            preliminary_segments.append(list(current_segment_chosen_matches))
            current_segment_chosen_matches = []

        current_segment_chosen_matches.append(
            {
                "edited_frame_data": edited_frame_row.to_dict(),
                "chosen_match_info": chosen_match_for_current_frame,
            }
        )

        last_confirmed_match_info = {
            "video_name": chosen_match_for_current_frame["original_video_name"],
            "original_timestamp_ms": chosen_match_for_current_frame[
                "original_timestamp_ms"
            ],
            "edited_timestamp_ms": current_edited_ts,  # 当前编辑帧的时间戳
        }

    if current_segment_chosen_matches:  # 添加最后一个片段
        preliminary_segments.append(list(current_segment_chosen_matches))

    print(
        f"Identified {len(preliminary_segments)} "
        f"preliminary segments after improved logic."
    )

    # --- 4. 片段偏移量计算 ---
    # --- 5. 帧匹配精炼 (使用完整的原始视频数据) ---
    all_refined_matches_list = []
    for seg_idx, segment_data_list in enumerate(preliminary_segments):
        if not segment_data_list:
            continue

        # 确定片段的原始视频并计算目标偏移量
        segment_original_video_name = segment_data_list[0]["chosen_match_info"][
            "original_video_name"
        ]

        offsets_in_segment = []
        for item in segment_data_list:
            offsets_in_segment.append(
                item["chosen_match_info"]["original_timestamp_ms"]
                - item["edited_frame_data"]["edited_timestamp_ms"]
            )

        if not offsets_in_segment:
            print(
                f"Warning: Segment {seg_idx} has no offsets. "
                "Skipping refinement for this segment."
            )
            # 可选地，将原始 chosen_match_info 添加到 all_refined_matches_list 作为备用
            for item in segment_data_list:
                all_refined_matches_list.append(
                    {
                        "edited_frame_number": item["edited_frame_data"][
                            "edited_frame_number"
                        ],
                        "edited_timestamp_ms": item["edited_frame_data"][
                            "edited_timestamp_ms"
                        ],
                        "refined_original_video_name": item["chosen_match_info"][
                            "original_video_name"
                        ],
                        "refined_original_frame_number": item["chosen_match_info"][
                            "original_frame_number"
                        ],
                        "refined_original_timestamp_ms": item["chosen_match_info"][
                            "original_timestamp_ms"
                        ],
                        "refined_phash_distance": item["chosen_match_info"].get(
                            "phash_distance", -1
                        ),  # pHash 在这里可能不相关
                        "comment": (
                            "Used chosen_match from preliminary segmentation "
                            "(no offset)"
                        ),
                    }
                )
            continue

        target_segment_offset_ms = np.median(offsets_in_segment)

        df_target_original_video_frames = df_all_original_frames[
            df_all_original_frames["original_video_name"] == segment_original_video_name
        ].copy()  # 使用 .copy() 以避免之后修改时出现 SettingWithCopyWarning

        if df_target_original_video_frames.empty:
            print(
                f"Warning: No frames found in df_all_original_frames for video "
                f"'{segment_original_video_name}' (Segment {seg_idx}). "
                f"Skipping segment."
            )
            for item in segment_data_list:  # 备用方案
                all_refined_matches_list.append(
                    {
                        "edited_frame_number": item["edited_frame_data"][
                            "edited_frame_number"
                        ],
                        "edited_timestamp_ms": item["edited_frame_data"][
                            "edited_timestamp_ms"
                        ],
                        "refined_original_video_name": item["chosen_match_info"][
                            "original_video_name"
                        ],
                        "refined_original_frame_number": item["chosen_match_info"][
                            "original_frame_number"
                        ],
                        "refined_original_timestamp_ms": item["chosen_match_info"][
                            "original_timestamp_ms"
                        ],
                        "refined_phash_distance": item["chosen_match_info"].get(
                            "phash_distance", -1
                        ),
                        "comment": (
                            "Used chosen_match (target original video frames not found)"
                        ),
                    }
                )
            continue

        # 为了更快的查找，如果尚未排序，则按时间戳排序
        df_target_original_video_frames.sort_values(
            "original_timestamp_ms", inplace=True
        )
        # 如果在处理巨大视频时性能成为问题，可以转换为 NumPy 数组以加快访问速度
        # original_ts_array = (
        #     df_target_original_video_frames['original_timestamp_ms'].to_numpy()
        # )

        for item in segment_data_list:
            edited_frame_data = item["edited_frame_data"]
            edited_ts = edited_frame_data["edited_timestamp_ms"]
            ideal_original_ts = edited_ts + target_segment_offset_ms

            # 在 df_target_original_video_frames 中找到最接近的帧
            # 这是此步骤中精炼匹配的核心
            df_target_original_video_frames["time_diff"] = (
                df_target_original_video_frames["original_timestamp_ms"]
                - ideal_original_ts
            ).abs()

            # 获取 time_diff 最小的行
            # 如果多个帧具有完全相同的最小 time_diff (对于毫秒级时间戳很少见),
            # 这里会选择第一个。你可以添加一个决胜局规则（例如，如果可用且需要，使用pHash）。
            if (
                df_target_original_video_frames.empty
            ):  # 上面应该已经捕获，但做防御性编程
                final_refined_match_series = pd.Series(
                    item["chosen_match_info"]
                )  # 备用方案
                comment = (
                    "Used chosen_match "
                    "(target original video frames empty at match time)"
                )
            else:
                final_refined_match_idx = df_target_original_video_frames[
                    "time_diff"
                ].idxmin()
                final_refined_match_series = df_target_original_video_frames.loc[
                    final_refined_match_idx
                ]
                comment = "Refined from full original video data"

            all_refined_matches_list.append(
                {
                    "edited_frame_number": edited_frame_data["edited_frame_number"],
                    "edited_timestamp_ms": edited_ts,
                    "refined_original_video_name": final_refined_match_series[
                        "original_video_name"
                    ],
                    "refined_original_frame_number": final_refined_match_series[
                        "original_frame_number"
                    ],
                    "refined_original_timestamp_ms": final_refined_match_series[
                        "original_timestamp_ms"
                    ],
                    # 在这里 pHash 距离不是直接可比的，因为我们主要不是用 pHash 来进行这一步的选择
                    "refined_phash_distance": final_refined_match_series.get(
                        "phash_distance", -1
                    ),  # 如果 pHash 在 df_all_original_frames 中
                    "comment": comment,
                }
            )

    if not all_refined_matches_list:
        print("No refined matches were generated. Cannot create final segments.")
        return

    df_refined_matches = pd.DataFrame(all_refined_matches_list)
    df_refined_matches.sort_values(by="edited_frame_number", inplace=True)

    # --- 6. 为 FFmpeg 生成最终片段 ---
    final_segments_for_ffmpeg = []
    if df_refined_matches.empty:
        print("df_refined_matches is empty. No segments to output.")
    else:
        current_ffmpeg_segment = {}
        for _, row in df_refined_matches.iterrows():
            if pd.isna(row["refined_original_video_name"]) or pd.isna(
                row["refined_original_frame_number"]
            ):
                if current_ffmpeg_segment:
                    final_segments_for_ffmpeg.append(current_ffmpeg_segment)
                    current_ffmpeg_segment = {}
                continue

            if not current_ffmpeg_segment:
                current_ffmpeg_segment = {
                    "edited_start_frame": row["edited_frame_number"],
                    "edited_start_time_ms": row["edited_timestamp_ms"],
                    "original_video_name": row["refined_original_video_name"],
                    "original_start_frame": row["refined_original_frame_number"],
                    "original_start_time_ms": row["refined_original_timestamp_ms"],
                    "edited_end_frame": row["edited_frame_number"],
                    "edited_end_time_ms": row["edited_timestamp_ms"],
                    "original_end_frame": row["refined_original_frame_number"],
                    "original_end_time_ms": row["refined_original_timestamp_ms"],
                }
            else:
                is_continuation = (
                    row["refined_original_video_name"]
                    == current_ffmpeg_segment["original_video_name"]
                    and int(row["refined_original_frame_number"])
                    == int(current_ffmpeg_segment["original_end_frame"])
                    + 1  # 确保是整数比较
                    and int(row["edited_frame_number"])
                    == int(current_ffmpeg_segment["edited_end_frame"]) + 1
                )
                if is_continuation:
                    current_ffmpeg_segment["edited_end_frame"] = row[
                        "edited_frame_number"
                    ]
                    current_ffmpeg_segment["edited_end_time_ms"] = row[
                        "edited_timestamp_ms"
                    ]
                    current_ffmpeg_segment["original_end_frame"] = row[
                        "refined_original_frame_number"
                    ]
                    current_ffmpeg_segment["original_end_time_ms"] = row[
                        "refined_original_timestamp_ms"
                    ]
                else:
                    final_segments_for_ffmpeg.append(current_ffmpeg_segment)
                    current_ffmpeg_segment = {
                        "edited_start_frame": row["edited_frame_number"],
                        "edited_start_time_ms": row["edited_timestamp_ms"],
                        "original_video_name": row["refined_original_video_name"],
                        "original_start_frame": row["refined_original_frame_number"],
                        "original_start_time_ms": row["refined_original_timestamp_ms"],
                        "edited_end_frame": row["edited_frame_number"],
                        "edited_end_time_ms": row["edited_timestamp_ms"],
                        "original_end_frame": row["refined_original_frame_number"],
                        "original_end_time_ms": row["refined_original_timestamp_ms"],
                    }
        if current_ffmpeg_segment:
            final_segments_for_ffmpeg.append(current_ffmpeg_segment)

    df_final_segments = pd.DataFrame(final_segments_for_ffmpeg)
    if not df_final_segments.empty:
        output_path = os.path.join(ABS_OUTPUT_DIR, FINAL_SEGMENTS_CSV_FILENAME)
        df_final_segments.to_csv(output_path, index=False)
        print(f"Final refined segments saved to: {output_path}")
    else:
        print("No final segments were generated for FFmpeg.")


# --- 使用示例 ---
if __name__ == "__main__":

    main_step4()
