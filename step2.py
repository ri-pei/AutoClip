import os
import glob
import imagehash
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import parse_frame_filename, time_str_to_milliseconds
from common import ABS_OUTPUT_DIR


def calculate_phash_for_video(video_name_no_ext):
    """
    为指定视频的所有帧计算pHash并保存到CSV文件。
    实现缓存机制，如果CSV已存在且有效则跳过处理。
    """
    # 使用从 common.py 导入的绝对路径
    frames_dir = os.path.join(ABS_OUTPUT_DIR, video_name_no_ext, "frames")
    csv_cache_path = os.path.join(
        ABS_OUTPUT_DIR, video_name_no_ext, f"{video_name_no_ext}_phash.csv"
    )

    if not os.path.exists(frames_dir):
        print(
            f"Frames directory not found for {video_name_no_ext} at "
            f"{frames_dir}. Skipping."
        )
        return

    # Get actual number of frame images in the directory
    actual_frame_files = glob.glob(os.path.join(frames_dir, "*.png"))
    actual_frame_count = len(actual_frame_files)

    if actual_frame_count == 0:
        print(f"No frames found in {frames_dir} for {video_name_no_ext}. Skipping.")
        return

    # Cache check
    if os.path.exists(csv_cache_path):
        try:
            df_cache = pd.read_csv(csv_cache_path)
            # Validate cache: check video name and frame count
            if (
                not df_cache.empty
                and df_cache["video_name"].iloc[0] == video_name_no_ext
                and len(df_cache) == actual_frame_count
            ):
                print(
                    f"Valid pHash cache found for {video_name_no_ext} with "
                    f"{len(df_cache)} frames. Skipping."
                )
                return
            else:
                print(
                    f"Invalid or incomplete pHash cache for "
                    f"{video_name_no_ext}. Regenerating..."
                )
        except pd.errors.EmptyDataError:
            print(f"Empty pHash cache file for {video_name_no_ext}. Regenerating...")
        except Exception as e:
            print(f"Error reading cache for {video_name_no_ext}: {e}. Regenerating...")

    frame_data_list = []

    # Sort files to ensure consistent order, though filenames should already imply order
    # Using os.listdir and then sorting is often more reliable across systems than glob
    image_filenames = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith(".png")]
    )

    print(f"Processing {len(image_filenames)} frames for {video_name_no_ext}...")
    for image_filename in tqdm(image_filenames, desc=f"Hashing {video_name_no_ext}"):
        full_image_path = os.path.join(frames_dir, image_filename)

        parsed_video_prefix, frame_num, time_str = parse_frame_filename(image_filename)

        if frame_num is None:
            print(f"Could not parse filename: {image_filename}. Skipping.")
            continue

        try:
            # OpenCV is generally faster for reading, Pillow for hashing
            # The images were already processed (resized, masked) in Step 1
            img_cv = cv2.imread(full_image_path)
            if img_cv is None:
                print(f"Warning: Could not read image {full_image_path}. Skipping.")
                continue

            # Convert OpenCV BGR image to PIL RGB image
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            # hash_val = imagehash.phash(img_pil)
            hash_val = imagehash.phash(img_pil, hash_size=16)

            timestamp_ms = time_str_to_milliseconds(time_str)

            # replace("\\", "/") 确保路径在不同操作系统中的一致性
            absolute_image_path = full_image_path.replace("\\", "/")

            frame_data_list.append(
                {
                    # Use the directory name as the definitive video name
                    "video_name": video_name_no_ext,
                    "image_filename": image_filename,
                    "frame_number": frame_num,
                    "timestamp_ms": timestamp_ms,
                    "phash": str(hash_val),
                    "image_path": absolute_image_path,
                }
            )

        except Exception as e:
            print(
                f"Error processing frame {image_filename} for {video_name_no_ext}: {e}"
            )

    if frame_data_list:
        df_frames = pd.DataFrame(frame_data_list)
        # Ensure output/<video_name_no_ext>/ directory exists for the CSV
        os.makedirs(os.path.dirname(csv_cache_path), exist_ok=True)
        df_frames.to_csv(
            csv_cache_path, index=False, lineterminator="\n"
        )  # Ensure no blank lines
        print(f"Saved pHash data for {video_name_no_ext} to {csv_cache_path}")
    else:
        print(f"No frame data collected for {video_name_no_ext}.")


def main_step2():
    """
    Main function for Step 2: Calculate pHash for all videos.
    """
    print("--- Running Step 2: Calculate and Cache pHash for Frames ---")

    # 使用从 common.py 导入的路径进行检查
    if not os.path.exists(ABS_OUTPUT_DIR):
        print(
            f"Error: Output directory '{ABS_OUTPUT_DIR}' not found. "
            "Please run Step 1 first."
        )
        return

    # 在配置好的输出目录中查找视频子目录
    video_subdirs = [
        d
        for d in os.listdir(ABS_OUTPUT_DIR)
        if os.path.isdir(os.path.join(ABS_OUTPUT_DIR, d))
    ]

    if not video_subdirs:
        print(
            f"No video-specific subdirectories found in '{ABS_OUTPUT_DIR}'. "
            "Ensure Step 1 completed correctly."
        )
        return

    for video_name_no_ext in video_subdirs:
        print(f"\nProcessing video: {video_name_no_ext}")
        calculate_phash_for_video(video_name_no_ext)

    print("\n--- Step 2 completed. ---")


if __name__ == "__main__":

    # --- Run Step 2 ---
    main_step2()

    # --- 示例：重新运行以测试缓存 ---
    # print("\n--- Re-running Step 2 to test caching ---")
    # main_step2()

    # --- Example of how to invalidate a cache (by deleting one frame or the CSV) ---
    # print("\n--- Testing cache invalidation ---")
    # test_invalidate_video = "original_video1"
    # test_csv_path = os.path.join(
    #     OUTPUT_PATH,
    #     test_invalidate_video,
    #     f"{test_invalidate_video}_phash.csv",
    # )
    # if os.path.exists(test_csv_path):
    #     os.remove(test_csv_path)
    #     print(f"Deleted cache file: {test_csv_path} for testing invalidation.")
    #     main_step2()
    # else:
    #     print(
    #         f"Cache file {test_csv_path} not found, "
    #         "Couldn't test invalidation by deletion."
    #     )
