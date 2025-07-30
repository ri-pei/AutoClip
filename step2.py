import os
import glob
import re
import pandas as pd
from PIL import Image
import imagehash  # pip install imagehash Pillow
import cv2  # pip install opencv-python
from tqdm import tqdm

# --- Configuration ---
CWD = os.getcwd()
OUTPUT_DIR_NAME = "output"
OUTPUT_PATH = os.path.join(CWD, OUTPUT_DIR_NAME)

# --- Helper Functions ---


def parse_frame_filename(filename):
    """
    Parses the frame filename to extract video name prefix, frame number, and time string.
    Example: <video_filename_no_ext>_frame_0000000_time_HH-MM-SS-mmm.png
    Returns: (video_prefix, frame_number, time_str) or (None, None, None) if no match.
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


def calculate_phash_for_video(video_name_no_ext):
    """
    Calculates pHash for all frames of a given video and saves to a CSV.
    Implements caching to skip if CSV already exists and is valid.
    """
    frames_dir = os.path.join(OUTPUT_PATH, video_name_no_ext, "frames")
    csv_cache_path = os.path.join(
        OUTPUT_PATH, video_name_no_ext, f"{video_name_no_ext}_phash.csv"
    )

    if not os.path.exists(frames_dir):
        print(
            f"Frames directory not found for {video_name_no_ext} at {frames_dir}. Skipping."
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
                    f"Valid pHash cache found for {video_name_no_ext} with {len(df_cache)} frames. Skipping."
                )
                return
            else:
                print(
                    f"Invalid or incomplete pHash cache for {video_name_no_ext}. Regenerating..."
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
            relative_image_path = os.path.relpath(full_image_path, CWD)

            frame_data_list.append(
                {
                    "video_name": video_name_no_ext,  # Use the directory name as the definitive video name
                    "image_filename": image_filename,
                    "frame_number": frame_num,
                    "timestamp_ms": timestamp_ms,
                    "phash": str(hash_val),
                    "image_path": relative_image_path.replace(
                        "\\", "/"
                    ),  # Ensure forward slashes for consistency
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

    if not os.path.exists(OUTPUT_PATH):
        print(
            f"Error: Output directory '{OUTPUT_PATH}' not found. Please run Step 1 first."
        )
        return

    # Identify video subdirectories in the output path
    # These subdirectories are named after the video files (without extension)
    video_subdirs = [
        d
        for d in os.listdir(OUTPUT_PATH)
        if os.path.isdir(os.path.join(OUTPUT_PATH, d))
    ]

    if not video_subdirs:
        print(
            f"No video-specific subdirectories found in '{OUTPUT_PATH}'. Ensure Step 1 completed correctly."
        )
        return

    for video_name_no_ext in video_subdirs:
        print(f"\nProcessing video: {video_name_no_ext}")
        calculate_phash_for_video(video_name_no_ext)

    print("\n--- Step 2 completed. ---")


if __name__ == "__main__":

    # --- Run Step 2 ---
    main_step2()

    # --- Example of how to re-run to test caching ---
    print("\n--- Re-running Step 2 to test caching ---")
    main_step2()

    # --- Example of how to invalidate a cache (e.g., by deleting one frame or the CSV) ---
    # print("\n--- Testing cache invalidation ---")
    # test_invalidate_video = "original_video1"
    # test_csv_path = os.path.join(OUTPUT_PATH, test_invalidate_video, f"{test_invalidate_video}_phash.csv")
    # if os.path.exists(test_csv_path):
    #     os.remove(test_csv_path)
    #     print(f"Deleted cache file: {test_csv_path} for testing invalidation.")
    #     main_step2()
    # else:
    #     print(f"Cache file {test_csv_path} not found, couldn't test invalidation by deletion.")
