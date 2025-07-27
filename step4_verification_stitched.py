import pandas as pd
import os
import cv2  # Using OpenCV
import glob
import subprocess # For running ffmpeg
from tqdm import tqdm # For progress bar

# --- Configuration ---
WORKING_DIR = "."
OUTPUT_DIR_NAME = "output"
FINAL_SEGMENTS_CSV_FILENAME = "final_video_segments_refined.csv" # Input CSV
EDITED_VIDEO_FILENAME = "my_edited_clip.mp4"  # Your original edited video, for audio track

STITCHED_IMAGES_OUTPUT_SUBDIR = "final_verification_stitched"
FINAL_VERIFICATION_VIDEO_FILENAME = "final_verification_video.mp4" # Output video

ADD_TEXT_LABELS_CV2 = True
FONT_CV2 = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_CV2 = 0.7
FONT_COLOR_CV2 = (0, 255, 255)  # Yellow in BGR
TEXT_THICKNESS_CV2 = 2

# --- Helper Functions ---
def get_frame_num_from_filename(filename_str):
    if not isinstance(filename_str, str): return None
    try:
        return int(filename_str.split('_frame_')[-1].split('_time_')[0])
    except: return None

def map_frame_numbers_to_paths(frames_dir):
    mapping = {}
    if not os.path.isdir(frames_dir):
        print(f"Warning: Frames directory not found: {frames_dir}")
        return mapping
    frame_files = glob.glob(os.path.join(frames_dir, "*.png"))
    for f_path in frame_files:
        frame_num = get_frame_num_from_filename(os.path.basename(f_path))
        if frame_num is not None:
            if frame_num not in mapping:
                 mapping[frame_num] = f_path
    return mapping

def stitch_images_vertically_cv2(image_top_path, image_bottom_path, output_path,
                                 edited_frame_num_text=None, original_frame_num_text=None):
    if os.path.exists(output_path): # Cache check
        return "cached" 

    try:
        img_top = cv2.imread(image_top_path)
        img_bottom = cv2.imread(image_bottom_path)

        if img_top is None:
            print(f"Error: Could not read top image: {image_top_path}")
            return "error_top"
        if img_bottom is None:
            print(f"Error: Could not read bottom image: {image_bottom_path}")
            return "error_bottom"

        h1, w1 = img_top.shape[:2]
        h2, w2 = img_bottom.shape[:2]

        if w1 != w2:
            target_height_img_bottom = int(h2 * (w1 / w2))
            img_bottom_resized = cv2.resize(img_bottom, (w1, target_height_img_bottom), interpolation=cv2.INTER_AREA)
            stitched_image = cv2.vconcat([img_top, img_bottom_resized])
        else:
            stitched_image = cv2.vconcat([img_top, img_bottom])

        if ADD_TEXT_LABELS_CV2:
            if edited_frame_num_text:
                cv2.putText(stitched_image, f"Edited: {edited_frame_num_text}", (10, 30), FONT_CV2, FONT_SCALE_CV2, FONT_COLOR_CV2, TEXT_THICKNESS_CV2)
            if original_frame_num_text:
                cv2.putText(stitched_image, f"Original: {original_frame_num_text}", (10, h1 + 30), FONT_CV2, FONT_SCALE_CV2, FONT_COLOR_CV2, TEXT_THICKNESS_CV2)
        
        cv2.imwrite(output_path, stitched_image)
        return "stitched"
    except Exception as e:
        print(f"Error stitching images {os.path.basename(image_top_path)} and {os.path.basename(image_bottom_path)} with OpenCV: {e}")
        return "error_stitch"

def get_video_framerate(video_path):
    """Attempts to get framerate using ffprobe."""
    try:
        ffprobe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        num, den = map(int, result.stdout.strip().split('/'))
        return num / den
    except FileNotFoundError:
        print("Error: ffprobe not found. Please ensure ffmpeg (which includes ffprobe) is installed and in your system's PATH.")
        return None
    except (subprocess.CalledProcessError, ValueError, ZeroDivisionError) as e:
        print(f"Error getting framerate for {video_path} with ffprobe: {e}")
        print(f"ffprobe output: {result.stdout if 'result' in locals() else 'N/A'}")
        print(f"ffprobe stderr: {result.stderr if 'result' in locals() else 'N/A'}")
        return None

# --- Main Script ---
if __name__ == "__main__":
    abs_working_dir = os.path.abspath(WORKING_DIR)
    base_output_dir = os.path.join(abs_working_dir, OUTPUT_DIR_NAME)
    segments_csv_path = os.path.join(base_output_dir, FINAL_SEGMENTS_CSV_FILENAME)
    
    stitched_img_abs_dir = os.path.join(base_output_dir, STITCHED_IMAGES_OUTPUT_SUBDIR)
    os.makedirs(stitched_img_abs_dir, exist_ok=True)

    abs_edited_video_path = os.path.join(abs_working_dir, EDITED_VIDEO_FILENAME) # Path to original edited video for audio

    if not os.path.exists(segments_csv_path):
        print(f"Error: Segments CSV file not found at {segments_csv_path}")
        exit()
    if not os.path.exists(abs_edited_video_path):
        print(f"Error: Original edited video file '{abs_edited_video_path}' not found. Needed for audio track.")
        exit()

    try:
        df_segments = pd.read_csv(segments_csv_path)
    except Exception as e:
        print(f"Error reading segments CSV {segments_csv_path}: {e}")
        exit()
        
    if df_segments.empty:
        print("Segments CSV is empty. Nothing to process.")
        exit()

    EDITED_VIDEO_NAME_NO_EXT_cfg = None # Use this for finding frame image paths
    # Infer EDITED_VIDEO_NAME_NO_EXT for frame paths (not for the EDITED_VIDEO_FILENAME for audio)
    coarse_match_csv_path = os.path.join(base_output_dir, "coarse_match_results.csv") 
    if os.path.exists(coarse_match_csv_path):
        try:
            df_coarse_temp = pd.read_csv(coarse_match_csv_path, nrows=1)
            if not df_coarse_temp.empty and 'edited_frame_filename' in df_coarse_temp.columns:
                first_ef_filename = df_coarse_temp['edited_frame_filename'].iloc[0]
                EDITED_VIDEO_NAME_NO_EXT_cfg = first_ef_filename.split('_frame_')[0]
                print(f"Inferred EDITED_VIDEO_NAME_NO_EXT (for frame paths): {EDITED_VIDEO_NAME_NO_EXT_cfg}")
        except Exception as e:
            print(f"Could not infer edited video name for frame paths from coarse_match_results.csv: {e}")
    
    if EDITED_VIDEO_NAME_NO_EXT_cfg is None:
        EDITED_VIDEO_NAME_NO_EXT_cfg = input("Could not infer edited video name for frame paths. Please enter the name (without extension) of the edited video's frame source: ")
        if not EDITED_VIDEO_NAME_NO_EXT_cfg:
            print("Edited video name (for frame paths) is required. Exiting.")
            exit()

    edited_video_frames_dir = os.path.join(base_output_dir, EDITED_VIDEO_NAME_NO_EXT_cfg, "frames")
    print(f"Mapping edited video frames from: {edited_video_frames_dir}")
    edited_frames_map = map_frame_numbers_to_paths(edited_video_frames_dir)
    if not edited_frames_map:
        print(f"Error: No edited frames found or mapped in {edited_video_frames_dir}.")

    original_videos_frames_maps = {}
    unique_original_videos = df_segments['original_video_name'].unique()
    for orig_video_name in unique_original_videos:
        if pd.isna(orig_video_name): continue
        orig_video_frames_dir = os.path.join(base_output_dir, orig_video_name, "frames")
        original_videos_frames_maps[orig_video_name] = map_frame_numbers_to_paths(orig_video_frames_dir)
        if not original_videos_frames_maps[orig_video_name]:
             print(f"Warning: No frames mapped for original video '{orig_video_name}' in {orig_video_frames_dir}.")

    processed_count = 0
    stitched_newly_count = 0
    cached_count = 0
    error_count = 0
    
    # --- Stitching Images ---
    print("\n--- Starting Image Stitching ---")
    # We need to know the full range of edited frame numbers to create a complete image sequence for ffmpeg
    if edited_frames_map:
        min_edited_frame_num_overall = min(edited_frames_map.keys())
        max_edited_frame_num_overall = max(edited_frames_map.keys())
        print(f"Overall edited frame range for sequence: {min_edited_frame_num_overall} to {max_edited_frame_num_overall}")
    else: # if edited_frames_map is empty, we can't determine range. This is an issue.
        min_edited_frame_num_overall = df_segments['edited_start_frame'].min() if not df_segments.empty else 0
        max_edited_frame_num_overall = df_segments['edited_end_frame'].max() if not df_segments.empty else 0
        if not edited_frames_map and df_segments.empty:
             print("CRITICAL: No edited frames mapped and segment CSV is empty. Cannot determine frame range for video.")
             exit()
        elif not edited_frames_map:
             print("Warning: Edited frames map is empty. Using segment CSV for frame range. This might lead to missing frames in video if CSV doesn't cover all.")


    # Create a dictionary to hold segment info for quick lookup
    segment_lookup = {}
    for _, segment_row in df_segments.iterrows():
        original_video_name = segment_row['original_video_name']
        seg_edited_start = int(segment_row['edited_start_frame'])
        seg_edited_end = int(segment_row['edited_end_frame'])
        seg_orig_start = int(segment_row['original_start_frame'])
        for i in range(seg_edited_end - seg_edited_start + 1):
            ef_num = seg_edited_start + i
            of_num = seg_orig_start + i
            segment_lookup[ef_num] = {
                'original_video_name': original_video_name,
                'original_frame_number': of_num
            }

    # Iterate through the complete range of edited frames to ensure all are processed for video
    for current_edited_frame_num in tqdm(range(int(min_edited_frame_num_overall), int(max_edited_frame_num_overall) + 1), desc="Stitching Images"):
        processed_count += 1
        edited_frame_path = edited_frames_map.get(current_edited_frame_num)
        
        if not edited_frame_path:
            # print(f"Warning: Edited frame {current_edited_frame_num} not found in map. Placeholder needed or skip for video.")
            # For now, we'll just skip, ffmpeg will complain about missing frame.
            # A more robust solution would be to create a placeholder image.
            error_count += 1
            continue

        match_info = segment_lookup.get(current_edited_frame_num)
        if not match_info:
            # print(f"Warning: No segment match info for edited frame {current_edited_frame_num}. Cannot find original pair.")
            # This frame of the edited video might not have a corresponding original segment.
            # Create a "stitched" image with only the top part or a placeholder for the bottom.
            # For simplicity, we'll skip making a pair for this.
            error_count += 1
            continue

        original_video_name = match_info['original_video_name']
        original_frame_num = match_info['original_frame_number']

        if pd.isna(original_video_name) or original_video_name not in original_videos_frames_maps:
            # print(f"Warning: Original video {original_video_name} map not found for edited frame {current_edited_frame_num}.")
            error_count += 1
            continue
            
        current_original_frames_map = original_videos_frames_maps[original_video_name]
        original_frame_path = current_original_frames_map.get(original_frame_num)

        if not original_frame_path:
            # print(f"Warning: Original frame {original_frame_num} (video: {original_video_name}) not found for edited frame {current_edited_frame_num}.")
            error_count += 1
            continue

        output_filename = f"stitched_edited_frame_{current_edited_frame_num:07d}.png" # Consistent naming for ffmpeg
        output_full_path = os.path.join(stitched_img_abs_dir, output_filename)
        
        ef_text = f"F:{current_edited_frame_num}"
        of_text = f"{str(original_video_name)[:15]} F:{original_frame_num}"

        result = stitch_images_vertically_cv2(edited_frame_path, original_frame_path, output_full_path,
                                            edited_frame_num_text=ef_text, original_frame_num_text=of_text)
        if result == "stitched":
            stitched_newly_count += 1
        elif result == "cached":
            cached_count += 1
        else: # error
            error_count += 1
            
    print(f"\n--- Image Stitching Summary ---")
    print(f"Total edited frames considered for sequence: {processed_count}")
    print(f"  Newly stitched images: {stitched_newly_count}")
    print(f"  Loaded from cache: {cached_count}")
    print(f"  Errors or missing pairs: {error_count}")
    
    if (stitched_newly_count + cached_count) == 0:
        print("No images were stitched or found in cache. Cannot create video.")
        exit()

    # --- Create Video from Stitched Images using FFmpeg ---
    print("\n--- Creating Video from Stitched Images ---")
    
    framerate = get_video_framerate(abs_edited_video_path)
    if framerate is None:
        print("Could not determine framerate of edited video. Using default 25 fps for output.")
        framerate = 25 # Default if ffprobe fails or not found

    output_video_path = os.path.join(base_output_dir, FINAL_VERIFICATION_VIDEO_FILENAME)
    
    # Ensure the start number for ffmpeg matches the first frame generated
    # The glob pattern for images will pick them up if they are named sequentially.
    # We need to provide the start_number if the sequence doesn't start at 0 or 1.
    # The `stitched_edited_frame_%07d.png` pattern assumes a sequence.
    # The `min_edited_frame_num_overall` might be the actual start number if all frames from that number exist.
    # However, ffmpeg's image sequence input usually assumes images are frame_001.png, frame_002.png etc.,
    # or if using -start_number, it refers to the first number in the %07d pattern.
    # If our files are `stitched_edited_frame_0000123.png`, `0000124.png`, etc.,
    # then `%07d` works, and we might need `-start_number 123`.
    # Let's assume the *lowest file number generated* is our start_number for ffmpeg.
    
    # Find the actual first numbered stitched image to set -start_number correctly
    all_stitched_images = sorted(glob.glob(os.path.join(stitched_img_abs_dir, "stitched_edited_frame_*.png")))
    if not all_stitched_images:
        print("No stitched images found in the output directory. Cannot create video.")
        exit()

    try:
        # Extract the number from the first filename, e.g., "stitched_edited_frame_0000123.png" -> 123
        first_stitched_filename = os.path.basename(all_stitched_images[0])
        start_number_for_ffmpeg = int(first_stitched_filename.split('_')[-1].replace('.png', ''))
    except (IndexError, ValueError) as e:
        print(f"Could not determine start number from first stitched image filename: {first_stitched_filename}. Error: {e}")
        print("Assuming start number 0 for ffmpeg, which might be incorrect.")
        start_number_for_ffmpeg = 0 # Fallback, might be problematic if files don't start near 0.

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(framerate),
        '-start_number', str(start_number_for_ffmpeg), # Crucial for correctly picking up image sequence
        '-i', os.path.join(stitched_img_abs_dir, 'stitched_edited_frame_%07d.png'),
        '-i', abs_edited_video_path,  # Input for audio
        '-c:v', 'libx264',
        '-preset', 'ultrafast',  # Speed priority for video encoding
        '-crf', '23',            # Decent quality, adjust if needed (lower is better quality, larger file)
        '-pix_fmt', 'yuv420p',   # For compatibility
        '-c:a', 'aac',           # Or 'copy' if you want to directly copy the audio stream
        '-b:a', '192k',          # Audio bitrate if re-encoding (not used if -c:a copy)
        '-map', '0:v:0',         # Map video from the first input (images)
        '-map', '1:a:0?',        # Map audio from the second input (original video), '?' makes it optional
        '-shortest',             # Finish encoding when the shortest input stream ends (usually audio or video)
        output_video_path
    ]
    if '-c:a' == 'copy': # If copying audio, no need for bitrate
        ffmpeg_cmd.remove('-b:a')
        ffmpeg_cmd.remove('192k')


    print("\nExecuting FFmpeg command:")
    print(" ".join(ffmpeg_cmd))

    try:
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"\nSuccessfully created verification video: {output_video_path}")
        else:
            print(f"\nError during FFmpeg execution (Return Code: {process.returncode}):")
            print("FFmpeg STDOUT:")
            print(stdout.decode(errors='ignore'))
            print("FFmpeg STDERR:")
            print(stderr.decode(errors='ignore'))
    except FileNotFoundError:
        print("\nError: ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH.")
    except Exception as e:
        print(f"\nAn unexpected error occurred while running FFmpeg: {e}")

    print("\n--- Script Finished ---")