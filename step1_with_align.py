import os
import subprocess
import cv2 # OpenCV for image processing and transform estimation
import numpy as np # For numerical operations with OpenCV
import glob
import shutil
import json

# --- 用户核心配置参数 ---
WORKING_DIR = "."
EDITED_VIDEO_FILENAME = "my_edited_clip.mp4" # 您的剪辑后视频文件名
MASK_RECT = (1085, 36, 1819-1085, 152-36) # (x,y,w,h) e.g. (0, 1080-80, 1920, 80) or None

# --- 新增：用户提供的参考帧路径 ---
# 用户必须提供这两个文件！它们应该是PNG格式的单帧图像。
# 内容必须是同一时刻的，用于计算变换。
REF_ORIGINAL_FRAME_PATH = "ref_original.png" # 例如: "input_素材1_frame_0000100_time_00-00-04-000.png"
REF_EDITED_FRAME_PATH = "ref_edited.png"     # 例如: "剪辑视频_frame_0000025_time_00-00-01-000.png"

# --- 新增：可选的用户提供的LUT文件路径 ---
# 如果用户有一个.cube文件用于颜色校正，请指定路径，否则设为None
USER_COLOR_LUT_PATH = None # 例如: "my_color_correction.cube"
# --- END 用户核心配置参数 ---


# --- 内部常量 ---
BASE_OUTPUT_DIR_NAME = "output"
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.flv')
MIN_MATCH_COUNT_GEO = 10 # SIFT/ORB匹配的最小特征点数
# --- END 内部常量 ---

# 全局变量存储计算出的变换参数
GEOMETRIC_TRANSFORM_PARAMS = None # Calculated once


def run_command(command_list):
    """执行外部命令并返回结果，包括进程对象以便检查returncode"""
    try:
        # print(f"  DEBUG CMD: {' '.join(command_list)}") # Uncomment for debugging ffmpeg commands
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        stdout, stderr = process.communicate()
        # We return the process object as well, so the caller can check process.returncode
        return stdout, stderr, process
    except FileNotFoundError:
        print(f"Error: Command {command_list[0]} not found. Is ffmpeg/ffprobe installed and in PATH?")
        raise
    except Exception as e:
        print(f"An unexpected error occurred with command {' '.join(command_list)}: {e}")
        raise

def get_video_metadata(video_path):
    # (保持不变)
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate', '-of', 'json', video_path
    ]
    stdout, _, process = run_command(cmd)
    if process.returncode != 0 and not stdout : # If error and no stdout, likely critical
        print(f"ffprobe failed to get metadata for {os.path.basename(video_path)} (return code {process.returncode})")
        return None
    if stdout:
        try:
            data = json.loads(stdout)
            if not data.get('streams'):
                # print(f"Warning: ffprobe returned no streams for metadata of {os.path.basename(video_path)}. Output: {stdout[:200]}")
                return None
            metadata = data['streams'][0]
            if isinstance(metadata.get('avg_frame_rate'), str) and '/' in metadata['avg_frame_rate']:
                num, den = map(int, metadata['avg_frame_rate'].split('/'))
                metadata['avg_frame_rate'] = num / den if den != 0 else 0
            else:
                metadata['avg_frame_rate'] = float(metadata.get('avg_frame_rate', 0))
            return metadata
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            # print(f"Error parsing ffprobe JSON for metadata of {os.path.basename(video_path)}: {e}. Output: {stdout[:200]}")
            return None
    return None


def get_frame_timestamps_map_json(video_path):
    # (保持不变)
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'frame=best_effort_timestamp_time,pts_time,media_type', '-of', 'json', video_path
    ]
    stdout, _, process = run_command(cmd)
    if process.returncode != 0 and not stdout:
        print(f"ffprobe failed to get timestamps for {os.path.basename(video_path)} (return code {process.returncode})")
        return {}

    timestamps_map = {}
    if not stdout:
        # print(f"  Could not get frame timestamps (stdout empty) for {os.path.basename(video_path)}")
        return {}
    try:
        data = json.loads(stdout)
        frames_data = data.get("frames", [])
        parsed_count = 0
        for frame_index, frame_info in enumerate(frames_data):
            if frame_info.get("media_type") == "video":
                ts_str = frame_info.get("best_effort_timestamp_time", frame_info.get("pts_time"))
                if ts_str is not None:
                    try:
                        timestamps_map[frame_index] = float(ts_str)
                        parsed_count += 1
                    except (ValueError, TypeError): pass
        # if parsed_count == 0 and len(frames_data) > 0:
            # print(f"  Warning: No timestamps parsed for {os.path.basename(video_path)} from {len(frames_data)} ffprobe entries.")
    except json.JSONDecodeError as e:
        # print(f"  Error decoding ffprobe JSON for timestamps of {os.path.basename(video_path)}: {e}. Output: {stdout[:200]}")
        return {}
    except Exception as e: # Catch any other unexpected error
        # print(f"  Unexpected error processing ffprobe JSON for {os.path.basename(video_path)}: {e}")
        return {}
    return timestamps_map


def format_timestamp_from_seconds(ts_seconds):
    # (保持不变)
    if ts_seconds < 0: ts_seconds = 0
    hours = int(ts_seconds / 3600); minutes = int((ts_seconds % 3600) / 60)
    seconds = int(ts_seconds % 60); milliseconds = int((ts_seconds - int(ts_seconds)) * 1000)
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}-{milliseconds:03d}"

def estimate_geometric_transform_from_refs(ref_original_path, ref_edited_path):
    """
    Estimates geometric transform (crop and scale) from a reference original frame
    to a reference edited frame using feature matching and homography.
    Returns a dictionary for ffmpeg's crop and scale filters, or None.
    """
    print(f"  Estimating geometric transform: '{os.path.basename(ref_original_path)}' vs '{os.path.basename(ref_edited_path)}'")
    img_orig = cv2.imread(ref_original_path)
    img_edit = cv2.imread(ref_edited_path)

    if img_orig is None:
        print(f"    Error: Could not read original reference image: {ref_original_path}")
        return None
    if img_edit is None:
        print(f"    Error: Could not read edited reference image: {ref_edited_path}")
        return None

    h_orig, w_orig = img_orig.shape[:2]
    h_edit, w_edit = img_edit.shape[:2]

    try: # Try SIFT first (more robust to scale)
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
        print("    Using SIFT for feature detection.")
    except AttributeError:
        print("    SIFT not available (try 'pip install opencv-contrib-python'). Falling back to ORB.")
        detector = cv2.ORB_create(nfeatures=2000) # More features for potentially better matching
        norm_type = cv2.NORM_HAMMING
    
    kp_orig, des_orig = detector.detectAndCompute(img_orig, None)
    kp_edit, des_edit = detector.detectAndCompute(img_edit, None)

    if des_orig is None or len(kp_orig) < MIN_MATCH_COUNT_GEO :
        print(f"    Error: Not enough keypoints/descriptors in original reference ({len(kp_orig) if kp_orig is not None else 0}).")
        return None
    if des_edit is None or len(kp_edit) < MIN_MATCH_COUNT_GEO:
        print(f"    Error: Not enough keypoints/descriptors in edited reference ({len(kp_edit) if kp_edit is not None else 0}).")
        return None

    # Match descriptors
    if norm_type == cv2.NORM_L2: # SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else: # ORB
        matcher = cv2.BFMatcher(norm_type, crossCheck=False) # Use knnMatch, so crossCheck=False

    matches = matcher.knnMatch(des_orig, des_edit, k=2)

    good_matches = []
    if matches:
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < 0.75 * n.distance: # Lowe's ratio test
                    good_matches.append(m)
    
    print(f"    Found {len(good_matches)} good matches (min required: {MIN_MATCH_COUNT_GEO}).")
    if len(good_matches) < MIN_MATCH_COUNT_GEO:
        print("    Not enough good matches to estimate transform reliably.")
        # You could draw matches here for debugging if needed
        # img_matches = cv2.drawMatches(img_orig, kp_orig, img_edit, kp_edit, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imwrite(os.path.join(os.path.dirname(ref_original_path), "debug_matches.png"), img_matches)
        # print("    (Saved debug_matches.png if you want to inspect)")
        return None

    src_pts = np.float32([kp_orig[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_edit[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate Homography (more general than Affine, can handle some perspective)
    M_homo, mask_homo = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)

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

    corners_edited_frame = np.float32([
        [0, 0], [w_edit - 1, 0], [w_edit - 1, h_edit - 1], [0, h_edit - 1]
    ]).reshape(-1, 1, 2)

    projected_corners_in_orig = cv2.perspectiveTransform(corners_edited_frame, M_inv_homo)
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
    if crop_x + crop_w > w_orig: crop_w = w_orig - crop_x
    if crop_y + crop_h > h_orig: crop_h = h_orig - crop_y
    
    if crop_w <= 0 or crop_h <= 0:
        print(f"    Error: Estimated crop dimensions are invalid (w={crop_w}, h={crop_h}).")
        return None

    # The ffmpeg crop filter takes: crop=width:height:x:y
    # The ffmpeg scale filter will then scale this cropped region to the dimensions of the edited video.
    # (which is h_edit, w_edit)
    
    transform_params = {
        'crop_w': crop_w,
        'crop_h': crop_h,
        'crop_x': crop_x,
        'crop_y': crop_y,
        'scale_target_w': w_edit, # The cropped part should be scaled to this width
        'scale_target_h': h_edit  # and this height
    }
    print(f"    Estimated geometric transform for ffmpeg:")
    print(f"      Crop: w={crop_w}, h={crop_h}, x={crop_x}, y={crop_y} (from original {w_orig}x{h_orig})")
    print(f"      Scale to: w={w_edit}, h={h_edit}")
    return transform_params


def process_video_for_frames(video_path, video_name_no_ext, main_output_folder,
                                 is_edited_video_flag,
                                 final_output_resolution, # This is the resolution of the EDITED_VIDEO
                                 mask_rect_config,
                                 # New parameters
                                 geom_transform_to_apply, # Dict from estimate_geometric_transform_from_refs
                                 color_lut_to_apply,      # Path to .cube LUT file or None
                                 # ref_frame_for_histmatch # Path to ref edited frame (complex to use directly)
                                 ):
    video_specific_output_dir = os.path.join(main_output_folder, video_name_no_ext)
    frames_output_dir = os.path.join(video_specific_output_dir, "frames")

    if os.path.exists(frames_output_dir):
        final_name_pattern = os.path.join(frames_output_dir, f"{video_name_no_ext}_frame_*_time_*.png")
        if glob.glob(final_name_pattern):
            # print(f"  Frames for '{video_name_no_ext}' seem to exist. Skipping.") # Reduce verbosity
            return True
        else:
            # print(f"  Frame dir '{frames_output_dir}' exists but no final files. Cleaning.")
            shutil.rmtree(frames_output_dir)
    
    os.makedirs(frames_output_dir, exist_ok=True)
    temp_ffmpeg_output_dir = os.path.join(video_specific_output_dir, "temp_ffmpeg_frames")
    if os.path.exists(temp_ffmpeg_output_dir): shutil.rmtree(temp_ffmpeg_output_dir)
    os.makedirs(temp_ffmpeg_output_dir, exist_ok=True)

    # print(f"  Processing video '{video_name_no_ext}':")
    vf_options = []

    # --- 1. Apply Geometric Transform (Crop and Scale) - ONLY FOR ORIGINAL VIDEOS ---
    if not is_edited_video_flag and geom_transform_to_apply:
        gt = geom_transform_to_apply
        vf_options.append(f"crop={gt['crop_w']}:{gt['crop_h']}:{gt['crop_x']}:{gt['crop_y']}")
        # Scale the (now cropped) video to match the edited video's resolution
        vf_options.append(f"scale={gt['scale_target_w']}:{gt['scale_target_h']}")
    elif is_edited_video_flag:
        # For the EDITED video itself, we only scale it if its resolution differs
        # from its own reported metadata (which defines final_output_resolution).
        # This situation should be rare if final_output_resolution is derived from edited video's metadata.
        current_meta = get_video_metadata(video_path) # Get actual current res of this file
        if current_meta and \
           (current_meta['width'] != final_output_resolution['width'] or \
            current_meta['height'] != final_output_resolution['height']):
            # print(f"    Scaling edited video '{video_name_no_ext}' to its defined final output resolution.")
            vf_options.append(f"scale={final_output_resolution['width']}:{final_output_resolution['height']}")


    # --- 2. Apply Color Transformation - ONLY FOR ORIGINAL VIDEOS ---
    if not is_edited_video_flag:
        if color_lut_to_apply and os.path.exists(color_lut_to_apply):
            # Ensure path is suitable for ffmpeg (e.g., escape special chars if any - simplified here)
            lut_path_escaped = color_lut_to_apply.replace('\\', '/') # Basic path normalization
            vf_options.append(f"lut3d=file='{lut_path_escaped}'")
            # print(f"    Applying 3D LUT: {color_lut_to_apply}")
        # else if ref_frame_for_histmatch: # Direct histmatch is complex for ffmpeg CLI here
            # print(f"    (Skipping histmatch for now - direct use with image ref is complex in simple -vf chain)")
            pass

    # --- 3. Apply Masking (for ALL videos, after all other transforms) ---
    # The frame dimensions at this point should be `final_output_resolution`.
    if mask_rect_config:
        x, y, w, h = mask_rect_config
        # Mask coordinates are relative to the frame *after* all previous transforms.
        img_w_mask, img_h_mask = final_output_resolution['width'], final_output_resolution['height']
        
        valid_mask = False
        if w > 0 and h > 0:
            x1_m, y1_m = max(0, x), max(0, y)
            eff_w_m, eff_h_m = min(img_w_mask - x1_m, w), min(img_h_mask - y1_m, h)
            if eff_w_m > 0 and eff_h_m > 0:
                vf_options.append(f"drawbox=x={x1_m}:y={y1_m}:w={eff_w_m}:h={eff_h_m}:color=black:t=fill")
                valid_mask = True
        # if not valid_mask and mask_rect_config:
            # print(f"    Warning: Mask {mask_rect_config} is invalid for res {img_w_mask}x{img_h_mask}. No mask applied.")

    # --- Construct and Run FFmpeg Command ---
    ffmpeg_filter_string = ",".join(vf_options)
    temp_frame_pattern = os.path.join(temp_ffmpeg_output_dir, f"ffmpeg_frame_%07d.png")
    
    extract_cmd = ['ffmpeg', '-y', '-i', video_path, '-an', '-vsync', 'vfr']
    if ffmpeg_filter_string:
        extract_cmd.extend(['-vf', ffmpeg_filter_string])
    extract_cmd.extend(['-q:v', '2', '-start_number', '0', temp_frame_pattern])
    
    # print(f"    Running ffmpeg for '{video_name_no_ext}'...")
    # if ffmpeg_filter_string: print(f"      Filter: {ffmpeg_filter_string[:200]}{'...' if len(ffmpeg_filter_string)>200 else ''}")
    
    _, stderr_ffmpeg, process_ffmpeg = run_command(extract_cmd)

    extracted_temp_files = sorted(glob.glob(os.path.join(temp_ffmpeg_output_dir, "ffmpeg_frame_*.png")))
    
    if process_ffmpeg.returncode != 0 or not extracted_temp_files:
        print(f"    FFmpeg failed or produced no frames for {video_name_no_ext} (Code: {process_ffmpeg.returncode}).")
        if stderr_ffmpeg: print(f"      FFmpeg STDERR:\n{stderr_ffmpeg[:1000]}\n") # Print a good chunk of stderr
        if os.path.exists(temp_ffmpeg_output_dir): shutil.rmtree(temp_ffmpeg_output_dir)
        return False
    # print(f"    FFmpeg extracted {len(extracted_temp_files)} frames.")

    # --- Rename frames ---
    # print(f"    Fetching timestamps & renaming {len(extracted_temp_files)} frames...")
    frame_ts_map = get_frame_timestamps_map_json(video_path)
    renamed_count = 0
    # (Timestamp and renaming logic remains largely the same)
    # ... (Loop through extracted_temp_files, get timestamp from frame_ts_map, os.rename)
    for i, temp_processed_path in enumerate(extracted_temp_files):
        timestamp_sec = frame_ts_map.get(i, 0.0) # Default to 0.0s if TS not found
        formatted_ts = format_timestamp_from_seconds(timestamp_sec)
        final_frame_name = f"{video_name_no_ext}_frame_{i:07d}_time_{formatted_ts}.png"
        final_frame_path = os.path.join(frames_output_dir, final_frame_name)
        try:
            os.rename(temp_processed_path, final_frame_path)
            renamed_count += 1
        except Exception as e:
            print(f"    Error renaming {temp_processed_path} to {final_frame_path}: {e}")
    # print(f"    Renamed {renamed_count} frames.")
    
    if os.path.exists(temp_ffmpeg_output_dir):
        try:
            if not os.listdir(temp_ffmpeg_output_dir): os.rmdir(temp_ffmpeg_output_dir)
            # else: print(f"    Warning: Temp dir '{temp_ffmpeg_output_dir}' not empty.")
        except OSError: pass # Ignore error if dir is not empty or other issue
            
    if renamed_count == 0 and len(extracted_temp_files) > 0:
        print(f"    WARNING: Extracted frames but NONE renamed for {video_name_no_ext}.")
        return False
    return True


def main_step1():
    global GEOMETRIC_TRANSFORM_PARAMS # Use the global variable

    abs_working_dir = os.path.abspath(WORKING_DIR)
    abs_main_output_dir = os.path.join(abs_working_dir, BASE_OUTPUT_DIR_NAME)
    os.makedirs(abs_main_output_dir, exist_ok=True)

    edited_video_path = os.path.join(abs_working_dir, EDITED_VIDEO_FILENAME)
    if not os.path.exists(edited_video_path):
        print(f"FATAL: Edited video '{EDITED_VIDEO_FILENAME}' not found in '{abs_working_dir}'.")
        return

    print(f"Processing edited video: {EDITED_VIDEO_FILENAME} to determine final output resolution...")
    edited_metadata = get_video_metadata(edited_video_path)
    if not edited_metadata or 'width' not in edited_metadata or 'height' not in edited_metadata:
        print(f"FATAL: Could not get metadata for edited video '{EDITED_VIDEO_FILENAME}'.")
        return
    
    final_output_resolution = {'width': edited_metadata['width'], 'height': edited_metadata['height']}
    print(f"Final output resolution (from edited video): {final_output_resolution['width']}x{final_output_resolution['height']}")

    # --- Estimate Geometric Transform (once) ---
    abs_ref_orig_path = os.path.join(abs_working_dir, REF_ORIGINAL_FRAME_PATH)
    abs_ref_edit_path = os.path.join(abs_working_dir, REF_EDITED_FRAME_PATH)

    if not (os.path.exists(abs_ref_orig_path) and os.path.exists(abs_ref_edit_path)):
        print(f"WARNING: Reference frame(s) for geometric transform not found:")
        if not os.path.exists(abs_ref_orig_path): print(f"  Missing: {abs_ref_orig_path}")
        if not os.path.exists(abs_ref_edit_path): print(f"  Missing: {abs_ref_edit_path}")
        print("  Geometric transforms (crop/scale for originals) will be skipped.")
        GEOMETRIC_TRANSFORM_PARAMS = None
    else:
        GEOMETRIC_TRANSFORM_PARAMS = estimate_geometric_transform_from_refs(abs_ref_orig_path, abs_ref_edit_path)
        if not GEOMETRIC_TRANSFORM_PARAMS:
            print("  Geometric transform estimation failed. Originals will not be specifically cropped/scaled.")

    abs_user_color_lut_path = None
    if USER_COLOR_LUT_PATH:
        abs_user_color_lut_path = os.path.join(abs_working_dir, USER_COLOR_LUT_PATH)
        if not os.path.exists(abs_user_color_lut_path):
            print(f"WARNING: User specified color LUT file not found: {abs_user_color_lut_path}")
            print("  Color LUT application will be skipped.")
            abs_user_color_lut_path = None
        else:
            print(f"  Will use color LUT: {abs_user_color_lut_path}")


    # --- Process all videos ---
    all_video_files_in_dir = []
    for ext in VIDEO_EXTENSIONS:
        all_video_files_in_dir.extend(glob.glob(os.path.join(abs_working_dir, f"*{ext.lower()}")))
        all_video_files_in_dir.extend(glob.glob(os.path.join(abs_working_dir, f"*{ext.upper()}")))
    all_video_files_in_dir = sorted(list(set(p for p in all_video_files_in_dir if os.path.isfile(p))))

    if not all_video_files_in_dir:
        print(f"No video files found in '{abs_working_dir}'.")
        return
    
    print(f"\nFound {len(all_video_files_in_dir)} video(s) to process.")

    for video_full_path in all_video_files_in_dir:
        video_file_name_with_ext = os.path.basename(video_full_path)
        video_name_no_ext, _ = os.path.splitext(video_file_name_with_ext)

        print(f"\n--- Processing: {video_file_name_with_ext} ---")
        is_edited = (video_full_path == edited_video_path)
        
        current_geom_transform = None
        current_color_lut = None
        if not is_edited: # Apply special transforms only to original素材
            current_geom_transform = GEOMETRIC_TRANSFORM_PARAMS
            current_color_lut = abs_user_color_lut_path

        process_video_for_frames(
            video_path=video_full_path,
            video_name_no_ext=video_name_no_ext,
            main_output_folder=abs_main_output_dir,
            is_edited_video_flag=is_edited,
            final_output_resolution=final_output_resolution,
            mask_rect_config=MASK_RECT,
            geom_transform_to_apply=current_geom_transform,
            color_lut_to_apply=current_color_lut
        )

    print("\n\nStep 1 (Frame Extraction and Preparation with Transforms) completed.")
    print(f"All frame outputs should be in subdirectories under: {abs_main_output_dir}")


if __name__ == '__main__':
    print("Starting Step 1: Frame Extraction and Preparation (with Transform Estimation)...")
    print(f"Working directory: {os.path.abspath(WORKING_DIR)}")
    print(f"Edited video file: {EDITED_VIDEO_FILENAME}")
    if MASK_RECT: print(f"Mask rectangle config: {MASK_RECT}")
    else: print("Mask rectangle: Not set.")
    
    print(f"Reference original frame for transforms: {REF_ORIGINAL_FRAME_PATH}")
    print(f"Reference edited frame for transforms  : {REF_EDITED_FRAME_PATH}")
    if USER_COLOR_LUT_PATH: print(f"User color LUT file: {USER_COLOR_LUT_PATH}")
    else: print("User color LUT file: Not set.")
    print("-" * 30)

    try:
        # Ensure OpenCV Contrib is installed if SIFT is desired.
        # `pip install opencv-python opencv-contrib-python numpy`
        main_step1()
    except FileNotFoundError as e: # More specific for ffmpeg/ffprobe
        if 'ffmpeg' in str(e).lower() or 'ffprobe' in str(e).lower():
            print("--------------------------------------------------------------------")
            print("FFMPEG/FFPROBE NOT FOUND. Please ensure they are installed and")
            print("added to your system's PATH environment variable.")
            print("Download from: https://ffmpeg.org/download.html")
            print("--------------------------------------------------------------------")
        else:
            print(f"A FileNotFoundError occurred: {e}") # Other file not found
    except Exception as e:
        print(f"An critical error occurred during Step 1 execution: {e}")
        import traceback
        traceback.print_exc()