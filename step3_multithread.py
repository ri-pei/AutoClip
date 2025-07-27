import os
import glob
import pandas as pd
import imagehash # For imagehash.hex_to_hash
import cv2
from tqdm import tqdm
import json # For storing list of dicts in CSV cell
import shutil # For cleaning up output directory if needed
import numpy as np
from sklearn.neighbors import BallTree
import concurrent.futures

# --- Configuration (MUST ALIGN with code_step1.py and user's actual video files) ---
# Working directory, same as in Step 1
WORKING_DIR = "."  # Or, e.g., "D:/my_video_project"

# Edited video's filename, same as in Step 1
EDITED_VIDEO_FILENAME = "my_edited_clip.mp4"  # Your edited video filename

# Base output directory name, same as in Step 1
OUTPUT_DIR_NAME = "output"

# --- Step 3 Specific Configuration ---
TOP_N_CANDIDATES = 5  # Number of top candidate matches to find for each edited frame
COARSE_MATCH_OUTPUT_SUBDIR_NAME = "coarse_match_visuals3" # For stitched images
COARSE_MATCH_CSV_FILENAME = "coarse_match_results3.csv"
HASH_BITS = 256 # Based on hash_size=16 for pHash (16*16=256 bits)
MAX_WORKERS_LOADING = min(8, os.cpu_count() + 4 if os.cpu_count() else 8)  # For loading CSVs
MAX_WORKERS_PROCESSING = min(8, os.cpu_count() or 1) # For processing edited frames
# --- END Configuration ---

# --- Helper Functions (mostly unchanged) ---

def hex_to_binary_array(hex_hash_str):
    try:
        img_hash_obj = imagehash.hex_to_hash(hex_hash_str)
        binary_array = img_hash_obj.hash.flatten()
        if len(binary_array) != HASH_BITS:
            # print(f"Warning: pHash string {hex_hash_str} did not result in {HASH_BITS} bits. Got {len(binary_array)} bits.")
            return None # Strict check
    except ValueError:
        return None
    return binary_array.astype(bool)

def load_single_phash_csv(args):
    """Loads a single pHash CSV file. Designed to be run in a thread pool."""
    csv_file_path, edited_video_phash_csv_path_norm = args
    
    if os.path.normpath(csv_file_path) == edited_video_phash_csv_path_norm:
        return None # Skip edited video's CSV

    source_video_name_from_dir = os.path.basename(os.path.dirname(csv_file_path))
    # print(f"  Thread loading pHash data for: {source_video_name_from_dir} from {os.path.basename(csv_file_path)}")
    
    try:
        df = pd.read_csv(csv_file_path)
        required_cols = ['video_name', 'image_filename', 'frame_number', 'timestamp_ms', 'phash', 'image_path']
        if not all(col in df.columns for col in required_cols):
            # print(f"Warning: CSV {os.path.basename(csv_file_path)} missing cols. Skipping.")
            return None
        if df.empty:
            return None # Skip empty DFs early
        
        df['phash'] = df['phash'].astype(str)
        df['frame_number'] = df['frame_number'].astype(int)
        df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'], errors='coerce')
        df.dropna(subset=['timestamp_ms', 'phash'], inplace=True)
        df = df[df['phash'].apply(lambda x: isinstance(x, str) and len(x) == HASH_BITS // 4)]

        if df.empty: # Check again after filtering
            return None

        if not df['video_name'].empty and df['video_name'].iloc[0] != source_video_name_from_dir:
            # print(f"    Warning: Mismatch in video name for {source_video_name_from_dir}. Forcing dir name.")
            df['video_name'] = source_video_name_from_dir
        
        # print(f"    Thread loaded {len(df)} frames for '{source_video_name_from_dir}'.")
        return df
    except FileNotFoundError:
        # print(f"Error: pHash CSV file not found at {csv_file_path}")
        return None
    except pd.errors.EmptyDataError:
        return None
    except Exception as e:
        # print(f"Error loading CSV {os.path.basename(csv_file_path)} in thread: {e}")
        return None

def stitch_images_vertically(img_path1, img_path2, output_path):
    try:
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        if img1 is None or img2 is None:
            # print(f"Warning: Could not read one or both images for stitching: {img_path1}, {img_path2}")
            return False

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if w1 != w2:
            # print(f"Warning: Image widths differ. Resizing second image.")
            target_height_img2 = int(h2 * (w1 / w2))
            img2_resized = cv2.resize(img2, (w1, target_height_img2), interpolation=cv2.INTER_AREA)
            stitched_image = cv2.vconcat([img1, img2_resized])
        else:
            stitched_image = cv2.vconcat([img1, img2])
        
        cv2.imwrite(output_path, stitched_image)
        return True
    except Exception:
        # print(f"Error stitching images {os.path.basename(img_path1)} and {os.path.basename(img_path2)}: {e}")
        return False
def process_single_edited_frame(args):
    """
    Processes a single frame from the edited video: queries BallTree, finds top N,
    and prepares data for main list and image stitching.
    Designed to be run in a thread pool.
    """
    # Unpack the arguments
    # args = (edited_frame_data_as_tuple, tree, df_originals, work_dir, visual_out_dir)
    edited_frame_row_as_tuple, tree_obj, df_originals_valid_obj, work_dir_str, visual_out_dir_str = args
    
    # Reconstruct Series from tuple for easier access
    # df_edited_video_columns_for_tuple is a global list of column names
    edited_frame_row = pd.Series(edited_frame_row_as_tuple, index=df_edited_video_columns_for_tuple)

    edited_phash_hex = str(edited_frame_row['phash'])
    edited_image_path_relative = edited_frame_row['image_path']
    
    # Ensure 'frame_number' can be converted to int BEFORE using it in the dict.
    try:
        edited_frame_num_int = int(edited_frame_row['frame_number'])
    except ValueError as e:
        print(f"Critical Error: Could not convert frame_number '{edited_frame_row['frame_number']}' to int "
              f"for edited frame file '{edited_frame_row['image_filename']}'. Original CSV data issue? Exception: {e}")
        # Fallback or reraise
        # For now, let's create a result dict indicating failure for this frame
        return {
            'edited_video_name': edited_frame_row.get('video_name', 'ERROR_VNAME'),
            'edited_frame_filename': edited_frame_row.get('image_filename', 'ERROR_FNAME'),
            'edited_frame_number': -1, # Indicate error
            'edited_timestamp_ms': float(edited_frame_row.get('timestamp_ms', 0.0)),
            'edited_phash': edited_phash_hex if 'phash' in edited_frame_row else 'ERROR_PHASH',
            'edited_image_path': edited_image_path_relative if 'image_path' in edited_frame_row else 'ERROR_PATH',
            'top_n_matches': json.dumps([{'error': 'Frame number conversion failed'}])
        }, None


    result_dict = {
        'edited_video_name': edited_frame_row['video_name'],
        'edited_frame_filename': edited_frame_row['image_filename'],
        'edited_frame_number': edited_frame_num_int, # Use the validated int
        'edited_timestamp_ms': float(edited_frame_row['timestamp_ms']),
        'edited_phash': edited_phash_hex,
        'edited_image_path': edited_image_path_relative,
        'top_n_matches': json.dumps([]) # Default to empty
    }

    edited_phash_binary = hex_to_binary_array(edited_phash_hex)
    if edited_phash_binary is None or len(edited_phash_binary) != HASH_BITS:
        # print(f"  Skipping edited frame {edited_frame_row['image_filename']} (thread) due to pHash conversion error.")
        return result_dict, None # Return dict with empty matches and None for stitch task

    # Use the unpacked argument names for clarity
    distances_prop, indices = tree_obj.query(
        edited_phash_binary.astype(np.int8).reshape(1, -1), 
        k=min(TOP_N_CANDIDATES, len(df_originals_valid_obj))
    )
    distances_int = (distances_prop[0] * HASH_BITS).round().astype(int)
    candidate_indices_in_valid_df = indices[0]

    top_n_selected_matches = []
    for i, original_idx in enumerate(candidate_indices_in_valid_df):
        original_frame_record = df_originals_valid_obj.iloc[original_idx]
        top_n_selected_matches.append({
            'original_video_name': original_frame_record['video_name'],
            'original_frame_filename': original_frame_record['image_filename'],
            'original_frame_number': int(original_frame_record['frame_number']),
            'original_timestamp_ms': float(original_frame_record['timestamp_ms']),
            'original_phash': original_frame_record['phash'],
            'original_image_path': original_frame_record['image_path'],
            'phash_distance': int(distances_int[i])
        })
    
    result_dict['top_n_matches'] = json.dumps(top_n_selected_matches)

    stitch_task_args = None
    if top_n_selected_matches:
        best_match = top_n_selected_matches[0]
        original_image_path_relative = best_match['original_image_path']
        
        stitched_image_filename = edited_frame_row['image_filename']
        # Use the unpacked argument name visual_out_dir_str
        stitched_image_output_path = os.path.join(visual_out_dir_str, stitched_image_filename)

        if not os.path.exists(stitched_image_output_path): # Cache check
            # Use the unpacked argument name work_dir_str
            full_edited_img_path = os.path.join(work_dir_str, edited_image_path_relative)
            full_original_img_path = os.path.join(work_dir_str, original_image_path_relative)
            
            if os.path.exists(full_edited_img_path) and os.path.exists(full_original_img_path):
                stitch_task_args = (full_edited_img_path, full_original_img_path, stitched_image_output_path)
    
    return result_dict, stitch_task_args
    
    
# --- Global variables for process_single_edited_frame (to avoid pickling large objects) ---
# These will be set in main_step3 before starting the thread pool for processing.
tree_global = None
df_all_originals_valid_global = None
abs_working_dir_global = None
coarse_match_visual_output_dir_global = None
df_edited_video_columns_for_tuple = None


# --- Main Step 3 Logic ---
def main_step3():
    global tree_global, df_all_originals_valid_global, abs_working_dir_global, \
           coarse_match_visual_output_dir_global, df_edited_video_columns_for_tuple

    # 0. Setup paths
    abs_working_dir_global = os.path.abspath(WORKING_DIR) # Set global
    abs_base_output_dir = os.path.join(abs_working_dir_global, OUTPUT_DIR_NAME)

    if not os.path.exists(abs_base_output_dir):
        print(f"Error: Base output directory '{abs_base_output_dir}' not found.")
        return

    edited_video_name_no_ext, _ = os.path.splitext(EDITED_VIDEO_FILENAME)
    edited_video_phash_csv_path = os.path.join(abs_base_output_dir, edited_video_name_no_ext, f"{edited_video_name_no_ext}_phash.csv")

    coarse_match_visual_output_dir_global = os.path.join(abs_base_output_dir, COARSE_MATCH_OUTPUT_SUBDIR_NAME) # Set global
    os.makedirs(coarse_match_visual_output_dir_global, exist_ok=True)
    
    coarse_match_results_csv_path = os.path.join(abs_base_output_dir, COARSE_MATCH_CSV_FILENAME)

    # 1. Load pHash data for edited video (main thread)
    print(f"Loading pHash data for edited video: {edited_video_name_no_ext}...")
    df_edited_video = load_single_phash_csv((edited_video_phash_csv_path, "dummy_to_force_load_this_one")) # Reuse loader, dummy path to avoid skip
    if df_edited_video is None or df_edited_video.empty:
        print(f"FATAL: pHash data for edited video ('{edited_video_name_no_ext}') is empty or could not be loaded. Check CSV: '{edited_video_phash_csv_path}'")
        return
    print(f"Loaded {len(df_edited_video)} frames for edited video '{edited_video_name_no_ext}'.")
    df_edited_video_columns_for_tuple = df_edited_video.columns.tolist() # Set global

    # 2. Load pHash data for all original source videos (multithreaded)
    print(f"\nLoading pHash data for original source videos (max_workers: {MAX_WORKERS_LOADING})...")
    original_source_phash_dfs = []
    all_phash_csv_files_glob = os.path.join(abs_base_output_dir, "*", "*_phash.csv")
    
    # Prepare arguments for thread pool
    norm_edited_path = os.path.normpath(edited_video_phash_csv_path)
    load_args_list = [(csv_file_path, norm_edited_path) for csv_file_path in glob.glob(all_phash_csv_files_glob)]

    # ...
    # Global variables are already set at this point.
    # The `edited_frames_args` list contains tuples, where each tuple is:
    # (tuple_of_row_values, tree_global, df_all_originals_valid_global, abs_working_dir_global, coarse_match_visual_output_dir_global)

    # Correctly construct the list of arguments for the ThreadPoolExecutor
    # Each element in process_args_list will be a single tuple, which is the `args` for process_single_edited_frame
    process_args_list = [
        (
            row_tuple, # This is the tuple of values for the edited frame row
            tree_global, 
            df_all_originals_valid_global,
            abs_working_dir_global,
            coarse_match_visual_output_dir_global
        ) for row_tuple in df_edited_video.itertuples(index=False, name=None) # row_tuple is already tuple(row)
    ]


    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_PROCESSING) as executor:
        # Pass each tuple from process_args_list as the single argument to process_single_edited_frame
        futures = [executor.submit(process_single_edited_frame, arg_tuple) for arg_tuple in process_args_list]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df_edited_video), desc="Matching Edited Frames"):
            result_dict, stitch_task_args = future.result()
            coarse_match_results_list.append(result_dict)
            if stitch_task_args:
                stitch_tasks_args_list.append(stitch_task_args)
    
    if not original_source_phash_dfs:
        print("FATAL: No pHash data successfully loaded for any original source videos.")
        return

    df_all_originals = pd.concat(original_source_phash_dfs, ignore_index=True)
    if df_all_originals.empty:
        print("FATAL: Combined original source pHash data is empty. Cannot build BallTree.")
        return
    print(f"Total original source frames loaded: {len(df_all_originals)}")
    
    print(f"\nPreparing {len(df_all_originals)} original source frames for BallTree...")
    original_phashes_binary_list = []
    valid_original_indices = []
    for idx, hex_hash in enumerate(df_all_originals['phash']): # This loop is relatively fast
        binary_arr = hex_to_binary_array(hex_hash)
        if binary_arr is not None and len(binary_arr) == HASH_BITS:
            original_phashes_binary_list.append(binary_arr)
            valid_original_indices.append(idx)

    if not original_phashes_binary_list:
        print("FATAL: No valid binary pHashes could be generated from original source frames. Cannot build BallTree.")
        return

    original_phashes_matrix = np.array(original_phashes_binary_list)
    df_all_originals_valid_global = df_all_originals.iloc[valid_original_indices].reset_index(drop=True) # Set global

    print(f"Building BallTree with {len(original_phashes_matrix)} original source frames (metric: hamming)...")
    tree_global = BallTree(original_phashes_matrix.astype(np.int8), metric='hamming') # Set global

    # 3. Perform coarse matching (multithreaded per edited frame)
    coarse_match_results_list = []
    stitch_tasks_args_list = []
    
    print(f"\nPerforming coarse matching (max_workers: {MAX_WORKERS_PROCESSING})...")
    
    # Prepare arguments for process_single_edited_frame
    # Pass data as tuples to make them hashable if executor needs it, and reduce pickling overhead
    # For Series/DataFrame, passing as tuple of (index, data_tuple) or just data_tuple if index not needed
    edited_frames_args = [
        (
            tuple(row), # Convert row to tuple
            tree_global, 
            df_all_originals_valid_global, # This is a DataFrame, potentially large. Shared by reference.
            abs_working_dir_global,
            coarse_match_visual_output_dir_global
        ) for row in df_edited_video.itertuples(index=False, name=None) # Pass rows as simple tuples
    ]


    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_PROCESSING) as executor:
        futures = [executor.submit(process_single_edited_frame, (args_tuple_row, *args_tuple_others)) 
                   for args_tuple_row, *args_tuple_others in edited_frames_args]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(df_edited_video), desc="Matching Edited Frames"):
            result_dict, stitch_task_args = future.result()
            coarse_match_results_list.append(result_dict)
            if stitch_task_args:
                stitch_tasks_args_list.append(stitch_task_args)
    
    # 4. Perform image stitching (can also be in a separate thread pool if many images)
    # For simplicity, doing it sequentially here after collecting all tasks.
    # Or, could integrate into the above loop if image stitching per frame is desired immediately.
    # Let's use another pool for stitching, as it's I/O bound.
    print(f"\nStitching {len(stitch_tasks_args_list)} images (max_workers: {MAX_WORKERS_PROCESSING})...")
    if stitch_tasks_args_list:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_PROCESSING) as executor:
            stitch_futures = [executor.submit(stitch_images_vertically, *args) for args in stitch_tasks_args_list]
            for _ in tqdm(concurrent.futures.as_completed(stitch_futures), total=len(stitch_futures), desc="Stitching Images"):
                pass # Just wait for completion

    # 5. Save the coarse match results CSV
    if coarse_match_results_list:
        # Sort results by edited_frame_number to maintain original order
        coarse_match_results_list.sort(key=lambda x: x['edited_frame_number'])
        df_coarse_results = pd.DataFrame(coarse_match_results_list)
        df_coarse_results.to_csv(coarse_match_results_csv_path, index=False, lineterminator='\n')
        print(f"\nCoarse match results (CSV) saved to: {coarse_match_results_csv_path}")
    else:
        print("\nNo coarse match results were generated.")
        
    print(f"Stitched comparison images (if any) saved in: {coarse_match_visual_output_dir_global}")
    print("\n--- Step 3 (Coarse Matching with BallTree & Multithreading) completed. ---")

if __name__ == '__main__':
    print("--- Running Step 3: Coarse Frame Matching (BallTree & Multithreading) ---")
    print(f"Working Directory: {os.path.abspath(WORKING_DIR)}")
    print(f"Edited Video Filename: {EDITED_VIDEO_FILENAME}")
    print(f"Hash Bits: {HASH_BITS}")
    print(f"Max Workers (Loading): {MAX_WORKERS_LOADING}")
    print(f"Max Workers (Processing): {MAX_WORKERS_PROCESSING}")
    print("-" * 30)
    print("Ensure: 'pandas', 'imagehash', 'opencv-python', 'tqdm', 'numpy', 'scikit-learn' installed.")
    print("-" * 30)
    
    try:
        main_step3()
    except Exception as e:
        print(f"\nAN UNEXPECTED CRITICAL ERROR occurred during Step 3 execution: {e}")
        import traceback
        traceback.print_exc()