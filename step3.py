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

# --- Configuration (MUST ALIGN with code_step1.py and user's actual video files) ---
# Working directory, same as in Step 1
WORKING_DIR = "."  # Or, e.g., "D:/my_video_project"

# Edited video's filename, same as in Step 1
EDITED_VIDEO_FILENAME = "my_edited_clip.mp4"  # Your edited video filename

# Base output directory name, same as in Step 1
OUTPUT_DIR_NAME = "output"

# --- Step 3 Specific Configuration ---
TOP_N_CANDIDATES = 5  # Number of top candidate matches to find for each edited frame
COARSE_MATCH_OUTPUT_SUBDIR_NAME = "coarse_match_visuals2" # For stitched images
COARSE_MATCH_CSV_FILENAME = "coarse_match_results2.csv"
HASH_BITS = 256 # Based on hash_size=16 for pHash (16*16=256 bits)
# --- END Configuration ---

# --- Helper Functions ---

def hex_to_binary_array(hex_hash_str):
    """
    Converts a hexadecimal pHash string to a 1D numpy boolean array.
    Assumes the hash string corresponds to HASH_BITS length.
    """
    try:
        img_hash_obj = imagehash.hex_to_hash(hex_hash_str)
        # .hash is usually a 2D boolean array (e.g., 16x16 for 256 bits)
        binary_array = img_hash_obj.hash.flatten() 
        if len(binary_array) != HASH_BITS:
            # This case should ideally not happen if CSVs are correct and HASH_BITS is set right.
            print(f"Warning: pHash string {hex_hash_str} did not result in {HASH_BITS} bits. "
                  f"Got {len(binary_array)} bits. Check HASH_BITS setting or pHash generation.")
            # Pad or truncate? For now, let it pass, BallTree might handle varying lengths if metric allows,
            # or raise an error later if all inputs to BallTree are not same dimension.
            # Best is to ensure consistent hash length from Step 2.
    except ValueError as e:
        print(f"Error converting hex string '{hex_hash_str}' to binary array: {e}. Returning None.")
        return None
    return binary_array.astype(bool) # Ensure boolean type for BallTree 'hamming' metric

def load_phash_data_from_csv(csv_path):
    """Loads pHash data from a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(csv_path)
        required_cols = ['video_name', 'image_filename', 'frame_number', 'timestamp_ms', 'phash', 'image_path']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: CSV {os.path.basename(csv_path)} is missing one or more required columns. Expected: {', '.join(required_cols)}.")
            return None
        if df.empty:
            return df 
        
        df['phash'] = df['phash'].astype(str)
        df['frame_number'] = df['frame_number'].astype(int)
        df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'], errors='coerce') 
        df.dropna(subset=['timestamp_ms', 'phash'], inplace=True) # Drop rows where essential data became NaN
        df = df[df['phash'].apply(lambda x: len(x) == HASH_BITS // 4)] # Ensure hex string length matches HASH_BITS

        return df
    except FileNotFoundError:
        print(f"Error: pHash CSV file not found at {csv_path}")
        return None
    except pd.errors.EmptyDataError:
        return pd.DataFrame() 
    except Exception as e:
        print(f"Error loading CSV {os.path.basename(csv_path)}: {e}")
        return None

def stitch_images_vertically(img_path1, img_path2, output_path):
    """
    Reads two images, stitches them vertically (img1 on top of img2),
    and saves the result. Paths are assumed to be absolute or resolvable.
    """
    try:
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        if img1 is None:
            print(f"Warning: Could not read image for stitching: {img_path1}")
            return False
        if img2 is None:
            print(f"Warning: Could not read image for stitching: {img_path2}")
            return False

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if w1 != w2:
            print(f"Warning: Image widths differ for stitching ({w1} vs {w2}) for "
                  f"{os.path.basename(img_path1)} and {os.path.basename(img_path2)}. "
                  f"Resizing second image to width {w1}.")
            target_height_img2 = int(h2 * (w1 / w2))
            img2_resized = cv2.resize(img2, (w1, target_height_img2), interpolation=cv2.INTER_AREA)
            stitched_image = cv2.vconcat([img1, img2_resized])
        else:
            stitched_image = cv2.vconcat([img1, img2])
        
        cv2.imwrite(output_path, stitched_image)
        return True
    except Exception as e:
        print(f"Error stitching images {os.path.basename(img_path1)} and {os.path.basename(img_path2)}: {e}")
        return False

# --- Main Step 3 Logic ---
def main_step3():
    # 0. Setup paths
    abs_working_dir = os.path.abspath(WORKING_DIR)
    abs_base_output_dir = os.path.join(abs_working_dir, OUTPUT_DIR_NAME)

    if not os.path.exists(abs_base_output_dir):
        print(f"Error: Base output directory '{abs_base_output_dir}' not found. Please run Step 1 & 2 first.")
        return

    edited_video_name_no_ext, _ = os.path.splitext(EDITED_VIDEO_FILENAME)
    edited_video_phash_csv_path = os.path.join(abs_base_output_dir, edited_video_name_no_ext, f"{edited_video_name_no_ext}_phash.csv")

    coarse_match_visual_output_dir = os.path.join(abs_base_output_dir, COARSE_MATCH_OUTPUT_SUBDIR_NAME)
    os.makedirs(coarse_match_visual_output_dir, exist_ok=True)
    
    coarse_match_results_csv_path = os.path.join(abs_base_output_dir, COARSE_MATCH_CSV_FILENAME)

    # 1. Load pHash data for edited video
    print(f"Loading pHash data for edited video: {edited_video_name_no_ext}...")
    if not os.path.exists(edited_video_phash_csv_path):
        print(f"FATAL: pHash CSV for edited video not found at '{edited_video_phash_csv_path}'. Please run Step 2 for the edited video.")
        return
    df_edited_video = load_phash_data_from_csv(edited_video_phash_csv_path)
    if df_edited_video is None or df_edited_video.empty:
        print(f"FATAL: pHash data for edited video ('{edited_video_name_no_ext}') is empty or could not be loaded. Check CSV: '{edited_video_phash_csv_path}'")
        return
    print(f"Loaded {len(df_edited_video)} frames for edited video '{edited_video_name_no_ext}'.")

    # 2. Load pHash data for all original source videos
    print("\nLoading pHash data for original source videos...")
    original_source_phash_dfs = []
    all_phash_csv_files_glob = os.path.join(abs_base_output_dir, "*", "*_phash.csv")
    
    for csv_file_path in glob.glob(all_phash_csv_files_glob):
        if os.path.normpath(csv_file_path) == os.path.normpath(edited_video_phash_csv_path):
            continue
        
        source_video_name_from_dir = os.path.basename(os.path.dirname(csv_file_path))
        print(f"  Attempting to load pHash data for: {source_video_name_from_dir} from {os.path.basename(csv_file_path)}")
        df_source = load_phash_data_from_csv(csv_file_path)
        
        if df_source is not None and not df_source.empty:
            if not df_source['video_name'].empty and df_source['video_name'].iloc[0] != source_video_name_from_dir:
                print(f"    Warning: Mismatch in video name for {source_video_name_from_dir}. "
                      f"CSV reports '{df_source['video_name'].iloc[0]}'. Using directory name.")
                df_source['video_name'] = source_video_name_from_dir
            original_source_phash_dfs.append(df_source)
            print(f"    Successfully loaded {len(df_source)} frames for '{source_video_name_from_dir}'.")
        elif df_source is not None and df_source.empty:
             print(f"    Note: pHash data for '{source_video_name_from_dir}' is empty or filtered out. CSV: {os.path.basename(csv_file_path)}")
        else:
            print(f"    Warning: Failed to load pHash data for '{source_video_name_from_dir}'. CSV: {os.path.basename(csv_file_path)}")

    if not original_source_phash_dfs:
        print("FATAL: No pHash data found for any original source videos. "
              f"Please ensure Step 2 has run for all source videos and their pHash CSVs are in '{abs_base_output_dir}/<source_video_name>/'.")
        return

    df_all_originals = pd.concat(original_source_phash_dfs, ignore_index=True)
    if df_all_originals.empty:
        print("FATAL: Combined original source pHash data is empty. Cannot build BallTree.")
        return
    
    print(f"\nPreparing {len(df_all_originals)} original source frames for BallTree...")
    
    # Convert hex pHashes of original frames to binary arrays for BallTree
    original_phashes_binary_list = []
    valid_original_indices = [] # Keep track of indices that yield valid binary arrays
    for idx, hex_hash in enumerate(df_all_originals['phash']):
        binary_arr = hex_to_binary_array(hex_hash)
        if binary_arr is not None and len(binary_arr) == HASH_BITS:
            original_phashes_binary_list.append(binary_arr)
            valid_original_indices.append(idx)
        else:
            print(f"  Skipping original frame (index {idx}, hash {hex_hash}) due to pHash conversion error or length mismatch.")

    if not original_phashes_binary_list:
        print("FATAL: No valid binary pHashes could be generated from original source frames. Cannot build BallTree.")
        return

    original_phashes_matrix = np.array(original_phashes_binary_list)
    # Filter df_all_originals to only include frames for which we have valid binary hashes
    df_all_originals_valid = df_all_originals.iloc[valid_original_indices].reset_index(drop=True)


    print(f"Building BallTree with {len(original_phashes_matrix)} original source frames (metric: hamming)...")
    tree = BallTree(original_phashes_matrix.astype(np.int8), metric='hamming') # BallTree expects numerical 0/1 for hamming

    # 3. Perform coarse matching using BallTree
    coarse_match_results_list = []
    
    print(f"\nPerforming coarse matching for {len(df_edited_video)} edited frames (Top {TOP_N_CANDIDATES} candidates using BallTree)...")
    for _, edited_frame_row in tqdm(df_edited_video.iterrows(), total=len(df_edited_video), desc="Matching Edited Frames"):
        edited_phash_hex = str(edited_frame_row['phash'])
        edited_image_path_relative = edited_frame_row['image_path']
        
        edited_phash_binary = hex_to_binary_array(edited_phash_hex)
        if edited_phash_binary is None or len(edited_phash_binary) != HASH_BITS:
            print(f"  Skipping edited frame {edited_frame_row['image_filename']} due to pHash conversion error or length mismatch.")
            # Add an entry with no matches, or skip? Let's skip for now, or add empty matches
            coarse_match_results_list.append({
                'edited_video_name': edited_frame_row['video_name'],
                'edited_frame_filename': edited_frame_row['image_filename'],
                'edited_frame_number': int(edited_frame_row['frame_number']),
                'edited_timestamp_ms': float(edited_frame_row['timestamp_ms']),
                'edited_phash': edited_phash_hex,
                'edited_image_path': edited_image_path_relative,
                'top_n_matches': json.dumps([]) # Empty list of matches
            })
            continue

        # Query the tree
        # .reshape(1,-1) to make it a 2D array for query
        # .astype(np.int8) to match tree input type
        distances_prop, indices = tree.query(edited_phash_binary.astype(np.int8).reshape(1, -1), k=min(TOP_N_CANDIDATES, len(original_phashes_matrix)))
        
        # distances_prop is fractional Hamming distance (proportion of differing bits)
        # Convert to integer Hamming distance (count of differing bits)
        distances_int = (distances_prop[0] * HASH_BITS).round().astype(int)
        candidate_indices_in_valid_df = indices[0]

        top_n_selected_matches = []
        for i, original_idx in enumerate(candidate_indices_in_valid_df):
            original_frame_record = df_all_originals_valid.iloc[original_idx]
            top_n_selected_matches.append({
                'original_video_name': original_frame_record['video_name'],
                'original_frame_filename': original_frame_record['image_filename'],
                'original_frame_number': int(original_frame_record['frame_number']),
                'original_timestamp_ms': float(original_frame_record['timestamp_ms']),
                'original_phash': original_frame_record['phash'],
                'original_image_path': original_frame_record['image_path'],
                'phash_distance': int(distances_int[i]) # ensure it's python int for JSON
            })

        coarse_match_results_list.append({
            'edited_video_name': edited_frame_row['video_name'],
            'edited_frame_filename': edited_frame_row['image_filename'],
            'edited_frame_number': int(edited_frame_row['frame_number']),
            'edited_timestamp_ms': float(edited_frame_row['timestamp_ms']),
            'edited_phash': edited_phash_hex,
            'edited_image_path': edited_image_path_relative,
            'top_n_matches': json.dumps(top_n_selected_matches)
        })

        # 4. Create and save stitched image for the best match
        if top_n_selected_matches:
            best_match = top_n_selected_matches[0]
            original_image_path_relative = best_match['original_image_path']
            
            stitched_image_filename = edited_frame_row['image_filename']
            stitched_image_output_path = os.path.join(coarse_match_visual_output_dir, stitched_image_filename)

            if not os.path.exists(stitched_image_output_path): # Cache check
                full_edited_img_path = os.path.join(abs_working_dir, edited_image_path_relative)
                full_original_img_path = os.path.join(abs_working_dir, original_image_path_relative)
                
                if not os.path.exists(full_edited_img_path):
                    print(f"    Warning: Edited frame image not found at {full_edited_img_path}. Cannot create stitched image.")
                    continue
                if not os.path.exists(full_original_img_path):
                    print(f"    Warning: Original candidate frame image not found at {full_original_img_path}. Cannot create stitched image.")
                    continue
                stitch_images_vertically(full_edited_img_path, full_original_img_path, stitched_image_output_path)
    
    # 5. Save the coarse match results CSV
    if coarse_match_results_list:
        df_coarse_results = pd.DataFrame(coarse_match_results_list)
        df_coarse_results.to_csv(coarse_match_results_csv_path, index=False, lineterminator='\n')
        print(f"\nCoarse match results (CSV) saved to: {coarse_match_results_csv_path}")
    else:
        print("\nNo coarse match results were generated.")
        
    print(f"Stitched comparison images (if any) saved in: {coarse_match_visual_output_dir}")
    print("\n--- Step 3 (Coarse Matching with BallTree) completed. ---")

if __name__ == '__main__':
    print("--- Running Step 3: Coarse Frame Matching (with BallTree) ---")
    print(f"Working Directory: {os.path.abspath(WORKING_DIR)}")
    print(f"Edited Video Filename (for pHash lookup): {EDITED_VIDEO_FILENAME}")
    print(f"Base Output Directory: {os.path.abspath(os.path.join(WORKING_DIR, OUTPUT_DIR_NAME))}")
    print(f"Top N Candidates per Edited Frame: {TOP_N_CANDIDATES}")
    print(f"Hash Bits (pHash length): {HASH_BITS}")
    print(f"Coarse Match Visuals Subdirectory: {COARSE_MATCH_OUTPUT_SUBDIR_NAME}")
    print(f"Coarse Match CSV Filename: {COARSE_MATCH_CSV_FILENAME}")
    print("-" * 30)

    print("Ensure you have 'pandas', 'imagehash', 'opencv-python', 'tqdm', 'numpy', 'scikit-learn' installed.")
    print("e.g., pip install pandas imagehash opencv-python tqdm numpy scikit-learn")
    print("-" * 30)
    
    try:
        main_step3()
    except Exception as e:
        print(f"\nAN UNEXPECTED CRITICAL ERROR occurred during Step 3 execution: {e}")
        import traceback
        traceback.print_exc()