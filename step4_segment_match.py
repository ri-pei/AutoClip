import pandas as pd
import json
import numpy as np
import os
import glob

# --- Constants (Adjust as needed) ---
WORKING_DIR = "." 
OUTPUT_DIR_NAME = "output"
COARSE_MATCH_CSV_FILENAME = "coarse_match_results2.csv"
FINAL_SEGMENTS_CSV_FILENAME = "final_video_segments_refined.csv"
PHASH_CSVS_DIR = f"{WORKING_DIR}/{OUTPUT_DIR_NAME}" # Directory where individual pHash CSVs are

OFFSET_JUMP_THRESHOLD_MS = 1000 # For detecting potential breaks
SUSTAINED_LOOKAHEAD_FRAMES = 3 # How many frames to check for a sustained break
# For checking if lookahead frames align with a *newly proposed* segment.
# Can be same or slightly more lenient than OFFSET_JUMP_THRESHOLD_MS
OFFSET_JUMP_THRESHOLD_MS_SUSTAINED = 1200 

# --- Helper Functions ---
def parse_time_from_filename(filename_str):
    if not isinstance(filename_str, str): # Guard against NaN or other types
        return None
    try:
        time_str = filename_str.split('_time_')[-1].replace('.png', '')
        parts = time_str.split('-')
        h, m, s, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return (h * 3600 + m * 60 + s) * 1000 + ms
    except Exception:
        return None

def get_frame_num_from_filename(filename_str):
    if not isinstance(filename_str, str):
        return None
    try:
        return int(filename_str.split('_frame_')[-1].split('_time_')[0])
    except Exception:
        return None

def load_coarse_matches_with_timestamps(filepath):
    df = pd.read_csv(filepath)
    df['top_n_matches'] = df['top_n_matches'].apply(json.loads)
    
    # Ensure edited_timestamp_ms and edited_frame_number are present
    if 'edited_timestamp_ms' not in df.columns:
        df['edited_timestamp_ms'] = df['edited_frame_filename'].apply(parse_time_from_filename)
    if 'edited_frame_number' not in df.columns:
        df['edited_frame_number'] = df['edited_frame_filename'].apply(get_frame_num_from_filename)
    
    df.dropna(subset=['edited_timestamp_ms', 'edited_frame_number'], inplace=True)
    df['edited_frame_number'] = df['edited_frame_number'].astype(int)
    df = df.sort_values(by='edited_frame_number').reset_index(drop=True)
    return df

def get_initial_best_match_from_top_n(top_n_list):
    if not top_n_list: return None
    return min(top_n_list, key=lambda x: x['phash_distance'])

def load_all_original_frames_data(phash_csvs_base_dir, edited_video_name_no_ext):
    all_original_dfs = []
    # Path pattern to find original video pHash CSVs
    # Assumes pHash CSVs are in subdirectories named after the video
    # e.g., output/<original_video_name>/<original_video_name>_phash.csv
    
    # Find all potential pHash CSVs first
    phash_files = []
    for root, dirs, files in os.walk(phash_csvs_base_dir):
        for file in files:
            if file.endswith("_phash.csv") and edited_video_name_no_ext not in file:
                 phash_files.append(os.path.join(root, file))
    
    print(f"Found potential original pHash files: {phash_files}")

    for f_path in phash_files:
        try:
            # Infer original_video_name from the CSV filename or its parent directory
            # This part is sensitive to your directory structure and naming
            original_video_name = os.path.basename(os.path.dirname(f_path)) # Assumes CSV is in a dir named after video
            if original_video_name == edited_video_name_no_ext or not original_video_name: # double check
                 original_video_name = os.path.basename(f_path).replace("_phash.csv", "")
                 if original_video_name == edited_video_name_no_ext: # Still the edited one? Skip.
                     continue


            df_orig = pd.read_csv(f_path)
            # Ensure standard column names; adapt if your pHash CSVs differ
            # We need: 'original_video_name', 'original_frame_number', 'original_timestamp_ms', 'original_phash'
            df_orig.rename(columns={
                'video_name': 'original_video_name_temp', # Avoid immediate overwrite if 'original_video_name' exists
                'frame_number': 'original_frame_number',
                'timestamp_ms': 'original_timestamp_ms',
                'phash': 'original_phash'
            }, inplace=True, errors='ignore') # errors='ignore' if some columns might not exist

            # If 'original_video_name' wasn't in the CSV, assign it
            if 'original_video_name_temp' not in df_orig.columns:
                 df_orig['original_video_name'] = original_video_name
            else:
                 df_orig['original_video_name'] = df_orig['original_video_name_temp']


            # Select necessary columns
            # Check if all expected columns are present after renaming and assignment
            required_cols = ['original_video_name', 'original_frame_number', 'original_timestamp_ms', 'original_phash']
            if not all(col in df_orig.columns for col in required_cols):
                print(f"Warning: Skipping {f_path} due to missing one of required columns: {required_cols}. Available: {df_orig.columns.tolist()}")
                continue

            all_original_dfs.append(df_orig[required_cols])
            print(f"Loaded {len(df_orig)} frames from {original_video_name} ({f_path})")

        except Exception as e:
            print(f"Error loading or processing original pHash CSV {f_path}: {e}")

    if not all_original_dfs:
        raise ValueError("No original video pHash data loaded. Check paths and file contents.")
    
    df_combined = pd.concat(all_original_dfs, ignore_index=True)
    # Ensure timestamps are numeric for sorting and calculations
    df_combined['original_timestamp_ms'] = pd.to_numeric(df_combined['original_timestamp_ms'], errors='coerce')
    df_combined.dropna(subset=['original_timestamp_ms'], inplace=True)
    return df_combined


# --- Main Logic ---
def run_segmentation_and_refinement(edited_video_name_no_ext_param):
    coarse_match_file = os.path.join(WORKING_DIR, OUTPUT_DIR_NAME, COARSE_MATCH_CSV_FILENAME)
    df_processed_frames = load_coarse_matches_with_timestamps(coarse_match_file)

    if df_processed_frames.empty:
        print("No coarse match data loaded. Exiting.")
        return

    try:
        # Pass the edited video name to exclude its pHash CSV
        df_all_original_frames = load_all_original_frames_data(PHASH_CSVS_DIR, edited_video_name_no_ext_param)
    except ValueError as e:
        print(e)
        return

    print(f"Total original frames loaded: {len(df_all_original_frames)}")
    if df_all_original_frames.empty:
        print("No original frames available for matching. Exiting.")
        return

    preliminary_segments = []
    current_segment_chosen_matches = [] # Stores (edited_frame_row, chosen_match_dict)

    # Initialize with the first frame's best match
    if len(df_processed_frames) == 0:
        print("df_processed_frames is empty after loading. Cannot proceed.")
        return

    # --- 3. Improved Preliminary Segmentation ---
    last_confirmed_match_info = None

    for idx, edited_frame_row in df_processed_frames.iterrows():
        current_edited_ts = edited_frame_row['edited_timestamp_ms']
        current_top_n = edited_frame_row['top_n_matches']
        
        current_initial_best_match = get_initial_best_match_from_top_n(current_top_n)

        if current_initial_best_match is None:
            # This frame has no matches, if a segment is open, close it
            if current_segment_chosen_matches:
                preliminary_segments.append(list(current_segment_chosen_matches))
                current_segment_chosen_matches = []
            last_confirmed_match_info = None # Reset
            print(f"Warning: Edited frame {edited_frame_row['edited_frame_number']} has no coarse matches. Skipping.")
            continue

        chosen_match_for_current_frame = current_initial_best_match # Default
        
        if last_confirmed_match_info is None: # First frame of a video or after a hard break
            is_break_decision = True # Effectively, it's the start of a new segment
        else:
            current_initial_offset = current_initial_best_match['original_timestamp_ms'] - current_edited_ts
            expected_offset_from_last = last_confirmed_match_info['original_timestamp_ms'] - last_confirmed_match_info['edited_timestamp_ms']
            
            potential_break = (
                current_initial_best_match['original_video_name'] != last_confirmed_match_info['video_name'] or
                abs(current_initial_offset - expected_offset_from_last) > OFFSET_JUMP_THRESHOLD_MS
            )
            is_break_decision = potential_break # Tentative

            if potential_break:
                # Attempt 1: Rescue from top_N
                rescued = False
                for candidate_match in current_top_n:
                    candidate_offset = candidate_match['original_timestamp_ms'] - current_edited_ts
                    if (candidate_match['original_video_name'] == last_confirmed_match_info['video_name'] and
                        abs(candidate_offset - expected_offset_from_last) <= OFFSET_JUMP_THRESHOLD_MS):
                        chosen_match_for_current_frame = candidate_match
                        is_break_decision = False # Rescued, no break
                        rescued = True
                        break
                
                if not rescued: # Attempt 2: Confirm sustained break (if still a break)
                    is_sustained = True # Assume sustained initially
                    if idx + SUSTAINED_LOOKAHEAD_FRAMES < len(df_processed_frames):
                        for k_lookahead in range(1, SUSTAINED_LOOKAHEAD_FRAMES + 1):
                            lookahead_idx = idx + k_lookahead
                            lookahead_edited_frame_row = df_processed_frames.iloc[lookahead_idx]
                            lookahead_edited_ts = lookahead_edited_frame_row['edited_timestamp_ms']
                            lookahead_top_n = lookahead_edited_frame_row['top_n_matches']
                            lookahead_best_match = get_initial_best_match_from_top_n(lookahead_top_n)

                            if lookahead_best_match is None: # A lookahead frame has no match
                                is_sustained = False; break 
                            
                            lookahead_offset = lookahead_best_match['original_timestamp_ms'] - lookahead_edited_ts
                            # Check if lookahead aligns with the *new* segment proposed by current_initial_best_match
                            if not (lookahead_best_match['original_video_name'] == current_initial_best_match['original_video_name'] and \
                               abs(lookahead_offset - (current_initial_best_match['original_timestamp_ms'] - current_edited_ts)) <= OFFSET_JUMP_THRESHOLD_MS_SUSTAINED):
                                is_sustained = False; break
                    else: # Not enough frames to look ahead
                        is_sustained = False # Cannot confirm, assume not sustained

                    if is_sustained:
                        is_break_decision = True # Confirmed break
                        chosen_match_for_current_frame = current_initial_best_match # This frame starts the new trend
                    else:
                        is_break_decision = False # Not sustained, keep in old segment (using its initial best)
                        chosen_match_for_current_frame = current_initial_best_match


        if is_break_decision and current_segment_chosen_matches: # is_break_decision is True means new segment
            preliminary_segments.append(list(current_segment_chosen_matches))
            current_segment_chosen_matches = []

        current_segment_chosen_matches.append({
            'edited_frame_data': edited_frame_row.to_dict(),
            'chosen_match_info': chosen_match_for_current_frame
        })
        
        last_confirmed_match_info = {
            'video_name': chosen_match_for_current_frame['original_video_name'],
            'original_timestamp_ms': chosen_match_for_current_frame['original_timestamp_ms'],
            'edited_timestamp_ms': current_edited_ts # ts of the current edited frame
        }

    if current_segment_chosen_matches: # Add the last segment
        preliminary_segments.append(list(current_segment_chosen_matches))

    print(f"Identified {len(preliminary_segments)} preliminary segments after improved logic.")

    # --- 4. Segment Offset Calculation ---
    # --- 5. Frame Matching Refinement (Using Full Original Video Data) ---
    all_refined_matches_list = []
    for seg_idx, segment_data_list in enumerate(preliminary_segments):
        if not segment_data_list: continue

        # Determine segment's original video and calculate target offset
        segment_original_video_name = segment_data_list[0]['chosen_match_info']['original_video_name']
        
        offsets_in_segment = []
        for item in segment_data_list:
            offsets_in_segment.append(item['chosen_match_info']['original_timestamp_ms'] - item['edited_frame_data']['edited_timestamp_ms'])
        
        if not offsets_in_segment: 
            print(f"Warning: Segment {seg_idx} has no offsets. Skipping refinement for this segment.")
            # Optionally, add original chosen_match_info to all_refined_matches_list as fallback
            for item in segment_data_list:
                 all_refined_matches_list.append({
                    'edited_frame_number': item['edited_frame_data']['edited_frame_number'],
                    'edited_timestamp_ms': item['edited_frame_data']['edited_timestamp_ms'],
                    'refined_original_video_name': item['chosen_match_info']['original_video_name'],
                    'refined_original_frame_number': item['chosen_match_info']['original_frame_number'],
                    'refined_original_timestamp_ms': item['chosen_match_info']['original_timestamp_ms'],
                    'refined_phash_distance': item['chosen_match_info'].get('phash_distance', -1), # pHash might not be relevant here
                    'comment': 'Used chosen_match from preliminary segmentation (no offset)'
                })
            continue

        target_segment_offset_ms = np.median(offsets_in_segment)

        df_target_original_video_frames = df_all_original_frames[
            df_all_original_frames['original_video_name'] == segment_original_video_name
        ].copy() # Use .copy() to avoid SettingWithCopyWarning if you modify it later

        if df_target_original_video_frames.empty:
            print(f"Warning: No frames found in df_all_original_frames for video '{segment_original_video_name}' (Segment {seg_idx}). Skipping segment.")
            for item in segment_data_list: # Fallback
                 all_refined_matches_list.append({
                    'edited_frame_number': item['edited_frame_data']['edited_frame_number'],
                    'edited_timestamp_ms': item['edited_frame_data']['edited_timestamp_ms'],
                    'refined_original_video_name': item['chosen_match_info']['original_video_name'],
                    'refined_original_frame_number': item['chosen_match_info']['original_frame_number'],
                    'refined_original_timestamp_ms': item['chosen_match_info']['original_timestamp_ms'],
                    'refined_phash_distance': item['chosen_match_info'].get('phash_distance', -1),
                    'comment': 'Used chosen_match (target original video frames not found)'
                })
            continue
            
        # For faster lookup, sort by timestamp if not already
        df_target_original_video_frames.sort_values('original_timestamp_ms', inplace=True)
        # Convert to NumPy array for faster access if performance is an issue on huge videos
        # original_ts_array = df_target_original_video_frames['original_timestamp_ms'].to_numpy()
        
        for item in segment_data_list:
            edited_frame_data = item['edited_frame_data']
            edited_ts = edited_frame_data['edited_timestamp_ms']
            ideal_original_ts = edited_ts + target_segment_offset_ms

            # Find closest frame in df_target_original_video_frames
            # This is the core of the refined matching for this step
            df_target_original_video_frames['time_diff'] = (df_target_original_video_frames['original_timestamp_ms'] - ideal_original_ts).abs()
            
            # Get the row with the minimum time_diff
            # If multiple frames have the exact same minimum time_diff (rare for ms timestamps),
            # this picks the first one. You could add a tie-breaker (e.g., pHash if available and desired).
            if df_target_original_video_frames.empty: # Should be caught above, but defensive
                final_refined_match_series = pd.Series(item['chosen_match_info']) # Fallback
                comment = 'Used chosen_match (target original video frames empty at match time)'
            else:
                final_refined_match_idx = df_target_original_video_frames['time_diff'].idxmin()
                final_refined_match_series = df_target_original_video_frames.loc[final_refined_match_idx]
                comment = 'Refined from full original video data'

            all_refined_matches_list.append({
                'edited_frame_number': edited_frame_data['edited_frame_number'],
                'edited_timestamp_ms': edited_ts,
                'refined_original_video_name': final_refined_match_series['original_video_name'],
                'refined_original_frame_number': final_refined_match_series['original_frame_number'],
                'refined_original_timestamp_ms': final_refined_match_series['original_timestamp_ms'],
                # pHash distance is not directly comparable here as we didn't use pHash for this selection step primarily
                'refined_phash_distance': final_refined_match_series.get('phash_distance', -1), # If phash was in df_all_original_frames
                'comment': comment
            })
    
    if not all_refined_matches_list:
        print("No refined matches were generated. Cannot create final segments.")
        return

    df_refined_matches = pd.DataFrame(all_refined_matches_list)
    df_refined_matches.sort_values(by='edited_frame_number', inplace=True)


    # --- 6. Generate Final Segments for FFmpeg ---
    final_segments_for_ffmpeg = []
    if df_refined_matches.empty:
        print("df_refined_matches is empty. No segments to output.")
    else:
        current_ffmpeg_segment = {}
        for _, row in df_refined_matches.iterrows():
            if pd.isna(row['refined_original_video_name']) or pd.isna(row['refined_original_frame_number']):
                if current_ffmpeg_segment:
                    final_segments_for_ffmpeg.append(current_ffmpeg_segment)
                    current_ffmpeg_segment = {}
                continue

            if not current_ffmpeg_segment:
                current_ffmpeg_segment = {
                    'edited_start_frame': row['edited_frame_number'],
                    'edited_start_time_ms': row['edited_timestamp_ms'],
                    'original_video_name': row['refined_original_video_name'],
                    'original_start_frame': row['refined_original_frame_number'],
                    'original_start_time_ms': row['refined_original_timestamp_ms'],
                    'edited_end_frame': row['edited_frame_number'],
                    'edited_end_time_ms': row['edited_timestamp_ms'],
                    'original_end_frame': row['refined_original_frame_number'],
                    'original_end_time_ms': row['refined_original_timestamp_ms']
                }
            else:
                is_continuation = (
                    row['refined_original_video_name'] == current_ffmpeg_segment['original_video_name'] and
                    int(row['refined_original_frame_number']) == int(current_ffmpeg_segment['original_end_frame']) + 1 and # Ensure int comparison
                    int(row['edited_frame_number']) == int(current_ffmpeg_segment['edited_end_frame']) + 1
                )
                if is_continuation:
                    current_ffmpeg_segment['edited_end_frame'] = row['edited_frame_number']
                    current_ffmpeg_segment['edited_end_time_ms'] = row['edited_timestamp_ms']
                    current_ffmpeg_segment['original_end_frame'] = row['refined_original_frame_number']
                    current_ffmpeg_segment['original_end_time_ms'] = row['refined_original_timestamp_ms']
                else:
                    final_segments_for_ffmpeg.append(current_ffmpeg_segment)
                    current_ffmpeg_segment = {
                        'edited_start_frame': row['edited_frame_number'],
                        'edited_start_time_ms': row['edited_timestamp_ms'],
                        'original_video_name': row['refined_original_video_name'],
                        'original_start_frame': row['refined_original_frame_number'],
                        'original_start_time_ms': row['refined_original_timestamp_ms'],
                        'edited_end_frame': row['edited_frame_number'],
                        'edited_end_time_ms': row['edited_timestamp_ms'],
                        'original_end_frame': row['refined_original_frame_number'],
                        'original_end_time_ms': row['refined_original_timestamp_ms']
                    }
        if current_ffmpeg_segment:
            final_segments_for_ffmpeg.append(current_ffmpeg_segment)

    df_final_segments = pd.DataFrame(final_segments_for_ffmpeg)
    if not df_final_segments.empty:
        output_path = os.path.join(WORKING_DIR, OUTPUT_DIR_NAME, FINAL_SEGMENTS_CSV_FILENAME)
        df_final_segments.to_csv(output_path, index=False)
        print(f"Final refined segments saved to: {output_path}")
    else:
        print("No final segments were generated for FFmpeg.")

# --- Example Usage ---
if __name__ == '__main__':
    # You need to determine the name of your edited video (without extension)
    # This is used by load_all_original_frames_data to avoid loading the edited video's pHash
    # For example, if your edited video is "my_edit.mp4"
    edited_video_name_no_extension = "edited_video" # Replace with your actual edited video name
    
    # Make sure PHASH_CSVS_DIR is set correctly if your pHash CSVs are not directly in "output/"
    # e.g., if they are in "output/video1_phash.csv", "output/video2_phash.csv" etc.
    # The current load_all_original_frames_data assumes they might be in subdirs like "output/video1/video1_phash.csv"
    # or directly in PHASH_CSVS_DIR. Adjust the glob pattern or path construction in that function if needed.

    # Example: Find the edited video name if your coarse_match_results.csv has 'edited_frame_filename'
    # This is a bit of a hack; ideally, you pass this name in.
    try:
        temp_df_coarse = pd.read_csv(os.path.join(WORKING_DIR, OUTPUT_DIR_NAME, COARSE_MATCH_CSV_FILENAME))
        if not temp_df_coarse.empty and 'edited_frame_filename' in temp_df_coarse.columns:
            # Takes the first filename and extracts the video name part
            first_filename = temp_df_coarse['edited_frame_filename'].iloc[0]
            edited_video_name_no_extension = first_filename.split('_frame_')[0]
            print(f"Guessed edited video name (no ext): {edited_video_name_no_extension}")
        else:
            print(f"Could not guess edited video name, using default: {edited_video_name_no_extension}")
    except FileNotFoundError:
        print(f"Coarse match file not found. Using default edited video name: {edited_video_name_no_extension}")
        
    run_segmentation_and_refinement(edited_video_name_no_extension)