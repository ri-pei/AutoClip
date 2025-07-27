import csv
import os
from fractions import Fraction
import urllib.parse

# --- Configuration ---
CSV_FILE_PATH = 'final_video_segments_refined.csv'
FCPXML_OUTPUT_PATH = 'converted_from_csv.fcpxml'
VIDEO_FRAME_RATE = 23.976  # Frame rate for the main sequence
# IMPORTANT: Update this to your actual working directory where original videos are located
WORKING_DIRECTORY = r'D:\Videos\beyond the time 2' 
PROJECT_NAME = "Beyond the time (CSV Import)" # Name for the FCPXML project
EVENT_NAME = "CSV Import Event" # Name for the FCPXML event

# --- Helper Functions ---
def get_frame_duration_params(fps_value):
    """Calculates numerator and denominator for FCPXML frameDuration from FPS."""
    # FCPXML frameDuration is 1/FPS. limit_denominator helps find a common fraction.
    f = Fraction(fps_value).limit_denominator(max_denominator=100000) # Increased max_denominator for precision
    return f.denominator, f.numerator # Returns (frame_duration_N, frame_duration_D)

def ms_to_fcpxml_time(ms, frame_duration_N, frame_duration_D):
    """
    Converts milliseconds to FCPXML time string (e.g., "1001/24000s").
    ms: time in milliseconds
    frame_duration_N: Numerator of frame duration (e.g., 1001 for 23.976fps)
    frame_duration_D: Denominator of frame duration (e.g., 24000 for 23.976fps)
    """
    if ms < 0: ms = 0 # Duration/time cannot be negative
    
    # Effective FPS = frame_duration_D / frame_duration_N
    # Number of frames = (time_in_seconds) * FPS
    num_frames_float = (ms / 1000.0) * (frame_duration_D / float(frame_duration_N))
    num_frames_int = round(num_frames_float) # Round to nearest whole frame
    
    # FCPXML time = num_integer_frames * frameDuration_N / frameDuration_D
    return f"{int(num_frames_int * frame_duration_N)}/{frame_duration_D}s"

# --- Main Script ---
def main():
    # --- Initialize parameters for the main sequence ---
    seq_fd_N, seq_fd_D = get_frame_duration_params(VIDEO_FRAME_RATE)
    main_sequence_format_id = "r0" # Standard ID for the main sequence format

    # --- Read CSV data ---
    segments = []
    try:
        with open(CSV_FILE_PATH, 'r', newline='', encoding='utf-8-sig') as f: # utf-8-sig handles potential BOM
            reader = csv.DictReader(f)
            for row in reader:
                # Convert time values to float and frames to int
                for key in ['edited_start_time_ms', 'original_start_time_ms', 'edited_end_time_ms', 'original_end_time_ms']:
                    row[key] = float(row[key])
                for key in ['edited_start_frame', 'original_start_frame', 'edited_end_frame', 'original_end_frame']:
                    row[key] = int(row[key])
                segments.append(row)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if not segments:
        print("CSV file is empty or could not be processed.")
        return

    # --- Discover unique assets and their properties ---
    unique_assets_info = {}
    # To store format definitions: key=(fd_N, fd_D), value=format_id
    # This avoids redefining identical formats.
    defined_formats_lookup = {} 
    
    resource_id_counter = 1 # Start resource IDs from r1 (r0 is reserved for sequence format)

    # Define main sequence format string (will be added to fcpxml_parts later)
    seq_format_name = f"FFVideoFormat1080p{VIDEO_FRAME_RATE:.3f}fps".replace(".","_")
    main_sequence_format_definition = \
        f'<format id="{main_sequence_format_id}" name="{seq_format_name}" width="1920" height="1080" frameDuration="{seq_fd_N}/{seq_fd_D}s"/>'


    for segment_row in segments:
        original_video_name_from_csv = segment_row['original_video_name']
        # Normalize: assume "original" in CSV means "original.mp4" to match example
        asset_key_name = "original.mp4" if original_video_name_from_csv == "original" else original_video_name_from_csv

        if asset_key_name not in unique_assets_info:
            current_asset_details = {}
            full_asset_path = os.path.join(WORKING_DIRECTORY, asset_key_name)
            current_asset_details['path'] = full_asset_path

            # --- Determine asset's native frame rate and total duration ---
            # This section requires assumptions or hardcoding if not in CSV
            if asset_key_name == "original.mp4": # Special handling from example
                asset_fps = VIDEO_FRAME_RATE
                current_asset_details['total_duration_str'] = "43165/6s" # From example FCPXML
                # This is for the <conform-rate srcFrameRate="..."> tag
                current_asset_details['conform_src_frame_rate'] =VIDEO_FRAME_RATE
            else:
                # For other assets, assume they match sequence FPS
                # This is a placeholder; ideally, you'd get this from media file metadata
                asset_fps = VIDEO_FRAME_RATE 
                current_asset_details['total_duration_str'] = "360000/1s" # Placeholder: 100 hours
                current_asset_details['conform_src_frame_rate'] = None # No conform needed if FPS matches sequence

            asset_fd_N_current, asset_fd_D_current = get_frame_duration_params(asset_fps)
            current_asset_details['fd_N'] = asset_fd_N_current
            current_asset_details['fd_D'] = asset_fd_D_current

            # Manage format IDs: reuse if an identical format is already defined
            format_params_key = (asset_fd_N_current, asset_fd_D_current)
            if format_params_key == (seq_fd_N, seq_fd_D): # Asset uses main sequence format
                current_asset_details['format_id'] = main_sequence_format_id
            elif format_params_key in defined_formats_lookup:
                current_asset_details['format_id'] = defined_formats_lookup[format_params_key]
            else: # New, unique format needed for this asset
                new_format_id = f"r{resource_id_counter}"
                resource_id_counter += 1
                defined_formats_lookup[format_params_key] = new_format_id
                current_asset_details['format_id'] = new_format_id
            
            current_asset_details['asset_id'] = f"r{resource_id_counter}"
            resource_id_counter += 1
            
            unique_assets_info[asset_key_name] = current_asset_details

    # --- Build FCPXML string parts ---
    fcpxml_parts = []
    fcpxml_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    fcpxml_parts.append('<!DOCTYPE fcpxml>') # As per example
    fcpxml_parts.append('<fcpxml version="1.8">')
    fcpxml_parts.append('    <resources>')
    fcpxml_parts.append(f'        {main_sequence_format_definition}')

    # Add unique asset format definitions to resources
    for (fmt_N, fmt_D), fmt_id in defined_formats_lookup.items():
        # Format name based on FPS for readability
        asset_fps_for_name = Fraction(fmt_D, fmt_N) 
        asset_format_name_str = f"FFVideoFormat1080p{float(asset_fps_for_name):.3f}fps".replace(".","_")
        fcpxml_parts.append(f'        <format id="{fmt_id}" name="{asset_format_name_str}" width="1920" height="1080" frameDuration="{fmt_N}/{fmt_D}s"/>')

    # Add asset definitions to resources
    for name, info in unique_assets_info.items():
        abs_path = os.path.abspath(info['path'])
        path_for_url = abs_path.replace(os.sep, '/') # Use forward slashes
        
        if os.name == 'nt' and path_for_url[1:2] == ':': # Check for drive letter e.g., "C:/"
             drive_part = path_for_url[:2] # e.g., "C:"
             rest_of_path = path_for_url[2:] # e.g., "/Videos/space needed/file.mp4"
             # Quote only the part after "C:", keeping "/" safe for directory structure
             quoted_rest_of_path = urllib.parse.quote(rest_of_path, safe='/')
             src_url = f'file://localhost/{drive_part}{quoted_rest_of_path}'
        else: # Unix-like paths (e.g., /foo/bar) or other non-drive-letter paths
            src_url = 'file://' + urllib.parse.quote(path_for_url, safe='/')
        
        fcpxml_parts.append(f'        <asset id="{info["asset_id"]}" name="{name}" src="{src_url}" duration="{info["total_duration_str"]}" hasVideo="1" format="{info["format_id"]}" audioChannels="2" hasAudio="1"/>')
    
    fcpxml_parts.append('    </resources>')
    fcpxml_parts.append('    <library>')
    fcpxml_parts.append(f'        <event name="{EVENT_NAME}">')
    fcpxml_parts.append(f'            <project name="{PROJECT_NAME}">')
    
    # Sequence total duration from the last segment's edited_end_time_ms
    total_sequence_duration_ms = segments[-1]['edited_end_time_ms'] if segments else 0
    total_sequence_duration_str = ms_to_fcpxml_time(total_sequence_duration_ms, seq_fd_N, seq_fd_D)
    
    fcpxml_parts.append(f'                <sequence tcStart="0/1s" format="{main_sequence_format_id}" tcFormat="NDF" duration="{total_sequence_duration_str}">')
    fcpxml_parts.append('                    <spine>')

    for segment_row in segments:
        original_video_name_from_csv = segment_row['original_video_name']
        asset_key_name = "original.mp4" if original_video_name_from_csv == "original" else original_video_name_from_csv
            
        asset_detail = unique_assets_info[asset_key_name]
        
        # Clip offset and duration are in sequence timebase
        offset_str = ms_to_fcpxml_time(segment_row['edited_start_time_ms'], seq_fd_N, seq_fd_D)
        clip_duration_ms = segment_row['edited_end_time_ms'] - segment_row['edited_start_time_ms']
        duration_str = ms_to_fcpxml_time(clip_duration_ms, seq_fd_N, seq_fd_D)
        
        # Clip 'start' (in-point in source) is in the asset's own timebase
        start_str = ms_to_fcpxml_time(segment_row['original_start_time_ms'], asset_detail['fd_N'], asset_detail['fd_D'])
        
        clip_name_in_spine = os.path.basename(asset_detail['path']) # Use filename for the clip name in spine

        clip_parts = [
            f'<asset-clip name="{clip_name_in_spine}" ref="{asset_detail["asset_id"]}" format="{asset_detail["format_id"]}" offset="{offset_str}" duration="{duration_str}" start="{start_str}" tcFormat="NDF" audioRole="dialogue" enabled="1">',
            '    <adjust-transform scale="1 1" position="0 0" anchor="0 0"/>' # Default transform
        ]
        
        # Add conform-rate if asset FPS differs from sequence FPS
        asset_fps_val = asset_detail['fd_D'] / float(asset_detail['fd_N'])
        seq_fps_val = seq_fd_D / float(seq_fd_N)

        if abs(asset_fps_val - seq_fps_val) > 1e-4 : # If FPS are different (with tolerance)
            src_conform_rate = asset_detail['conform_src_frame_rate']
            if src_conform_rate is None: # Fallback if not specifically set (e.g. for non-"original.mp4" assets)
                 src_conform_rate = int(round(asset_fps_val))
            clip_parts.insert(1, f'    <conform-rate srcFrameRate="{src_conform_rate}"/>')
        
        clip_parts.append('</asset-clip>')
        
        # Indent and add to main FCPXML parts
        fcpxml_parts.extend(['                        ' + part for part in clip_parts])


    fcpxml_parts.append('                    </spine>')
    fcpxml_parts.append('                </sequence>')
    fcpxml_parts.append('            </project>')
    fcpxml_parts.append('        </event>')
    fcpxml_parts.append('    </library>')
    fcpxml_parts.append('</fcpxml>')

    # --- Write to FCPXML file ---
    try:
        with open(FCPXML_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fcpxml_parts))
        print(f"FCPXML file successfully created at: {os.path.abspath(FCPXML_OUTPUT_PATH)}")
    except Exception as e:
        print(f"Error writing FCPXML file: {e}")

if __name__ == '__main__':
    main()