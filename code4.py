import csv
import pathlib
import math # For math.gcd

# --- Configuration ---
CSV_FILE_PATH = 'final_video_segments_refined.csv'
OUTPUT_FCPXML_PATH = 'final_cut_pro_export.fcpxml'

# IMPORTANT: Replace with the actual ABSOLUTE path to your video files directory
# Example for Windows: r"D:\MyProjects\VideoEdit\SourceMedia"
# Example for macOS/Linux: "/Users/yourname/Movies/SourceMedia"
WORKING_DIRECTORY = r"D:\Videos\beyond the time 2" 

VIDEO_FRAMERATE = 24.976 # Frames per second for the project and interpretation of media

# --- Helper Functions ---

def get_frame_duration_str(framerate):
    """
    Calculates the FCPXML frameDuration string (e.g., "1001/24000s") from a framerate.
    The frame duration is the reciprocal of the framerate (D/N s if framerate is N/D fps).
    """
    # Handle common known exact fractions first for precision
    if framerate == 23.976: # Common for 24000/1001
        return "1001/24000s"
    if framerate == 29.97: # Common for 30000/1001
        return "1001/30000s"
    if framerate == 59.94: # Common for 60000/1001
        return "1001/60000s"

    # For other framerates, calculate N/D representation.
    # e.g., framerate = 25.0 -> N=25, D=1. Duration = 1/25s.
    # e.g., framerate = 24.976. This is 24976/1000 fps. Duration = 1000/24976 s.
    # We need to simplify this fraction.
    
    # Represent framerate as a fraction N/D. Max denominator for precision.
    # Let's assume up to 3 decimal places for framerate input for this method
    numerator_fr = int(framerate * 1000) # e.g., 24.976 -> 24976
    denominator_fr = 1000                 # So framerate is 24976/1000
    
    # Frame duration is D_fr / N_fr
    num_duration = denominator_fr
    den_duration = numerator_fr
    
    common_divisor = math.gcd(num_duration, den_duration)
    num_simplified = num_duration // common_divisor
    den_simplified = den_duration // common_divisor
    return f"{num_simplified}/{den_simplified}s"

MAIN_FORMAT_ID = "r0" # Main format ID for sequence and clips
FRAME_DURATION_STR = get_frame_duration_str(VIDEO_FRAMERATE)

def ms_to_fcpxml_time_str(ms_value_str):
    """
    Converts a millisecond string value to FCPXML time string format (e.g., "12345/1000s").
    Rounds to the nearest millisecond if the input implies sub-millisecond precision not used.
    """
    try:
        ms_value = float(ms_value_str)
    except ValueError:
        print(f"Warning: Could not convert '{ms_value_str}' to float. Defaulting to 0ms.")
        ms_value = 0.0

    if ms_value == 0.0:
        return "0/1s" # FCPXML common representation for zero time

    # Convert ms to an integer numerator and 1000 as denominator
    # Round to nearest integer millisecond to avoid issues with float precision if any
    numerator = int(round(ms_value)) 
    denominator = 1000
    
    common = math.gcd(numerator, denominator)
    num_simplified = numerator // common
    den_simplified = denominator // common
    return f"{num_simplified}/{den_simplified}s"


# --- Main Script ---
def main():
    all_segments_data = []
    try:
        with open(CSV_FILE_PATH, 'r', newline='', encoding='utf-8-sig') as f: # utf-8-sig handles potential BOM
            reader = csv.DictReader(f)
            for row in reader:
                all_segments_data.append(row)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if not all_segments_data:
        print("CSV file is empty or format is incorrect.")
        return

    # 1. Collect asset information and overall sequence duration
    assets_info = {}  # Stores unique assets: asset_filename -> {'id': rX, 'max_original_end_ms': YYY}
    asset_id_counter = 1 # Start asset-specific r_id from 1 (r0 is for the main format)
    max_edited_end_time_ms_val = 0.0

    for segment in all_segments_data:
        try:
            original_video_name_from_csv = segment['original_video_name']
            
            # Construct the actual asset filename (e.g., add .mp4 if missing)
            path_obj = pathlib.Path(original_video_name_from_csv)
            if not path_obj.suffix: # If "original" becomes "original.mp4"
                actual_asset_filename = f"{original_video_name_from_csv}.mp4"
            else:
                actual_asset_filename = original_video_name_from_csv
            
            # Store the processed filename back for consistent use
            segment['_actual_asset_filename'] = actual_asset_filename

            if actual_asset_filename not in assets_info:
                assets_info[actual_asset_filename] = {
                    'id': f'r{asset_id_counter}',
                    'max_original_end_ms': 0.0
                }
                asset_id_counter += 1
            
            current_original_end_ms = float(segment['original_end_time_ms'])
            if current_original_end_ms > assets_info[actual_asset_filename]['max_original_end_ms']:
                assets_info[actual_asset_filename]['max_original_end_ms'] = current_original_end_ms

            current_edited_end_ms = float(segment['edited_end_time_ms'])
            if current_edited_end_ms > max_edited_end_time_ms_val:
                max_edited_end_time_ms_val = current_edited_end_ms
        except KeyError as e:
            print(f"Error: Missing expected column in CSV: {e}. Problematic row: {segment}")
            return
        except ValueError as e:
            print(f"Error: Could not parse time/frame value in CSV: {e}. Problematic row: {segment}")
            return

    sequence_duration_fcpxml_str = ms_to_fcpxml_time_str(str(max_edited_end_time_ms_val))

    # 2. Build FCPXML content as a list of strings
    fcpxml_content = []
    fcpxml_content.append('<?xml version="1.0" encoding="UTF-8"?>')
    fcpxml_content.append('<!DOCTYPE fcpxml>') # No version for DOCTYPE in example
    fcpxml_content.append('<fcpxml version="1.8">') # FCPXML root version
    
    # --- Resources ---
    fcpxml_content.append('    <resources>')
    
    # Main format definition (used by sequence and assets)
    # Assuming 1920x1080, can be parameterized. Added colorSpace from example.
    fcpxml_content.append(f'        <format id="{MAIN_FORMAT_ID}" name="FFVideoFormat_{int(VIDEO_FRAMERATE*1000)}p" frameDuration="{FRAME_DURATION_STR}" width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>')

    # Asset definitions
    working_dir_path = pathlib.Path(WORKING_DIRECTORY)
    for asset_filename, info in assets_info.items():
        asset_full_path = working_dir_path / asset_filename
        asset_uri = asset_full_path.as_uri() 
        
        # Adjust URI for Windows to match file://localhost/D:/... style if applicable
        if asset_uri.startswith('file:///') and asset_full_path.drive: # Checks for Windows drive letter
             asset_uri = "file://localhost/" + asset_uri[len('file:///'):]

        asset_duration_fcpxml_str = ms_to_fcpxml_time_str(str(info['max_original_end_ms']))
        
        # Basic asset structure. Assuming video and stereo audio. Added audioRate from example.
        fcpxml_content.append(f'        <asset id="{info["id"]}" name="{asset_filename}" src="{asset_uri}" start="0/1s" duration="{asset_duration_fcpxml_str}" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" format="{MAIN_FORMAT_ID}" audioRate="48000"/>')
    
    fcpxml_content.append('    </resources>')

    # --- Library, Event, Project, Sequence ---
    fcpxml_content.append('    <library location="">') # Empty location for simplicity, as in some exports
    fcpxml_content.append('        <event name="CSV Imported Event">')
    fcpxml_content.append('            <project name="CSV Imported Project">')
    
    # Sequence settings. tcStart="0/1s" is standard. tcFormat="NDF" (Non-Drop Frame).
    # Added audioLayout and audioRate from example.
    fcpxml_content.append(f'                <sequence format="{MAIN_FORMAT_ID}" duration="{sequence_duration_fcpxml_str}" tcStart="0/1s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">')
    fcpxml_content.append('                    <spine>')

    # Clips in Spine
    for i, segment in enumerate(all_segments_data):
        asset_actual_filename = segment['_actual_asset_filename'] # Get the processed name
        asset_ref_id = assets_info[asset_actual_filename]['id']
        
        edited_start_ms_str = segment['edited_start_time_ms']
        original_start_ms_str = segment['original_start_time_ms']
        edited_end_ms_str = segment['edited_end_time_ms']

        clip_offset_fcpxml_str = ms_to_fcpxml_time_str(edited_start_ms_str)
        clip_media_start_fcpxml_str = ms_to_fcpxml_time_str(original_start_ms_str)
        
        clip_duration_ms = float(edited_end_ms_str) - float(edited_start_ms_str)
        if clip_duration_ms < 0:
            print(f"Warning: Segment {i+1} ('{asset_actual_filename}') has negative calculated duration ({clip_duration_ms}ms). Setting duration to 0.")
            clip_duration_ms = 0.0 
        clip_duration_fcpxml_str = ms_to_fcpxml_time_str(str(clip_duration_ms))

        # Using asset-clip as it directly references an asset from resources
        # Added audioRole and adjust-volume from typical Resolve exports
        fcpxml_content.append(f'                        <asset-clip name="{asset_actual_filename} (Segment {i+1})" offset="{clip_offset_fcpxml_str}" duration="{clip_duration_fcpxml_str}" start="{clip_media_start_fcpxml_str}" ref="{asset_ref_id}" format="{MAIN_FORMAT_ID}" tcFormat="NDF" audioRole="dialogue" enabled="1">')
        fcpxml_content.append( '                            <adjust-transform scale="1 1" position="0 0" anchor="0 0"/>')
        fcpxml_content.append( '                            <adjust-volume amount="0dB"/>') # Default volume
        fcpxml_content.append( '                        </asset-clip>')
        
    fcpxml_content.append('                    </spine>')
    fcpxml_content.append('                </sequence>')
    fcpxml_content.append('            </project>')
    fcpxml_content.append('        </event>')
    fcpxml_content.append('    </library>')
    fcpxml_content.append('</fcpxml>')

    # 3. Write to FCPXML file
    try:
        with open(OUTPUT_FCPXML_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fcpxml_content))
        print(f"FCPXML file successfully created at: {OUTPUT_FCPXML_PATH}")
    except Exception as e:
        print(f"Error writing FCPXML file: {e}")

if __name__ == '__main__':
    main()