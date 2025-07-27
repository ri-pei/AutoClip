import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom # For pretty printing
import os
from pathlib import Path # For robust path handling

# --- SCRIPT CONFIGURATION ---
# WORK_DIR: Path to the directory containing original video files.
# Example: "D:/Videos/beyond the time 2/" or "/mnt/videos/project_x/"
# IMPORTANT: Ensure this path uses forward slashes '/' even on Windows for URI compatibility.
WORK_DIR = "D:/Videos/beyond the time 2/"

# FRAME_RATE_FLOAT: Video frame rate as a float (e.g., 23.976, 25.0, 29.97)
FRAME_RATE_FLOAT = 23.976

# FRAME_DURATION_STR: Frame duration as a string "numerator/denominator s"
# This is CRITICAL for all time calculations in FCPXML.
# Example: "1001/24000s" for 23.976 FPS
# Example: "1/25s" for 25 FPS
# Example: "1001/30000s" for 29.97 FPS
FRAME_DURATION_STR = "1001/24000s" # Matches example

# VIDEO_WIDTH: Video width in pixels (e.g., 1920)
VIDEO_WIDTH = 1920

# VIDEO_HEIGHT: Video height in pixels (e.g., 1080)
VIDEO_HEIGHT = 1080

# FCPXML FORMAT NAME: Name for the <format> element in <resources>
# Example: "FFVideoFormat1080p2398"
FCPXML_FORMAT_NAME = f"FFVideoFormat{VIDEO_HEIGHT}p{str(FRAME_RATE_FLOAT).replace('.', '')}" # e.g., FFVideoFormat1080p23976

# EVENT AND PROJECT NAMES (can be customized)
EVENT_NAME = "Timeline 1 (Resolve)"
PROJECT_NAME = "Timeline 1 (Resolve)"

# ASSET DURATION PLACEHOLDER: A large duration for source assets if their true duration isn't known.
# The format must match "frames_x_numerator/denominator s".
# Example: "172832660/24000s" (from provided example, roughly 2 hours at 23.976)
# This assumes 172832660 is already (total_frames_in_source_asset * frame_duration_numerator)
# For a 2-hour asset at 23.976 (1001/24000s):
# 2 hours * 3600 s/hr * (24000 / 1001) frames/s = ~172654 frames
# 172654 * 1001 = 172826654. So the example value is close to a 2h asset.
# Let's use the example's value, as it's just a placeholder for Resolve.
ASSET_DURATION_PLACEHOLDER_NUMERATOR_PRODUCT = 172832660 # This is (total_frames * fd_num)

# --- HELPER FUNCTIONS ---

def parse_frame_duration(fd_str):
    """Parses FRAME_DURATION_STR into numerator and denominator."""
    try:
        parts = fd_str.replace('s', '').split('/')
        if len(parts) != 2:
            raise ValueError("FRAME_DURATION_STR must be in 'num/den s' format.")
        num = int(parts[0])
        den = int(parts[1])
        return num, den
    except Exception as e:
        print(f"Error parsing FRAME_DURATION_STR ('{fd_str}'): {e}")
        raise

def format_time_value(frames, fd_numerator, fd_denominator):
    """Formats frame count into 'value/denominator s' string."""
    # The 'value' here is frames * fd_numerator
    return f"{int(frames * fd_numerator)}/{fd_denominator}s"

def format_fcpxml_path(work_dir, filename):
    """Formats a local file path into a file URI for FCPXML."""
    # Ensure work_dir ends with a slash if it doesn't have one
    if not work_dir.endswith('/'):
        work_dir += '/'
    
    # Construct the full path using Path for OS-agnostic joining
    # Note: WORK_DIR is expected to use '/'
    full_path_str = work_dir + filename
    
    # Create a Path object. We assume WORK_DIR is absolute.
    # If WORK_DIR could be relative, os.path.abspath would be needed first.
    # For 'file://localhost/', the path should be absolute.
    # Python's pathlib.Path.as_uri() handles spaces (%20) and drive letters correctly.
    # On Windows, as_uri() produces 'file:///D:/path/to/file'.
    # The example has 'file://localhost/D:/...'.
    # We'll manually construct to match the example's 'localhost' style if needed,
    # but Resolve usually handles 'file:///' fine. Let's test as_uri() first.
    
    # Pathlib's as_uri converts 'D:\path' to 'file:///D:/path'
    # The example uses 'file://localhost/D:/path'
    # Let's try to conform strictly.
    
    # Ensure forward slashes for the path part
    path_for_uri = Path(full_path_str).as_posix() 
    
    # For Windows drive letter paths like D:/..., prefix with /
    if ":" in path_for_uri and path_for_uri[1] == ":": # e.g. D:/
        path_for_uri = "/" + path_for_uri
        
    return f"file://localhost{path_for_uri}"


# --- MAIN SCRIPT LOGIC ---

def csv_to_fcpxml(csv_filepath, output_fcpxml_filepath):
    """
    Converts a CSV file with video segment data to an FCPXML file.
    """
    print(f"Starting conversion of '{csv_filepath}' to FCPXML...")

    # Parse frame duration configuration
    try:
        fd_numerator, fd_denominator = parse_frame_duration(FRAME_DURATION_STR)
    except ValueError:
        return # Error already printed

    segments = []
    original_video_names = set()
    max_edited_end_frame = 0

    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                try:
                    segment = {
                        'edited_start_frame': int(row['edited_start_frame']),
                        'edited_end_frame': int(row['edited_end_frame']),
                        'original_video_name': row['original_video_name'],
                        'original_start_frame': int(row['original_start_frame']),
                        # 'original_end_frame': int(row['original_end_frame']), # Not directly used for clip duration
                    }
                    segments.append(segment)
                    original_video_names.add(segment['original_video_name'])
                    if segment['edited_end_frame'] > max_edited_end_frame:
                        max_edited_end_frame = segment['edited_end_frame']
                except KeyError as e:
                    print(f"CSV Error: Missing expected column: {e} in row {i+2}")
                    return
                except ValueError as e:
                    print(f"CSV Error: Invalid integer value in row {i+2}: {e}")
                    return
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{csv_filepath}'")
        return
    except Exception as e:
        print(f"Error reading CSV file '{csv_filepath}': {e}")
        return

    if not segments:
        print("No segments found in CSV file. Aborting.")
        return

    # --- Build FCPXML Structure ---
    fcpxml = ET.Element("fcpxml", version="1.9")

    # ** Resources **
    resources = ET.SubElement(fcpxml, "resources")
    
    # Format definition (shared)
    format_id = "r0"
    fmt = ET.SubElement(resources, "format",
                        id=format_id,
                        name=FCPXML_FORMAT_NAME,
                        width=str(VIDEO_WIDTH),
                        height=str(VIDEO_HEIGHT),
                        frameDuration=FRAME_DURATION_STR)

    # Asset definitions
    asset_map = {} # To map original_video_name to asset_id (r1, r2, ...)
    asset_id_counter = 1
    for name_key in sorted(list(original_video_names)): # Sort for consistent rX IDs
        asset_id = f"r{asset_id_counter}"
        asset_map[name_key] = asset_id
        
        asset_filename = f"{name_key}.mp4" # Assuming .mp4 extension
        asset_src_path = format_fcpxml_path(WORK_DIR, asset_filename)
        
        # Asset duration: use placeholder based on configuration
        asset_duration_str = f"{ASSET_DURATION_PLACEHOLDER_NUMERATOR_PRODUCT}/{fd_denominator}s"

        asset = ET.SubElement(resources, "asset",
                              id=asset_id,
                              name=asset_filename,
                              start=f"0/{fd_denominator}s", # Assets usually start at 0
                              duration=asset_duration_str,
                              hasVideo="1",
                              format=format_id)
        ET.SubElement(asset, "media-rep",
                      kind="original-media",
                      src=asset_src_path)
        asset_id_counter += 1

    # ** Library **
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name=EVENT_NAME)
    project = ET.SubElement(event, "project", name=PROJECT_NAME)

    # Sequence
    # Sequence duration: (last_edited_end_frame + 1) * fd_numerator / fd_denominator
    seq_duration_val = (max_edited_end_frame + 1) * fd_numerator
    seq_duration_str = f"{seq_duration_val}/{fd_denominator}s"
    
    sequence = ET.SubElement(project, "sequence",
                             tcStart=f"0/{fd_denominator}s",
                             duration=seq_duration_str,
                             tcFormat="NDF",
                             format=format_id) # References the format_id ("r0")
    
    spine = ET.SubElement(sequence, "spine")

    # Asset-Clips from CSV segments
    for i, seg_data in enumerate(segments):
        clip_name = f"clip{i+1:04d}" # e.g., clip0001, clip0002
        
        # offset: edited_start_frame * fd_numerator / fd_denominator s
        offset_str = format_time_value(seg_data['edited_start_frame'], fd_numerator, fd_denominator)
        
        # duration: (edited_end_frame - edited_start_frame + 1) * fd_numerator / fd_denominator s
        clip_frame_duration = (seg_data['edited_end_frame'] - seg_data['edited_start_frame'] + 1)
        duration_str = format_time_value(clip_frame_duration, fd_numerator, fd_denominator)
        
        # start: original_start_frame * fd_numerator / fd_denominator s
        start_str = format_time_value(seg_data['original_start_frame'], fd_numerator, fd_denominator)
        
        asset_ref_id = asset_map[seg_data['original_video_name']]

        asset_clip = ET.SubElement(spine, "asset-clip",
                                   name=clip_name,
                                   ref=asset_ref_id,
                                   offset=offset_str,
                                   duration=duration_str,
                                   start=start_str,
                                   tcFormat="NDF",
                                   format=format_id, # References the format_id ("r0")
                                   enabled="1")
        
        # Adjust-transform (can be fixed as per example)
        ET.SubElement(asset_clip, "adjust-transform",
                      scale="1 1",
                      anchor="0 0",
                      position="0 0")

    # --- Output FCPXML ---
    # Add DOCTYPE declaration. ElementTree doesn't handle this directly.
    doctype_str = '<!DOCTYPE fcpxml>\n'
    
    # Convert ElementTree to string
    # Python 3.9+ has ET.indent() for pretty printing.
    # For broader compatibility, use minidom for pretty printing.
    rough_string = ET.tostring(fcpxml, encoding='utf-8', method='xml')
    reparsed = minidom.parseString(rough_string)
    pretty_xml_str = reparsed.toprettyxml(indent="    ", encoding="UTF-8").decode('utf-8')

    # Remove minidom's default XML declaration as we add our own with DOCTYPE
    if pretty_xml_str.startswith("<?xml"):
        pretty_xml_str = pretty_xml_str.split("?>", 1)[1].lstrip()

    final_xml_content = f'<?xml version="1.0" encoding="UTF-8"?>\n{doctype_str}{pretty_xml_str}'
    
    try:
        with open(output_fcpxml_filepath, 'w', encoding='utf-8') as f:
            f.write(final_xml_content)
        print(f"Successfully generated FCPXML: '{output_fcpxml_filepath}'")
    except IOError as e:
        print(f"Error writing FCPXML file '{output_fcpxml_filepath}': {e}")


if __name__ == "__main__":
    # --- Configuration for running the script ---
    input_csv_file = "final_video_segments_refined.csv"
    
    # Generate output filename based on input CSV name
    base_name, _ = os.path.splitext(input_csv_file)
    output_fcpxml_file = f"{base_name}_output.fcpxml"

    # Validate WORK_DIR (must exist)
    if not os.path.isdir(WORK_DIR):
        print(f"Error: WORK_DIR '{WORK_DIR}' does not exist or is not a directory.")
        print("Please configure WORK_DIR at the top of the script.")
    else:
        csv_to_fcpxml(input_csv_file, output_fcpxml_file)