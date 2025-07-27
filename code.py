import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from pathlib import Path
from fractions import Fraction # For precise frame duration fractions

# --- Configuration ---
WORKING_DIRECTORY = r"d:\Videos\beyond the time 2"  # Use raw string for Windows paths
VIDEO_FRAMERATE = 24.976  # Frames per second of the source/timeline
CSV_FILE_PATH = 'final_video_segments_refined.csv'
FCPXML_OUTPUT_PATH = 'converted_from_csv.fcpxml'
DEFAULT_VIDEO_EXTENSION = ".mp4" # Assumed extension for files in original_video_name

# --- Helper Functions ---
def ms_to_fcpxml_time(ms_float):
    """Converts milliseconds (float) to FCPXML time string "numerator/1000s"."""
    return f"{int(round(ms_float))}/1000s"

def fps_to_frame_duration_str(fps):
    """Converts FPS to FCPXML frameDuration string "numerator/denominator_s"."""
    if fps == 23.976: # Common NTSC rate
        return "1001/24000s"
    f = Fraction(1 / fps).limit_denominator(max_denominator=100000)
    return f"{f.numerator}/{f.denominator}s"

def prettify_xml(element):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

# --- Main Script ---
def main():
    # 1. Read CSV data and preprocess
    segments = []
    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8-sig') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                try:
                    segments.append({
                        'edited_start_time_ms': float(row['edited_start_time_ms']),
                        'original_video_name': row['original_video_name'],
                        'original_start_time_ms': float(row['original_start_time_ms']),
                        'edited_end_time_ms': float(row['edited_end_time_ms']),
                        'original_end_time_ms': float(row['original_end_time_ms']),
                    })
                except ValueError as e:
                    print(f"Skipping row due to data conversion error: {row} - {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if not segments:
        print("No segments found in CSV or CSV could not be read.")
        return

    # Calculate unique assets and their estimated total durations
    assets_info = {}
    for seg in segments:
        name_key = seg['original_video_name']
        if name_key not in assets_info:
            assets_info[name_key] = {
                'max_original_end_ms': 0.0,
                'file_name': f"{name_key}{DEFAULT_VIDEO_EXTENSION}"
            }
        assets_info[name_key]['max_original_end_ms'] = max(
            assets_info[name_key]['max_original_end_ms'],
            seg['original_end_time_ms']
        )

    # Assign asset IDs (r1, r2, ...)
    asset_id_counter = 1
    for name_key in assets_info:
        assets_info[name_key]['id'] = f"r{asset_id_counter}"
        asset_id_counter += 1
    
    # Calculate total sequence duration
    if segments:
        total_sequence_duration_ms = max(seg['edited_end_time_ms'] for seg in segments)
    else:
        total_sequence_duration_ms = 0.0
    
    fcpxml_frame_duration = fps_to_frame_duration_str(VIDEO_FRAMERATE)
    one_frame_duration_ms = (1.0 / VIDEO_FRAMERATE) * 1000.0

    # 2. Build XML structure
    fcpxml_root = ET.Element("fcpxml", version="1.8")

    # Resources
    resources = ET.SubElement(fcpxml_root, "resources")
    # Format (using 'r0' as ID like in the example)
    video_format = ET.SubElement(resources, "format", id="r0", name=f"FFVideoFormat1080p{VIDEO_FRAMERATE:.2f}".replace(".",""),
                                 width="1920", height="1080", frameDuration=fcpxml_frame_duration)

    # Assets
    for name_key, info in assets_info.items():
        asset_path = Path(WORKING_DIRECTORY) / info['file_name']
        asset_uri = asset_path.as_uri() 
        if asset_uri.startswith("file:///") and not asset_uri.startswith("file://localhost/"):
             asset_uri = asset_uri.replace("file:///", "file://localhost/", 1)

        ET.SubElement(resources, "asset", id=info['id'], name=info['file_name'],
                      start="0/1s", 
                      duration=ms_to_fcpxml_time(info['max_original_end_ms']),
                      hasVideo="1", format="r0", 
                      audioSources="1", hasAudio="1", audioChannels="2", 
                      src=asset_uri)

    # Library
    library = ET.SubElement(fcpxml_root, "library")
    event = ET.SubElement(library, "event", name="Converted Event") 
    project = ET.SubElement(event, "project", name="Converted Project") 

    sequence = ET.SubElement(project, "sequence", format="r0",
                             duration=ms_to_fcpxml_time(total_sequence_duration_ms),
                             tcStart="0/1s", tcFormat="NDF") 
    spine = ET.SubElement(sequence, "spine")

    # Clips in spine
    for seg_idx, seg in enumerate(segments):
        asset_id = assets_info[seg['original_video_name']]['id']
        asset_total_duration_ms = assets_info[seg['original_video_name']]['max_original_end_ms']
        clip_name = assets_info[seg['original_video_name']]['file_name']
        
        edited_duration_ms = seg['edited_end_time_ms'] - seg['edited_start_time_ms']
        if edited_duration_ms <= 0:
            edited_duration_ms = one_frame_duration_ms

        start_fcpxml = ms_to_fcpxml_time(seg['edited_start_time_ms'])
        offset_fcpxml = ms_to_fcpxml_time(seg['original_start_time_ms']) 
        duration_fcpxml = ms_to_fcpxml_time(edited_duration_ms)

        clip_element = ET.SubElement(spine, "clip",
                                     name=clip_name,
                                     offset=offset_fcpxml, 
                                     duration=duration_fcpxml, 
                                     start=start_fcpxml, 
                                     tcFormat="NDF", enabled="1", format="r0"
                                     )
        
        ET.SubElement(clip_element, "adjust-transform", scale="1 1", position="0 0", anchor="0 0")

        # Video sub-element
        # This is the corrected call, ensuring standard equals signs and commas
        ET.SubElement(clip_element, "video",
                      ref=asset_id,
                      start="0/1s",
                      offset="0/1s", 
                      duration=ms_to_fcpxml_time(asset_total_duration_ms),
                      enabled="1"
                      )

    # 3. Write to FCPXML file
    rough_string = ET.tostring(fcpxml_root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml_as_string = reparsed.toprettyxml(indent="    ", newl="\n")
    
    lines = pretty_xml_as_string.splitlines()
    if lines and lines[0].strip().startswith("<?xml"): # Check if lines is not empty
        final_output_string = lines[0] + "\n" + \
                              "<!DOCTYPE fcpxml>" + "\n" + \
                              "\n".join(lines[1:])
    else: 
        final_output_string = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + \
                              "<!DOCTYPE fcpxml>\n" + \
                              pretty_xml_as_string # This case might need adjustment if pretty_xml_as_string is empty
                              
    try:
        with open(FCPXML_OUTPUT_PATH, 'w', encoding='utf-8') as outfile:
            outfile.write(final_output_string)
        print(f"Successfully converted CSV to FCPXML: {FCPXML_OUTPUT_PATH}")
    except IOError:
        print(f"Error: Could not write FCPXML file to {FCPXML_OUTPUT_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred during file writing: {e}")


if __name__ == '__main__':
    main()