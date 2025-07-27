import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from pathlib import Path
from fractions import Fraction

# --- Configuration ---
WORKING_DIRECTORY = r"d:\Videos\beyond the time 2"
VIDEO_FRAMERATE = 24.976 # 23.976 might be more standard if source is film-like
CSV_FILE_PATH = 'final_video_segments_refined.csv'
FCPXML_OUTPUT_PATH = 'converted_from_csv_corrected_logic_v2.fcpxml' # New output name
DEFAULT_VIDEO_EXTENSION = ".mp4"

# --- Helper Functions ---
def ms_to_fcpxml_time(ms_float):
    return f"{int(round(ms_float))}/1000s"

def fps_to_frame_duration_str(fps):
    if fps == 23.976: # A common film/NTSC-compatible rate
        return "1001/24000s"
    # For 24.976, this will be 125/3122s
    f = Fraction(1 / fps).limit_denominator(max_denominator=100000)
    return f"{f.numerator}/{f.denominator}s"

def prettify_xml(element):
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

# --- Main Script ---
def main():
    segments = []
    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8-sig') as infile:
            reader = csv.DictReader(infile)
            for row_idx, row in enumerate(reader):
                try:
                    edited_start_time_ms = float(row['edited_start_time_ms'])
                    original_start_time_ms = float(row['original_start_time_ms'])
                    edited_end_time_ms = float(row['edited_end_time_ms']) # Still needed for source segment calc
                    original_end_time_ms = float(row['original_end_time_ms'])

                    segments.append({
                        'edited_start_time_ms': edited_start_time_ms,
                        'original_video_name': row['original_video_name'],
                        'original_start_time_ms': original_start_time_ms,
                        'edited_end_time_ms': edited_end_time_ms, # Still needed for source segment calc
                        'original_end_time_ms': original_end_time_ms,
                    })
                except ValueError as e:
                    print(f"Skipping row {row_idx + 1} due to data conversion error: {row} - {e}")
                    continue
                except KeyError as e:
                    print(f"Skipping row {row_idx + 1} due to missing CSV column: {e} in row {row}")
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

    assets_info = {}
    for seg in segments:
        name_key = seg['original_video_name']
        if name_key not in assets_info:
            assets_info[name_key] = {
                'max_source_media_out_point_ms': 0.0, # Max out-point reached in this source file
                'file_name': f"{name_key}{DEFAULT_VIDEO_EXTENSION}",
                'id': ''
            }
        
        # Under the new logic:
        # Source In-Point = seg['edited_start_time_ms']
        # Duration Used from Source = seg['original_end_time_ms'] - seg['original_start_time_ms']
        # (This source duration MUST be positive)
        source_duration_for_this_segment_ms = seg['original_end_time_ms'] - seg['original_start_time_ms']
        if source_duration_for_this_segment_ms < 0: source_duration_for_this_segment_ms = 0 # Cannot be negative

        source_out_point_for_this_segment_ms = seg['edited_start_time_ms'] + source_duration_for_this_segment_ms
        
        assets_info[name_key]['max_source_media_out_point_ms'] = max(
            assets_info[name_key]['max_source_media_out_point_ms'],
            source_out_point_for_this_segment_ms
        )

    asset_id_counter = 1
    for name_key in assets_info:
        assets_info[name_key]['id'] = f"r{asset_id_counter}"
        asset_id_counter += 1
    
    if segments:
        total_sequence_duration_ms = max(seg['original_end_time_ms'] for seg in segments if seg['original_end_time_ms'] is not None)
    else:
        total_sequence_duration_ms = 0.0
    
    fcpxml_frame_duration = fps_to_frame_duration_str(VIDEO_FRAMERATE)
    one_frame_duration_ms = (1.0 / VIDEO_FRAMERATE) * 1000.0

    fcpxml_root = ET.Element("fcpxml", version="1.8")
    resources = ET.SubElement(fcpxml_root, "resources")
    video_format = ET.SubElement(resources, "format", id="r0", name=f"FFVideoFormat1080p{VIDEO_FRAMERATE:.2f}".replace(".",""),
                                 width="1920", height="1080", frameDuration=fcpxml_frame_duration)

    for name_key, info in assets_info.items():
        asset_path = Path(WORKING_DIRECTORY) / info['file_name']
        asset_uri = asset_path.as_uri() 
        if asset_uri.startswith("file:///") and not asset_uri.startswith("file://localhost/"):
             asset_uri = asset_uri.replace("file:///", "file://localhost/", 1)

        # This duration is the estimated total physical duration of the source file
        asset_physical_duration_ms = info['max_source_media_out_point_ms']
        if asset_physical_duration_ms <= 0: # Ensure it's at least one frame for safety
            asset_physical_duration_ms = one_frame_duration_ms

        ET.SubElement(resources, "asset", id=info['id'], name=info['file_name'],
                      start="0/1s", 
                      duration=ms_to_fcpxml_time(asset_physical_duration_ms),
                      hasVideo="1", format="r0", 
                      audioSources="1", hasAudio="1", audioChannels="2", 
                      src=asset_uri)

    library = ET.SubElement(fcpxml_root, "library")
    event = ET.SubElement(library, "event", name="Converted Event") 
    project = ET.SubElement(event, "project", name="Converted Project") 

    sequence = ET.SubElement(project, "sequence", format="r0",
                             duration=ms_to_fcpxml_time(total_sequence_duration_ms),
                             tcStart="0/1s", tcFormat="NDF") 
    spine = ET.SubElement(sequence, "spine")

    for seg_idx, seg in enumerate(segments):
        asset_id = assets_info[seg['original_video_name']]['id']
        clip_name = assets_info[seg['original_video_name']]['file_name']
        
        timeline_start_ms = seg['original_start_time_ms']
        source_offset_ms = seg['edited_start_time_ms']
        
        timeline_clip_duration_ms = seg['original_end_time_ms'] - seg['original_start_time_ms']
        
        if timeline_clip_duration_ms <= 0:
            timeline_clip_duration_ms = one_frame_duration_ms

        timeline_start_fcpxml = ms_to_fcpxml_time(timeline_start_ms)
        source_offset_fcpxml = ms_to_fcpxml_time(source_offset_ms)
        timeline_duration_fcpxml = ms_to_fcpxml_time(timeline_clip_duration_ms)

        clip_element = ET.SubElement(spine, "clip",
                                     name=clip_name,
                                     offset=source_offset_fcpxml,
                                     duration=timeline_duration_fcpxml,
                                     start=timeline_start_fcpxml,
                                     tcFormat="NDF", enabled="1", format="r0"
                                     )
        
        ET.SubElement(clip_element, "adjust-transform", scale="1 1", position="0 0", anchor="0 0")

        asset_physical_total_duration_ms = assets_info[seg['original_video_name']]['max_source_media_out_point_ms']
        if asset_physical_total_duration_ms <= 0: # Safety for video element duration
            asset_physical_total_duration_ms = one_frame_duration_ms


        ET.SubElement(clip_element, "video",
                      ref=asset_id,
                      start="0/1s",
                      offset="0/1s", 
                      duration=ms_to_fcpxml_time(asset_physical_total_duration_ms),
                      enabled="1"
                      )

    rough_string = ET.tostring(fcpxml_root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml_as_string = reparsed.toprettyxml(indent="    ", newl="\n")
    
    lines = pretty_xml_as_string.splitlines()
    if lines and lines[0].strip().startswith("<?xml"):
        final_output_string = lines[0] + "\n" + \
                              "<!DOCTYPE fcpxml>" + "\n" + \
                              "\n".join(lines[1:])
    else: 
        final_output_string = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + \
                              "<!DOCTYPE fcpxml>\n" + \
                              pretty_xml_as_string
                              
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