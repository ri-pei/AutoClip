import csv
import os
from fractions import Fraction
from urllib.parse import quote

# Configuration variables
WORKING_DIRECTORY = r"D:\Videos\beyond the time 2"
FRAME_RATE = 24.976  # fps

def ms_to_frames(ms, fps=24.976):
    """Convert milliseconds to frame count"""
    return int(round((ms / 1000.0) * fps))

def frames_to_rational(frames, fps_base=24000):
    """Convert frame count to rational time format"""
    # Scale frames to the base fps (typically 24000 for FCPXML)
    scaled_frames = int(round(frames * fps_base / 24.976))
    return f"{scaled_frames}/{fps_base}s"

def get_video_duration_rational():
    """
    Get video duration in rational format.
    Using a standard duration that should cover most video content.
    """
    return "8259251/24000s"

def create_fcpxml_from_csv(csv_file, output_file):
    """Convert CSV segments to FCPXML format"""
    
    # Read CSV data
    segments = []
    video_sources = set()
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append({
                'edited_start_ms': float(row['edited_start_time_ms']),
                'edited_end_ms': float(row['edited_end_time_ms']),
                'original_start_ms': float(row['original_start_time_ms']),
                'original_end_ms': float(row['original_end_time_ms']),
                'video_name': row['original_video_name']
            })
            video_sources.add(row['original_video_name'])
    
    # Calculate total duration from last segment
    total_duration_ms = segments[-1]['edited_end_ms']
    total_duration_frames = ms_to_frames(total_duration_ms)
    
    # Start building FCPXML
    fcpxml_content = []
    fcpxml_content.append('<?xml version="1.0" encoding="UTF-8"?>')
    fcpxml_content.append('<!DOCTYPE fcpxml>')
    fcpxml_content.append('<fcpxml version="1.8">')
    
    # Resources section - Create only ONE asset per unique video file
    fcpxml_content.append('    <resources>')
    fcpxml_content.append('        <format width="1920" frameDuration="1001/24000s" name="FFVideoFormat1080p2398" height="1080" id="r0"/>')
    
    # Create a single asset for each unique video source
    video_to_id = {}
    for i, video_name in enumerate(sorted(video_sources)):
        asset_id = f"r{i+1}"
        video_to_id[video_name] = asset_id
        
        # Create file path with proper URL encoding
        video_file = f"{video_name}.mp4"
        file_path = os.path.join(WORKING_DIRECTORY, video_file).replace('\\', '/')
        encoded_path = quote(file_path, safe=':/')
        
        video_duration = get_video_duration_rational()
        
        fcpxml_content.append(f'        <asset start="0/1s" name="{video_file}" audioSources="1" hasVideo="1" format="r0" id="{asset_id}" src="file://localhost/{encoded_path}" duration="{video_duration}" hasAudio="1" audioChannels="2"/>')
    
    fcpxml_content.append('    </resources>')
    
    # Library section
    fcpxml_content.append('    <library>')
    fcpxml_content.append('        <event name="Converted Timeline">')
    fcpxml_content.append('            <project name="Converted Timeline">')
    
    # Calculate sequence duration using proper rational format
    sequence_duration = frames_to_rational(total_duration_frames, 6000)  # Using 6000 base like the sample
    fcpxml_content.append(f'                <sequence tcStart="0/1s" format="r0" tcFormat="NDF" duration="{sequence_duration}">')
    fcpxml_content.append('                    <spine>')
    
    # Create clips for each segment
    cumulative_frames = 0
    
    for i, segment in enumerate(segments):
        asset_id = video_to_id[segment['video_name']]
        
        # Calculate frame-based durations
        edited_start_frames = ms_to_frames(segment['edited_start_ms'])
        edited_end_frames = ms_to_frames(segment['edited_end_ms'])
        original_start_frames = ms_to_frames(segment['original_start_ms'])
        original_end_frames = ms_to_frames(segment['original_end_ms'])
        
        # Duration in frames (use edited duration for consistency)
        duration_frames = edited_end_frames - edited_start_frames
        
        # Convert to rational time formats
        offset_rational = frames_to_rational(cumulative_frames, 4800)  # Timeline position
        start_rational = frames_to_rational(original_start_frames, 1)  # Source media start
        duration_rational = frames_to_rational(duration_frames, 24000)  # Clip duration
        
        # Get video file name
        video_file = f"{segment['video_name']}.mp4"
        video_duration = get_video_duration_rational()
        
        # Add clip to spine
        fcpxml_content.append(f'                        <clip start="{start_rational}" offset="{offset_rational}" name="{video_file}" enabled="1" format="r0" tcFormat="NDF" duration="{duration_rational}">')
        fcpxml_content.append('                            <adjust-transform scale="1 1" position="0 0" anchor="0 0"/>')
        fcpxml_content.append(f'                            <video start="0/1s" offset="0/1s" ref="{asset_id}" duration="{video_duration}"/>')
        fcpxml_content.append('                        </clip>')
        
        # Update cumulative position for next clip
        cumulative_frames += duration_frames
        
        # Debug output
        print(f"Clip {i+1}: offset={cumulative_frames-duration_frames} frames, duration={duration_frames} frames, source_start={original_start_frames} frames")
    
    # Close tags
    fcpxml_content.append('                    </spine>')
    fcpxml_content.append('                </sequence>')
    fcpxml_content.append('            </project>')
    fcpxml_content.append('        </event>')
    fcpxml_content.append('    </library>')
    fcpxml_content.append('</fcpxml>')
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fcpxml_content))
    
    print(f"\nFCPXML file created: {output_file}")
    print(f"Total segments: {len(segments)}")
    print(f"Video sources: {len(video_sources)} (should match Media Pool items)")
    print(f"Total duration: {total_duration_ms/1000:.2f} seconds ({total_duration_frames} frames)")
    
    # Verify continuity
    print("\nSegment continuity check:")
    for i, segment in enumerate(segments):
        start_ms = segment['edited_start_ms']
        end_ms = segment['edited_end_ms']
        if i > 0:
            prev_end = segments[i-1]['edited_end_ms']
            gap = start_ms - prev_end
            if abs(gap) > 1:  # Allow 1ms tolerance
                print(f"  WARNING: Gap of {gap}ms between segment {i} and {i+1}")
        print(f"  Segment {i+1}: {start_ms}ms - {end_ms}ms (duration: {end_ms-start_ms}ms)")

def main():
    """Main function to run the conversion"""
    csv_file = "final_video_segments_refined.csv"
    output_file = "converted_timeline.fcpxml"
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found!")
        return
    
    try:
        create_fcpxml_from_csv(csv_file, output_file)
        print("\nConversion completed successfully!")
        print("\nTroubleshooting tips:")
        print("- Check that all video files exist in the working directory")
        print("- Verify file paths are accessible to DaVinci Resolve")
        print("- If clips still missing, check the debug output for timing issues")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
