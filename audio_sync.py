import os
import subprocess
import tempfile
import shutil
import argparse
import re

def check_tools():
    """Checks if ffmpeg and audio-offset-finder are accessible."""
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found. Please install it and add to PATH.")
        return False
    # audio-offset-finder is a python script usually installed with pip
    # We can check if its command-line interface is callable.
    try:
        subprocess.run(["audio-offset-finder", "-h"], 
                       capture_output=True, check=True, text=True, encoding='utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: audio-offset-finder command not found or not working.")
        print("       Please ensure it's installed (pip install audio-offset-finder) and your environment is correct.")
        return False
    return True

def run_command(command, cwd=None):
    """Runs a command and prints its output, raises exception on error."""
    print(f"Running: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', cwd=cwd)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print("--- STDOUT ---")
        print(stdout)
        print("--- STDERR ---")
        print(stderr)
        raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
    # print(stdout) # Optionally print stdout for successful commands too
    return stdout

def convert_to_wav(input_file, output_wav, temp_dir, sample_rate=48000, channels=2):
    """Converts an audio or video file's first audio stream to WAV."""
    print(f"Converting '{os.path.basename(input_file)}' to WAV...")
    output_path = os.path.join(temp_dir, output_wav)
    
    cmd = [
        "ffmpeg", "-y",  # Overwrite output files without asking
        "-i", input_file,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # WAV format (signed 16-bit little-endian PCM)
        "-ar", str(sample_rate), # Audio sample rate
        "-ac", str(channels),    # Audio channels (stereo)
        "-map_metadata", "-1", # Remove metadata
        # Ensure we select ONLY the first audio stream.
        # If no audio stream 0:a:0 exists, ffmpeg will error out,
        # which is desirable as the script requires audio.
        "-map", "0:a:0",
        output_path # This must be the last argument for output file
    ]

    # The previous conditional logic that could add a duplicate "-map 0:a:0" is removed.
    # The single "-map 0:a:0" above handles selecting the first audio stream.

    try:
        run_command(cmd)
    except subprocess.CalledProcessError as e:
        # Check if the error is specifically "Stream map ... matches no streams"
        # This can happen if the input video genuinely has no audio.
        if e.stderr and ("Stream map '0:a:0' matches no streams" in e.stderr or \
                         "Stream map 'a:0' matches no streams" in e.stderr): # 'a:0' for audio-only files
            print(f"ERROR: No audio stream found in '{os.path.basename(input_file)}' to convert to WAV.")
            raise ValueError(f"No audio stream found in file: {input_file}") from e
        else:
            # For other ffmpeg errors, re-raise the original exception
            raise
    return output_path

def get_audio_offset(wav1_path, wav2_path, temp_dir, resolution=256):
    """
    Finds offset between two WAV files using audio-offset-finder.
    A positive offset means wav2_path (external audio) is LATER than wav1_path (video audio).
    """
    print(f"Finding offset: '{os.path.basename(wav1_path)}' within '{os.path.basename(wav2_path)}' using resolution {resolution}...")
    
    cmd = [
        "audio-offset-finder",
        "--find-offset-of", wav1_path,
        "--within", wav2_path,
        "--resolution", str(resolution),
    ]
    
    output = run_command(cmd, cwd=temp_dir)
    
    # Make regex more flexible for "Offset:" line
    # Handles "Offset: <value> seconds" or "Offset: <value> (seconds)"
    # Also handles optional INFO:root prefix or similar logging prefixes
    offset_match = re.search(r"Offset:\s*([\-\d\.]+)\s*(?:\(seconds\)|seconds)?", output, re.IGNORECASE)
    
    # Make regex more flexible for "Score:" line
    # Handles "Score: <value>" or "Standard score: <value>"
    score_match = re.search(r"(?:Standard\s)?Score:\s*([\d\.]+)", output, re.IGNORECASE)

    if not offset_match:
        print("--- audio-offset-finder output ---")
        print(output)
        raise ValueError("Could not parse offset from audio-offset-finder output.")
    
    offset_seconds = float(offset_match.group(1))
    score = 0.0
    if score_match:
        try:
            score = float(score_match.group(1))
        except ValueError:
            print(f"Warning: Could not parse score from: {score_match.group(1)}")

    print(f"Offset found: {offset_seconds:.4f} seconds (positive means external audio is LATER than video audio)")
    print(f"Confidence score: {score:.4f}")

    min_score = 0.1 # Adjust as needed. Your score of 20 is very good.
    if score < min_score and score != 0.0:
        print(f"WARNING: Confidence score {score:.4f} is below threshold {min_score}. Sync might be poor.")
            
    return offset_seconds

def replace_audio_with_offset(video_file, external_audio_file, output_file, offset_seconds):
    """
    Replaces audio in video_file with external_audio_file, applying offset.
    - video_file: Path to the original video.
    - external_audio_file: Path to the new audio track (original format, not WAV).
    - output_file: Path for the new MKV.
    - offset_seconds: Calculated offset. Positive means external_audio_file starts LATER
                      than video_file's original audio.
                      So, if offset_seconds > 0, external_audio needs to be shifted *earlier* / video delayed.
                      If offset_seconds < 0, external_audio needs to be shifted *later* / video advanced.
    """
    print(f"Creating output MKV: '{output_file}' with offset {offset_seconds:.4f}s")
    
    # FFmpeg command construction:
    # - Copy video stream from original video.
    # - Use the external audio stream.
    # - Apply offset.
    
    # If offset_seconds > 0: external audio (file2) is LATER than video audio (file1).
    # To sync, we need to "start" the external audio earlier by `offset_seconds`.
    # This is achieved by using -ss on the external audio input.
    # Or, delay the video using -itsoffset on the video input.
    # Let's apply modification to the external audio.

    # If offset_seconds < 0: external audio (file2) is EARLIER than video audio (file1).
    # To sync, we need to delay the external audio by `abs(offset_seconds)`.
    # This is achieved by the `adelay` filter.

    cmd = [
        "ffmpeg", "-y",
        "-i", video_file, # Input 0 (video)
    ]

    # Handle the offset for the external audio file (Input 1)
    delay_ms_str = ""
    if offset_seconds > 0: # External audio is LATE, needs to be shifted EARLIER (trim start)
        cmd.extend(["-ss", str(offset_seconds), "-i", external_audio_file])
        print(f"   External audio is LATE by {offset_seconds:.3f}s. Seeking into external audio by this amount.")
    elif offset_seconds < 0: # External audio is EARLY, needs to be shifted LATER (add delay)
        cmd.extend(["-i", external_audio_file])
        delay_ms = int(abs(offset_seconds) * 1000)
        delay_ms_str = f"{delay_ms}|{delay_ms}" # For stereo, apply to all channels
        print(f"   External audio is EARLY by {abs(offset_seconds):.3f}s. Delaying external audio by {delay_ms}ms.")
    else: # No offset
        cmd.extend(["-i", external_audio_file])
        print("   No offset. Using external audio as is.")

    cmd.extend([
        "-map", "0:v:0",         # Map video from first input (video_file)
        "-map", "1:a:0",         # Map audio from second input (external_audio_file)
        "-c:v", "copy",          # Copy video stream without re-encoding
        "-c:a", "aac",           # Re-encode new audio to AAC (common for MKV)
                                 # Or use "copy" if external_audio_file is already in a good format (e.g. AAC, AC3)
                                 # and doesn't need adelay. If adelay is used, re-encoding is often necessary.
        "-b:a", "320k",          # Example bitrate for AAC
    ])

    if delay_ms_str: # Only add adelay filter if needed
        cmd.extend(["-af", f"adelay={delay_ms_str},aresample=async=1"]) # aresample for safety with adelay
        # If using "-c:a copy" with adelay, ffmpeg might complain or fail. Re-encoding is safer.
        # If you want to copy and adelay is needed, it's more complex (e.g. -itsoffset on the audio input, but that has other implications)

    cmd.append(output_file)
    
    run_command(cmd)
    print(f"Successfully created '{output_file}'")


def main():
    if not check_tools():
        return

    parser = argparse.ArgumentParser(description="Replace video audio track with an external one, synchronizing them.")
    parser.add_argument("video_file", help="Path to the input video file (e.g., input.mp4)")
    parser.add_argument("external_audio_file", help="Path to the external audio file (e.g., track.mp3)")
    parser.add_argument("-o", "--output_file", help="Path for the output MKV file. Defaults to [video_basename]_synced.mkv")
    parser.add_argument("--sample_rate", type=int, default=48000, help="Sample rate for temporary WAV conversion (Hz). Default: 48000")
    parser.add_argument("--channels", type=int, default=2, help="Number of channels for temporary WAV conversion. Default: 2 (stereo)")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary WAV files after processing.")
    parser.add_argument("--aof_resolution", type=int, default=128, 
                        help="Resolution (samples) for audio-offset-finder analysis. "
                             "Smaller is more precise but slower. Default: 256 (results in ~64 sample hop). "
                             "Audio-offset-finder's own default is 1024.")


    args = parser.parse_args()

    # ... (rest of main: file checks, output_file logic, temp_dir creation) ...
    if not os.path.exists(args.video_file):
        print(f"Error: Video file not found: {args.video_file}")
        return
    if not os.path.exists(args.external_audio_file):
        print(f"Error: External audio file not found: {args.external_audio_file}")
        return

    output_file = args.output_file
    if not output_file:
        base, _ = os.path.splitext(args.video_file)
        output_file = f"{base}_synced.mkv"

    temp_dir = tempfile.mkdtemp(prefix="audio_sync_")
    print(f"Temporary directory: {temp_dir}")


    try:
        # 1. Convert video audio to WAV
        video_audio_wav = convert_to_wav(args.video_file, "video_audio.wav", temp_dir, args.sample_rate, args.channels)
        
        # 2. Convert external audio to WAV
        external_audio_wav = convert_to_wav(args.external_audio_file, "external_audio.wav", temp_dir, args.sample_rate, args.channels)

        # 3. Find offset
        offset = get_audio_offset(video_audio_wav, external_audio_wav, temp_dir, resolution=args.aof_resolution)

        # 4. Create new MKV with synchronized audio
        replace_audio_with_offset(args.video_file, args.external_audio_file, output_file, offset)

    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        print(f"An error occurred: {e}")
    finally:
        if not args.keep_temp and os.path.exists(temp_dir):
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        elif args.keep_temp:
            print(f"Temporary files kept at: {temp_dir}")

if __name__ == "__main__":
    main()