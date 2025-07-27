import os
import subprocess
import pandas as pd
import glob
import json # For parsing 'top_n_matches' if using coarse_match_results.csv
import re

# --- 用户配置 ---
# 工作目录，同之前的脚本
WORKING_DIR = "."

# 剪辑后视频的文件名 (需要包含后缀)
EDITED_VIDEO_FILENAME_WITH_EXT = "my_edited_clip.mp4" # 例如 "my_edited_clip.mp4"

# 包含匹配结果的CSV文件路径 (相对于 WORKING_DIR/OUTPUT_DIR_NAME)
# 如果使用步骤3的粗略匹配结果:
MATCH_RESULTS_CSV = "coarse_match_results2.csv"
# 如果使用步骤4的精确匹配结果 (假设文件名为 fine_match_results.csv):
# MATCH_RESULTS_CSV = "fine_match_results.csv" # 取消注释并修改此行以使用精确匹配结果

# 包含拼接对比图的文件夹路径 (相对于 WORKING_DIR/OUTPUT_DIR_NAME)
# 如果使用步骤3的粗略匹配拼接图:
STITCHED_IMAGES_SUBDIR = "coarse_match_visuals2"
# 如果使用步骤4的精确匹配拼接图 (假设文件夹名为 fine_match_visuals):
# STITCHED_IMAGES_SUBDIR = "fine_match_visuals" # 取消注释并修改此行

# 输出对比视频的文件名 (不包含后缀，会自动添加.mp4)
COMPARISON_VIDEO_OUTPUT_NAME = "comparison_video_with_audio"

# ffmpeg 可执行文件的路径 (如果不在系统PATH中，请指定完整路径)
FFMPEG_PATH = "ffmpeg" # 或者例如 "C:/ffmpeg/bin/ffmpeg.exe"

# 内部常量
OUTPUT_DIR_NAME = "output"
# --- END 用户配置 ---

def run_command(command_list):
    """执行外部命令并返回结果"""
    print(f"Executing command: {' '.join(command_list)}")
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error executing command: {' '.join(command_list)}")
            print(f"FFmpeg/FFprobe STDERR:\n{stderr}")
        else:
            print(f"Command executed successfully.")
            if stdout: print(f"FFmpeg/FFprobe STDOUT:\n{stdout}")
            if stderr: print(f"FFmpeg/FFprobe STDERR (non-fatal):\n{stderr}") # Print non-fatal stderr too
        return stdout, stderr, process.returncode
    except FileNotFoundError:
        print(f"Error: Command {command_list[0]} not found. Is ffmpeg/ffprobe installed and in PATH?")
        raise
    except Exception as e:
        print(f"An unexpected error occurred with command {' '.join(command_list)}: {e}")
        raise

def get_video_fps(video_path):
    """使用 ffprobe 获取视频的平均帧率"""
    cmd = [
        FFMPEG_PATH.replace("ffmpeg", "ffprobe"), # Use ffprobe
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate',
        '-of', 'csv=s=x:p=0', # Output format: numerator/denominator
        video_path
    ]
    stdout, _, returncode = run_command(cmd)
    if returncode == 0 and stdout:
        try:
            if '/' in stdout.strip():
                num, den = map(int, stdout.strip().split('/'))
                return num / den if den != 0 else 0
            else:
                return float(stdout.strip())
        except ValueError:
            print(f"Error: Could not parse FPS from ffprobe output: {stdout.strip()}")
            return None
    return None

def parse_frame_filename_for_timestamp_ms(filename):
    """从帧文件名中提取毫秒级时间戳。格式: ..._time_HH-MM-SS-mmm.png"""
    match = re.search(r"_time_(\d{2}-\d{2}-\d{2}-\d{3})\.png$", filename)
    if match:
        time_str = match.group(1)
        parts = time_str.split('-')
        if len(parts) == 4:
            h, m, s, ms = map(int, parts)
            return (h * 3600 + m * 60 + s) * 1000 + ms
    return None


def main():
    abs_working_dir = os.path.abspath(WORKING_DIR)
    abs_base_output_dir = os.path.join(abs_working_dir, OUTPUT_DIR_NAME)

    edited_video_full_path = os.path.join(abs_working_dir, EDITED_VIDEO_FILENAME_WITH_EXT)
    if not os.path.exists(edited_video_full_path):
        print(f"FATAL: Edited video '{EDITED_VIDEO_FILENAME_WITH_EXT}' not found at '{edited_video_full_path}'.")
        return

    match_results_csv_path = os.path.join(abs_base_output_dir, MATCH_RESULTS_CSV)
    if not os.path.exists(match_results_csv_path):
        print(f"FATAL: Match results CSV '{MATCH_RESULTS_CSV}' not found at '{match_results_csv_path}'.")
        return

    stitched_images_dir = os.path.join(abs_base_output_dir, STITCHED_IMAGES_SUBDIR)
    if not os.path.exists(stitched_images_dir):
        print(f"FATAL: Stitched images directory '{STITCHED_IMAGES_SUBDIR}' not found at '{stitched_images_dir}'.")
        return

    print(f"Reading match results from: {match_results_csv_path}")
    try:
        df_matches = pd.read_csv(match_results_csv_path)
    except Exception as e:
        print(f"Error reading CSV {match_results_csv_path}: {e}")
        return

    if 'edited_frame_filename' not in df_matches.columns:
        print(f"Error: CSV file must contain 'edited_frame_filename' column.")
        return

    # 获取剪辑视频的帧率
    print(f"Getting FPS of the edited video: {edited_video_full_path}")
    fps = get_video_fps(edited_video_full_path)
    if fps is None or fps <= 0:
        print(f"Error: Could not determine FPS for '{edited_video_full_path}'. Defaulting to 25 fps, but this might be incorrect.")
        fps = 25 # Fallback FPS, adjust if necessary

    print(f"Using FPS: {fps} for encoding the comparison video.")

    # 准备拼接图像文件列表，并按时间戳排序
    # 我们依赖CSV中 'edited_frame_filename' 的顺序，这个顺序应该对应于剪辑视频的原始帧序。
    # CSV中的 'edited_timestamp_ms' (如果存在) 或从文件名中解析的时间戳可以用来验证顺序。
    
    stitched_image_paths_ordered = []
    for _, row in df_matches.iterrows():
        edited_frame_filename = row['edited_frame_filename']
        stitched_image_path = os.path.join(stitched_images_dir, edited_frame_filename)
        if os.path.exists(stitched_image_path):
            stitched_image_paths_ordered.append(stitched_image_path)
        else:
            print(f"Warning: Stitched image not found for {edited_frame_filename} at {stitched_image_path}. Skipping this frame.")

    if not stitched_image_paths_ordered:
        print("No stitched images found to create a video. Exiting.")
        return

    print(f"Found {len(stitched_image_paths_ordered)} stitched images to include in the video.")

    # 创建一个临时的文本文件，列出所有要编码的图像 (ffmpeg input file list)
    temp_image_list_file = os.path.join(abs_base_output_dir, "temp_image_list_for_ffmpeg.txt")
    with open(temp_image_list_file, 'w', encoding='utf-8') as f:
        for img_path in stitched_image_paths_ordered:
            # FFmpeg on Windows might need escaped backslashes or forward slashes
            # Using forward slashes is generally safer for cross-platform ffmpeg input files.
            f.write(f"file '{img_path.replace(os.sep, '/')}'\n")
            # duration 1/fps can also be added here per frame if needed,
            # but -r on input and output usually handles it.
            # f.write(f"duration {1/fps}\n") # Less common for image sequences

    # 步骤1: 从图像序列创建无声视频
    temp_video_output_path = os.path.join(abs_base_output_dir, f"{COMPARISON_VIDEO_OUTPUT_NAME}_no_audio.mp4")
    if os.path.exists(temp_video_output_path):
        os.remove(temp_video_output_path) # Remove if exists to avoid ffmpeg prompt

    # 获取第一张拼接图的分辨率，确保视频编码参数正确
    first_image = cv2.imread(stitched_image_paths_ordered[0])
    if first_image is None:
        print(f"Error: Could not read the first stitched image: {stitched_image_paths_ordered[0]}")
        if os.path.exists(temp_image_list_file): os.remove(temp_image_list_file)
        return
    height, width = first_image.shape[:2]
    # 确保宽度和高度是偶数，某些编码器需要
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1
    if width <= 0 or height <=0:
        print(f"Error: Invalid image dimensions after adjustment: {width}x{height}")
        if os.path.exists(temp_image_list_file): os.remove(temp_image_list_file)
        return


    ffmpeg_cmd_video = [
        FFMPEG_PATH,
        '-y',  # Overwrite output files without asking
        '-r', str(fps),  # Input frame rate (same as output)
        '-f', 'concat',  # Input format is a concatenation file
        '-safe', '0',    # Allow unsafe file paths in concat file (use with caution if paths are user-generated)
        '-i', temp_image_list_file,
        '-s', f'{width}x{height}', # Set video size from first image
        '-c:v', 'libx264',  # Video codec
        '-pix_fmt', 'yuv420p',  # Pixel format, widely compatible
        '-r', str(fps),  # Output frame rate
        '-crf', '23',    # Constant Rate Factor (quality, lower is better, 18-28 is common)
        '-preset', 'ultrafast', # Encoding speed/compression (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        temp_video_output_path
    ]
    print("\nCreating video from stitched images (no audio)...")
    _, _, video_retcode = run_command(ffmpeg_cmd_video)

    if os.path.exists(temp_image_list_file):
        os.remove(temp_image_list_file) # Clean up temp file

    if video_retcode != 0:
        print("Error creating video from images. Aborting audio merge.")
        return

    if not os.path.exists(temp_video_output_path) or os.path.getsize(temp_video_output_path) == 0:
        print(f"Error: Video without audio ({temp_video_output_path}) was not created or is empty.")
        return

    # 步骤2: 将原始剪辑视频的音频合并到新创建的视频中
    final_video_output_path = os.path.join(abs_base_output_dir, f"{COMPARISON_VIDEO_OUTPUT_NAME}.mp4")
    if os.path.exists(final_video_output_path):
        os.remove(final_video_output_path)

    ffmpeg_cmd_audio_merge = [
        FFMPEG_PATH,
        '-y',
        '-i', temp_video_output_path,  # Input: video without audio
        '-i', edited_video_full_path,   # Input: original edited video (for audio)
        '-c:v', 'copy',             # Copy video stream as is
        '-c:a', 'aac',              # Audio codec (or 'copy' if original audio is compatible and desired)
        '-b:a', '192k',             # Audio bitrate (if re-encoding)
        '-map', '0:v:0',            # Map video from first input
        '-map', '1:a:0?',            # Map audio from second input (if exists, '?' makes it optional)
        '-shortest',                # Finish encoding when the shortest input stream ends
        final_video_output_path
    ]
    print("\nMerging audio from original edited video...")
    _, _, merge_retcode = run_command(ffmpeg_cmd_audio_merge)

    if merge_retcode == 0 and os.path.exists(final_video_output_path) and os.path.getsize(final_video_output_path) > 0 :
        print(f"\nSuccessfully created comparison video with audio: {final_video_output_path}")
    else:
        print("\nError merging audio or final video not created.")
        print("The video without audio is available at:", temp_video_output_path)
        print("You might need to manually merge audio using ffmpeg or a video editor.")
        return # Keep the no_audio version if merge fails

    # 清理临时的无声视频
    if os.path.exists(temp_video_output_path):
        try:
            os.remove(temp_video_output_path)
            print(f"Cleaned up temporary file: {temp_video_output_path}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_video_output_path}: {e}")

    print("\nComparison video creation process finished.")

if __name__ == '__main__':
    try:
        # Need opencv for reading image dimensions
        import cv2
    except ImportError:
        print("OpenCV (cv2) is not installed. Please install it using: pip install opencv-python")
        exit()
        
    main()