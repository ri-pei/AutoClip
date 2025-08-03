# --- 用户核心配置参数 ---
# 参考每个变量注释，所有相对路径的配置项会转换为基于WORKING_DIR的绝对路径

# 若为空则使用当前目录，否则按照设置的完整路径
WORKING_DIR = "."

# 若只是文件名则使用工作目录下的这个文件，若是路径则使用该路径
EDITED_VIDEO_FILENAME = "my_edited_clip.mp4"

# 输出目录，若只是文件夹名则是工作目录下的该文件夹，若是路径则使用该路径
OUTPUT_DIR = "output"

# 源视频文件夹，若只是文件夹名则是工作目录下的该文件夹，若是路径则使用该路径
# 该文件夹下所有子文件夹中的所有视频文件将被处理。
# 注意：如果是单个视频文件，可以放在工作目录下，并设置SOURCE_VIDEO_FOLDER为"."
# 如果是多个视频文件，请确保它们都在这个文件夹中。
SOURCE_VIDEO_FOLDER = "source"


# --- Step 1 相关设置 ---
# 遮罩矩形配置，格式为 (x, y, w, h)，如果不需要遮罩则设为None
# 例如 (0, 1080-80, 1920, 80) or None
MASK_RECT = (
    1085,
    36,
    1819 - 1085,
    152 - 36,
)

# 用户提供的参考帧路径必须提供这两个文件，置于工作目录下。
# PNG格式的单帧图像，内容必须是同一时刻的，用于计算变换。
REF_ORIGINAL_FRAME_PATH = "ref_original.png"
REF_EDITED_FRAME_PATH = "ref_edited.png"

# 可选的LUT文件路径。 ！该功能尚未实现
# 如果用户有一个.cube文件用于颜色校正，请指定路径，否则设为None
USER_COLOR_LUT_PATH = None  # 例如: "my_color_correction.cube"


# --- Step 3 相关设置 ---
COARSE_MATCH_CSV_FILENAME = "coarse_match_results2.csv"
IF_RECONSTRUCTED_VIDEO = True  # 是否生成重建视频

IF_STITCHED_IMAGES = False  # 是否生成拼接图像。速度慢，Debug用，更推荐直接重建视频
COARSE_MATCH_OUTPUT_SUBDIR_NAME = "coarse_match_visuals2"  # stitched images文件夹


# --- Step 4 相关设置 ---
FINAL_SEGMENTS_CSV_FILENAME = "final_video_segments_refined.csv"


# --- Step 5 相关设置 ---
FRAME_RATE_FLOAT = 23.976  # 假设视频的平均帧率为23.976
FCPXML_EVENT_NAME = "Final Video Segments"
FCPXML_PROJECT_NAME = "My Project"  # 同时也是最终文件名


# --- END 用户核心配置参数 ---
