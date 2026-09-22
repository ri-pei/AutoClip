# main.py

import argparse
import time
import traceback

# 导入每个步骤的模块
import step1
import step2
import step3
import step4
import step5
from common import check_dependencies
from settings import load_settings


def run_step(step_function, step_name, settings):
    """
    一个辅助函数，用于执行单个步骤，打印状态并捕获错误。
    :param step_function: 要调用的步骤主函数 (例如, step1.main_step1)
    :param step_name: 步骤的名称 (例如, "STEP 1: Frame Extraction")
    :return: 如果成功则返回 True, 失败则返回 False
    """
    print(f"\n{'=' * 25} RUNNING {step_name.upper()} {'=' * 25}")
    start_time = time.time()
    try:
        step_function(settings)
        end_time = time.time()
        duration = end_time - start_time
        print(f"--- {step_name} COMPLETED SUCCESSFULLY in {duration:.2f} seconds ---")
        return True
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n!!! CRITICAL ERROR IN {step_name} after {duration:.2f} seconds !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("--- End Traceback ---\n")
        print("The process cannot continue due to the error above.")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="从剪辑 MV 反向匹配 source 视频并生成 FCPXML。"
    )
    parser.add_argument(
        "command", choices=("all", "step1", "step2", "step3", "step4", "step5")
    )
    parser.add_argument("--config", default="config.toml", help="TOML 配置文件")
    return parser.parse_args()


def main():
    """
    项目的主执行函数。
    """
    print("==========================================================")
    print("===      Automatic Video Clip Finder and Re-editor     ===")
    print("==========================================================")

    args = parse_args()
    settings = load_settings(args.config)
    print("正在检查项目依赖...")
    check_dependencies()
    print("依赖检查通过。")

    overall_start_time = time.time()
    steps = [
        ("step1", step1.main_step1, "STEP 1: Frame Extraction and Pre-processing"),
        ("step2", step2.main_step2, "STEP 2: pHash Calculation for all frames"),
        ("step3", step3.main_step3, "STEP 3: Batched pHash Matching"),
        ("step4", step4.main_step4, "STEP 4: Segment Refinement and Finalization"),
        ("step5", step5.csv_to_fcpxml, "STEP 5: FCPXML Generation for Editing Software"),
    ]
    selected_steps = steps if args.command == "all" else [item for item in steps if item[0] == args.command]
    for _, step_function, step_name in selected_steps:
        if not run_step(step_function, step_name, settings):
            raise SystemExit(1)

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time

    print("\n==========================================================")
    if args.command == "all":
        print("===            ALL STEPS COMPLETED! 🎉🎉🎉           ===")
    else:
        print(f"===                 {args.command.upper()} COMPLETED                 ===")
    print("==========================================================")
    print(f"Total execution time: {total_duration:.2f} seconds.")
    if args.command in ("all", "step5"):
        print("The final .fcpxml file has been generated in your output directory.")
        print("You can now import it into DaVinci Resolve or Final Cut Pro.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"错误: {error}")
        raise SystemExit(1) from error
