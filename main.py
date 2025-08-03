# main.py

import sys
import time
import traceback

# 导入每个步骤的模块
import step1
import step2
import step3
import step4
import step5


def run_step(step_function, step_name):
    """
    一个辅助函数，用于执行单个步骤，打印状态并捕获错误。
    :param step_function: 要调用的步骤主函数 (例如, step1.main_step1)
    :param step_name: 步骤的名称 (例如, "STEP 1: Frame Extraction")
    :return: 如果成功则返回 True, 失败则返回 False
    """
    print(f"\n{'=' * 25} RUNNING {step_name.upper()} {'=' * 25}")
    start_time = time.time()
    try:
        step_function()
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


def main():
    """
    项目的主执行函数。
    """
    print("==========================================================")
    print("===      Automatic Video Clip Finder and Re-editor     ===")
    print("==========================================================")

    overall_start_time = time.time()

    # 依次执行所有步骤
    if not run_step(step1.main_step1, "STEP 1: Frame Extraction and Pre-processing"):
        sys.exit(1)  # 如果步骤1失败，则退出

    if not run_step(step2.main_step2, "STEP 2: pHash Calculation for all frames"):
        sys.exit(1)

    if not run_step(step3.main_step3, "STEP 3: Coarse Matching with BallTree"):
        sys.exit(1)

    if not run_step(step4.main_step4, "STEP 4: Segment Refinement and Finalization"):
        sys.exit(1)

    if not run_step(
        step5.csv_to_fcpxml, "STEP 5: FCPXML Generation for Editing Software"
    ):
        sys.exit(1)

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time

    print("\n==========================================================")
    print("===            ALL STEPS COMPLETED! 🎉🎉🎉           ===")
    print("==========================================================")
    print(f"Total execution time: {total_duration:.2f} seconds.")
    print("The final .fcpxml file has been generated in your output directory.")
    print("You can now import it into DaVinci Resolve or Final Cut Pro.")


if __name__ == "__main__":
    main()
