import os
import pandas as pd
import json

# --- 用户配置 ---
# 工作目录，同之前的脚本
WORKING_DIR = "."

# 基础输出目录名
OUTPUT_DIR_NAME = "output"

# 输入的粗略匹配结果CSV文件名 (位于 WORKING_DIR/OUTPUT_DIR_NAME/ 下)
INPUT_CSV_FILENAME = "coarse_match_results2.csv"

# 输出的格式化后的CSV文件名 (将保存在 WORKING_DIR/OUTPUT_DIR_NAME/ 下)
OUTPUT_FORMATTED_CSV_FILENAME = "coarse_match_results_readable2.csv"

# Top N 的值，需要与生成原始CSV时使用的 TOP_N_CANDIDATES 一致
# 这个值决定了要为每个候选者创建多少新列
# 例如，如果 TOP_N_CANDIDATES 是 5, 脚本会尝试为最多5个候选者创建列
TOP_N_VALUE = 5 # 请确保这个值与步骤3中 TOP_N_CANDIDATES 的设置一致
# --- END 用户配置 ---

def format_timestamp_from_milliseconds(ms):
    """将毫秒数格式化为 HH-MM-SS-mmm"""
    if pd.isna(ms) or ms < 0:
        return "" # 或者返回 None, "N/A" 等
    
    total_seconds = ms / 1000
    hours = int(total_seconds / 3600)
    minutes = int((total_seconds % 3600) / 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(ms % 1000)
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}-{milliseconds:03d}"

def main():
    abs_working_dir = os.path.abspath(WORKING_DIR)
    abs_base_output_dir = os.path.join(abs_working_dir, OUTPUT_DIR_NAME)

    input_csv_path = os.path.join(abs_base_output_dir, INPUT_CSV_FILENAME)
    output_csv_path = os.path.join(abs_base_output_dir, OUTPUT_FORMATTED_CSV_FILENAME)

    if not os.path.exists(input_csv_path):
        print(f"错误: 输入的CSV文件未找到: {input_csv_path}")
        return

    print(f"正在读取CSV文件: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"读取CSV文件时出错 {input_csv_path}: {e}")
        return

    if 'top_n_matches' not in df.columns:
        print(f"错误: 输入的CSV文件中缺少 'top_n_matches' 列。")
        return

    print("正在处理 'top_n_matches' 列...")
    
    # 用于存储新数据的列表
    all_rows_data = []

    for index, row in df.iterrows():
        new_row = row.to_dict() # 复制原始行数据
        
        try:
            top_n_matches_json_str = new_row.pop('top_n_matches') # 从新行中移除原始列
            if pd.isna(top_n_matches_json_str):
                candidate_matches = []
            else:
                candidate_matches = json.loads(top_n_matches_json_str)
        except json.JSONDecodeError:
            print(f"警告: 解析第 {index+2} 行的 'top_n_matches' JSON时出错。此行的候选匹配将为空。内容: {top_n_matches_json_str[:100]}...")
            candidate_matches = []
        except Exception as e:
            print(f"警告: 处理第 {index+2} 行的 'top_n_matches' 时发生未知错误: {e}。此行的候选匹配将为空。")
            candidate_matches = []


        # 为每个Top N候选创建新列
        for i in range(TOP_N_VALUE):
            col_prefix = f"match_{i+1}"
            timestamp_col_name = f"{col_prefix}_orig_time"
            distance_col_name = f"{col_prefix}_phash_dist"
            
            if i < len(candidate_matches):
                match_info = candidate_matches[i]
                original_ts_ms = match_info.get('original_timestamp_ms')
                phash_dist = match_info.get('phash_distance')
                
                new_row[timestamp_col_name] = format_timestamp_from_milliseconds(original_ts_ms)
                new_row[distance_col_name] = phash_dist if pd.notna(phash_dist) else ""
            else:
                # 如果候选者数量少于 TOP_N_VALUE，则用空值填充
                new_row[timestamp_col_name] = ""
                new_row[distance_col_name] = ""
        
        all_rows_data.append(new_row)

    if not all_rows_data:
        print("没有数据可处理或写入。")
        return

    # 创建新的DataFrame
    df_formatted = pd.DataFrame(all_rows_data)
    
    # 确保列的顺序（可选，但有助于可读性）
    # 先获取原始列（除了 'top_n_matches'）
    original_cols = [col for col in df.columns if col != 'top_n_matches']
    new_match_cols = []
    for i in range(TOP_N_VALUE):
        new_match_cols.append(f"match_{i+1}_orig_time")
        new_match_cols.append(f"match_{i+1}_phash_dist")
    
    # 确保所有预期的新列都存在于df_formatted中，以防万一
    final_columns_order = original_cols + [col for col in new_match_cols if col in df_formatted.columns]
    # 如果df_formatted中由于某种原因没有生成所有预期的列，只保留实际存在的列
    df_formatted = df_formatted.reindex(columns=final_columns_order)


    print(f"正在将格式化后的数据写入: {output_csv_path}")
    try:
        df_formatted.to_csv(output_csv_path, index=False, lineterminator='\n')
        print("成功生成可读格式的CSV文件。")
    except Exception as e:
        print(f"写入格式化后的CSV文件时出错 {output_csv_path}: {e}")

if __name__ == '__main__':
    main()