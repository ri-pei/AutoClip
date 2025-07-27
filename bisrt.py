import sys
import os
import re

def parse_srt(srt_path):
    """读取 srt 文件，按段落解析为 (index, time_range, content) 列表"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n{2,}', content.strip())
    parsed = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            index = lines[0]
            time_range = lines[1]
            text = '\n'.join(lines[2:])
            parsed.append((index, time_range, text))
    return parsed

def parse_txt(txt_path):
    """读取 txt 文件，多个换行统一视为分段"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 将连续两个及以上换行统一成一个
    normalized = re.sub(r'\n{2,}', '\n', content.strip())
    lines = normalized.split('\n')
    return [line.strip() for line in lines if line.strip()]

def merge_subtitles(srt_data, txt_data):
    """合并原字幕和补充文本为双语字幕，使用 \\n 连接为一行"""
    if len(srt_data) != len(txt_data):
        raise ValueError(f"SRT 段数 ({len(srt_data)}) 与 TXT 行数 ({len(txt_data)}) 不一致")
    merged = []
    for (index, time_range, orig_text), extra_text in zip(srt_data, txt_data):
        combined_text = f"{orig_text}\\n{extra_text}"  # 注意这里是两个反斜杠
        block = f"{index}\n{time_range}\n{combined_text}\n"
        merged.append(block)
    return '\n'.join(merged)

def main():
    if len(sys.argv) < 2:
        print("用法: python merge_bilingual_srt.py 原始字幕.srt [补充字幕.txt]")
        return
    
    srt_path = sys.argv[1]
    if not os.path.exists(srt_path):
        print(f"找不到 SRT 文件: {srt_path}")
        return

    if len(sys.argv) >= 3:
        txt_path = sys.argv[2]
    else:
        txt_path = os.path.splitext(srt_path)[0] + '.txt'

    if not os.path.exists(txt_path):
        print(f"找不到 TXT 文件: {txt_path}")
        return

    srt_data = parse_srt(srt_path)
    txt_data = parse_txt(txt_path)

    try:
        merged_content = merge_subtitles(srt_data, txt_data)
    except ValueError as e:
        print("错误：", e)
        return

    output_path = os.path.splitext(srt_path)[0] + '_bilingual.srt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(merged_content)

    print(f"合并完成，输出文件：{output_path}")

if __name__ == '__main__':
    main()