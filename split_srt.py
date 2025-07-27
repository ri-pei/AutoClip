import sys
import os
import re

def parse_srt_blocks(content):
    blocks = re.split(r'\n{2,}', content.strip())
    parsed = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            index = lines[0]
            timestamp = lines[1]
            text_lines = lines[2:]
            parsed.append((index, timestamp, text_lines))
    return parsed

def write_srt_file(filename, blocks):
    with open(filename, 'w', encoding='utf-8') as f:
        for i, (index, timestamp, text) in enumerate(blocks, 1):
            f.write(f"{i}\n{timestamp}\n{text}\n\n")

def split_bilingual_srt(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = parse_srt_blocks(content)
    zh_blocks = []
    jp_blocks = []

    for index, timestamp, text_lines in blocks:
        if len(text_lines) >= 2:
            zh = text_lines[0].strip()
            jp = text_lines[1].strip()
        elif len(text_lines) == 1:
            zh = text_lines[0].strip()
            jp = ''
        else:
            zh = jp = ''
        zh_blocks.append((index, timestamp, zh))
        jp_blocks.append((index, timestamp, jp))

    base, _ = os.path.splitext(srt_path)
    zh_path = base + "_zh.srt"
    jp_path = base + "_jp.srt"

    write_srt_file(zh_path, zh_blocks)
    write_srt_file(jp_path, jp_blocks)

    print(f"生成：\n  {zh_path}\n  {jp_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python split_bilingual_srt.py 文件名.srt")
        sys.exit(1)
    srt_file = sys.argv[1]
    split_bilingual_srt(srt_file)	
