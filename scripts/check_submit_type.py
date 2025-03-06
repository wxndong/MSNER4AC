"""
这是一个脚本
分析提交是否漏词
"""

import sys


def read_and_trim(filename):
    """读取文件并去除末尾连续空行"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]

    # 去除末尾连续空行
    end = len(lines)
    while end > 0 and lines[end - 1].strip() == '':
        end -= 1
    return lines[:end]


def compare_files(file1, file2):
    """核心比较函数，返回差异信息"""
    diffs = []
    line_num = 1

    # 并行遍历两个文件的每一行
    for l1, l2 in zip(file1, file2):
        # 空行状态判断
        is_empty1 = l1.strip() == ''
        is_empty2 = l2.strip() == ''

        # 空行状态检查
        if is_empty1 != is_empty2:
            diffs.append(('empty', line_num, is_empty1, is_empty2))
        # 非空行内容检查
        elif not is_empty1:
            col1 = l1.split('\t', 1)[0]  # 文件1的第一列
            col2 = l2  # 文件2的整行内容
            if col1 != col2:
                diffs.append(('content', line_num, col1, col2))

        line_num += 1

    return diffs


def main():
    if len(sys.argv) != 3:
        print("使用方法: python check_submit_type.py <带标签文件> <纯文本文件>")
        return

    try:
        file1 = read_and_trim(sys.argv[1])
        file2 = read_and_trim(sys.argv[2])
    except FileNotFoundError as e:
        print(f"错误：文件不存在 - {e.filename}")
        return

    # 初步行数检查
    if len(file1) != len(file2):
        print(f"❌ 行数不一致 (去除末尾空行后：文件1={len(file1)}行，文件2={len(file2)}行)")
        return

    # 详细比对
    diffs = compare_files(file1, file2)

    if not diffs:
        print("✅ 文件内容完全一致")
        return

    print("❌ 发现不一致：")

    # 分类统计差异
    empty_diffs = [d for d in diffs if d[0] == 'empty']
    content_diffs = [d for d in diffs if d[0] == 'content']

    # 输出空行差异
    if empty_diffs:
        print(f"\n[空行差异] 共{len(empty_diffs)}处：")
        for d in empty_diffs[:3]:
            _, line, is_e1, is_e2 = d
            status1 = "有空行" if is_e1 else "无空行"
            status2 = "有空行" if is_e2 else "无空行"
            print(f"  第{line}行：文件1 {status1.ljust(8)} vs 文件2 {status2}")

    # 输出内容差异
    if content_diffs:
        print(f"\n[内容差异] 共{len(content_diffs)}处：")
        for d in content_diffs[:3]:
            _, line, col1, col2 = d
            print(f"  第{line}行：文件1 → '{col1}' | 文件2 → '{col2}'")

    # 显示剩余差异数量
    remaining = len(diffs) - 3
    if remaining > 0:
        print(f"\n...剩余{remaining}处差异未显示")


if __name__ == "__main__":
    main()