def remove_empty_lines(input_path, output_path):
    """高效去除文件中的空行（包含仅空白字符的行）
    
    参数：
        input_path (str): 输入文件路径
        output_path (str): 输出文件路径
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            # 保留非空行（包含可见内容的行）
            if line.strip():
                outfile.write(line)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="文件空行清理工具")
    parser.add_argument('input', help='输入文件路径')
    parser.add_argument('output', help='输出文件路径')
    args = parser.parse_args()
    
    remove_empty_lines(args.input, args.output)