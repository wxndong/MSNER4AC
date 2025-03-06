import argparse
from collections import defaultdict

def count_entities(filename):
    counts = defaultdict(int)
    current_entity = None  # 跟踪当前未闭合的实体类型
    entity_types = set()  # 存储所有遇到的实体类型

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            tag = parts[1]

            # 处理O标签
            if tag == "O":
                if current_entity is not None:
                    counts[current_entity] += 1
                    current_entity = None
                continue

            # 分解标签前缀和实体类型
            if '-' in tag:
                prefix, entity_type = tag.split('-', 1)
            else:
                continue  # 跳过非法格式

            # 动态识别新的实体类型
            if entity_type not in entity_types:
                entity_types.add(entity_type)

            # BIOES逻辑处理
            if prefix == "B":
                if current_entity is not None:  # 前一个实体未闭合
                    counts[current_entity] += 1
                current_entity = entity_type
            elif prefix == "S":
                counts[entity_type] += 1
                if current_entity is not None:  # 处理冲突情况
                    counts[current_entity] += 1
                    current_entity = None
            elif prefix == "E":
                if current_entity == entity_type:
                    counts[entity_type] += 1
                    current_entity = None
                else:
                    pass  # 非法情况跳过
            elif prefix == "I":
                pass  # 仅延续实体，不做操作

        # 处理文件末尾未闭合的实体
        if current_entity is not None:
            counts[current_entity] += 1

    return counts, entity_types

def main():
    parser = argparse.ArgumentParser(description='统计BIOES格式数据集实体类型数量')
    parser.add_argument('input_file', help='输入文件路径')
    args = parser.parse_args()

    entity_counts, entity_types = count_entities(args.input_file)

    # 输出结果按字母排序
    print(f"{'Entity':<10}Count")
    print('-' * 15)
    for entity, count in sorted(entity_counts.items(), key=lambda x: x[0]):
        print(f"{entity:<10}{count}")

    # 输出所有遇到的实体类型
    print("\nEncountered Entity Types:")
    print(", ".join(sorted(entity_types)))

if __name__ == "__main__":
    main()