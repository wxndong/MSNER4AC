import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_ner_dataset(file_path):
    """BIOES格式的古汉语NER数据集分析工具"""
    try:
        # ================== 数据读取与预处理 ==================
        sentences = []
        current_sentence = []
        line_counter = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                line_counter += 1
                
                # 处理空行（句子边界）
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue
                
                # 格式校验
                if '\t' not in line:
                    print(f"警告：第 {line_num} 行格式错误（缺少制表符），已跳过")
                    continue
                
                char, tag = line.split('\t', 1)
                current_sentence.append(tag)
            
            # 处理最后一个句子
            if current_sentence:
                sentences.append(current_sentence)

        # ================== 基础统计 ==================
        tag_counts = defaultdict(int)
        category_counts = defaultdict(int)
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        # ================== 实体提取与统计 ==================
        def extract_entities(sentence):
            """精确提取BIOES格式的实体及其位置"""
            entities = []
            current_entity = []
            expected_type = None
            
            for idx, tag in enumerate(sentence):
                prefix = tag.split('-')[0] if '-' in tag else 'O'
                entity_type = tag.split('-')[1] if '-' in tag else None
                
                # 实体开始
                if prefix in ['B', 'S']:
                    if current_entity:
                        entities.append(current_entity)  # 保存未闭合实体
                    current_entity = [{
                        'start': idx,
                        'end': idx,
                        'type': entity_type,
                        'tags': [tag]
                    }]
                    if prefix == 'S':
                        entities.append(current_entity)
                        current_entity = []
                
                # 实体延续
                elif prefix == 'I':
                    if current_entity and entity_type == expected_type:
                        current_entity[-1]['end'] = idx
                        current_entity[-1]['tags'].append(tag)
                    else:
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = []  # 非法情况重置
                
                # 实体结束
                elif prefix == 'E':
                    if current_entity and entity_type == expected_type:
                        current_entity[-1]['end'] = idx
                        current_entity[-1]['tags'].append(tag)
                        entities.append(current_entity)
                        current_entity = []
                    else:
                        if current_entity:
                            entities.append(current_entity)
                        current_entity = []
                
                # 其他情况处理
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = []
                
                expected_type = entity_type if prefix in ['B', 'I'] else None
            
            # 处理未闭合实体
            if current_entity:
                entities.append(current_entity)
            
            return [ent[0] for ent in entities if ent]  # 展平嵌套列表

        # ================== 统计计算 ==================
        entity_stats = defaultdict(lambda: {
            'count': 0,
            'lengths': [],
            'positions': []
        })
        
        for sent_idx, sentence in enumerate(sentences):
            # 基础统计
            prev_tag = None
            for tag in sentence:
                tag_counts[tag] += 1
                
                # 记录类别统计
                if tag != 'O':
                    category = tag.split('-')[1]
                    category_counts[category] += 1
                
                # 转移统计
                if prev_tag is not None:
                    transition_counts[prev_tag][tag] += 1
                prev_tag = tag
            
            # 实体统计
            entities = extract_entities(sentence)
            for ent in entities:
                ent_type = ent['type']
                length = ent['end'] - ent['start'] + 1
                entity_stats[ent_type]['count'] += 1
                entity_stats[ent_type]['lengths'].append(length)
                entity_stats[ent_type]['positions'].append(
                    (sent_idx, ent['start'], ent['end'])
                )

        # ================== 分析报告生成 ==================
        def generate_length_report(stats):
            """生成实体长度分析报告"""
            report = []
            for ent_type in ['NR', 'NS', 'T']:
                if ent_type not in stats:
                    continue
                data = stats[ent_type]
                lengths = data['lengths']
                if not lengths:
                    continue
                
                avg_len = sum(lengths) / len(lengths)
                max_len = max(lengths)
                length_dist = pd.Series(lengths).value_counts().sort_index()
                mode_len = length_dist.idxmax()
                mode_percent = length_dist.max() / len(lengths)
                
                report.append({
                    '实体类型': ent_type,
                    '总数': len(lengths),
                    '平均长度': f"{avg_len:.2f}",
                    '最大长度': max_len,
                    '最常见长度': f"{mode_len} ({mode_percent:.1%})",
                    '长度分布': dict(length_dist.items())
                })
            return pd.DataFrame(report)

        # ================== 输出结果 ==================
        print(f"\n{'='*40} 数据集分析报告 {'='*40}")
        print(f"文件路径: {os.path.abspath(file_path)}")
        print(f"总句子数: {len(sentences)}")
        print(f"总标签数: {sum(tag_counts.values())}")
        
        # 标签分布
        print("\n【标签分布】")
        print(pd.Series(tag_counts).sort_index().to_string())
        
        # 实体统计
        print("\n【实体统计】")
        df_entities = pd.DataFrame({
            '数量': [entity_stats[t].get('count',0) for t in ['NR','NS','T']],
            '平均长度': [np.mean(entity_stats[t].get('lengths',[0])) for t in ['NR','NS','T']]
        }, index=['NR', 'NS', 'T'])
        print(df_entities)
        
        # 长度分析
        print("\n【实体长度分布】")
        length_report = generate_length_report(entity_stats)
        if not length_report.empty:
            print(length_report.set_index('实体类型'))
        else:
            print("未检测到有效实体")
        
        # 转移分析
        print("\n【前5常见转移模式】")
        flat_trans = []
        for from_tag, to_tags in transition_counts.items():
            for to_tag, count in to_tags.items():
                flat_trans.append( (f"{from_tag}→{to_tag}", count) )
        print(pd.Series(dict(sorted(flat_trans, key=lambda x: -x[1])[:5])).to_string())
        
        # 非法转移检测（示例）
        print("\n【潜在非法转移检测】")
        illegal_pairs = [
            ('O', 'I-NR'), ('E-NR', 'I-NS'), 
            ('B-NR', 'I-NS'), ('I-NR', 'B-NS')
        ]
        found = []
        for pair in illegal_pairs:
            count = transition_counts.get(pair[0], {}).get(pair[1], 0)
            if count > 0:
                found.append( (f"{pair[0]}→{pair[1]}", count) )
        if found:
            print("检测到可能非法的跨实体转移:")
            print(pd.Series(dict(found)).to_string())
        else:
            print("未发现明显非法转移")

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
    except Exception as e:
        print(f"分析失败：{str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("请输入BIOES格式的数据文件路径：")
    
    analyze_ner_dataset(input_file)