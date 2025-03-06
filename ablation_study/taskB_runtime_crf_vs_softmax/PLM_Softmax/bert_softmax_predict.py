import torch
import logging
from transformers import BertTokenizerFast
from bert_softmax_model import BERT_Softmax
from bert_softmax_data_processing import generate_label_map
import time  # 添加 time 模块用于计时

# 配置日志记录和设备
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model_path, tokenizer_path, test_path, output_path):
    """使用 BERT+Softmax 模型进行预测"""
    # 加载模型和分词器
    try:
        model = BERT_Softmax.from_pretrained(model_path).to(device)
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise

    # 标签映射配置
    label_map = generate_label_map()
    id2label = {v: k for k, v in label_map.items()}

    # 记录预测开始时间
    print("Starting prediction...")
    start_time = time.time()

    with open(test_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8', newline='\n') as f_out:

        for line_idx, line in enumerate(f_in):
            line = line.strip()
            if not line:
                f_out.write("\n")
                continue

            # 字符级处理流程
            chars = list(line)
            all_labels = []
            max_chunk_length = 510  # 与CRF版本保持一致的切分策略

            # 分块预测逻辑
            for chunk_idx in range(0, len(chars), max_chunk_length):
                chunk = chars[chunk_idx:chunk_idx + max_chunk_length]

                # 与CRF版本保持一致的tokenization流程
                inputs = tokenizer(
                    chunk,
                    is_split_into_words=True,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt',
                    return_token_type_ids=False
                ).to(device)

                # 提取模型需要的输入，避免传递 offset_mapping 等无关参数
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                # 预测逻辑调整
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)  # Get the full output
                    logits = outputs['logits']  # Access logits from the dictionary
                predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

                # 与CRF版本一致的标签对齐策略
                word_ids = inputs.word_ids(batch_index=0)
                valid_tags = [(wid, tag) for wid, tag in zip(word_ids, predictions) if wid is not None]

                # 去重策略保持与CRF版本一致
                seen_wids = set()
                chunk_labels = []
                for wid, tag in sorted(valid_tags, key=lambda x: x[0]):
                    if wid not in seen_wids:
                        seen_wids.add(wid)
                        chunk_labels.append(id2label.get(tag, 'O'))

                # 严格长度校验
                if len(chunk_labels) != len(chunk):
                    logger.warning(f"对齐异常: 预测标签数({len(chunk_labels)}) != 字符数({len(chunk)})")
                    chunk_labels = ['O'] * len(chunk)

                all_labels.extend(chunk_labels)

            # 最终写入格式与CRF版本完全一致
            for char, label in zip(chars, all_labels[:len(chars)]):
                f_out.write(f"{char}\t{label}\n")
            f_out.write("\n")

    # 记录预测结束时间并计算耗时
    end_time = time.time()
    prediction_time = end_time - start_time  # 计算总预测时间（秒）

    # 转换为小时、分钟、秒格式
    hours = int(prediction_time // 3600)
    minutes = int((prediction_time % 3600) // 60)
    seconds = prediction_time % 60
    print(f"Prediction completed in: {hours} hours, {minutes} minutes, {seconds:.2f} seconds (Total: {prediction_time:.2f} seconds)")


if __name__ == "__main__":
    # 配置路径（保持与CRF版本相同的参数结构）
    config = {
        "md_dir": "../../../hy-tmp/models/Ablation_bert_softmax_lr5e-5_linear",
        "test_file": "../datasets/test/raw/testset_B.txt",
        "output_file": "../results/ablation_Softmax_result_B.txt"
    }
    predict(config["md_dir"], config["md_dir"], config["test_file"], config["output_file"])