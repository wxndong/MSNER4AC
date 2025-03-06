import torch
import logging
from transformers import BertTokenizerFast
from bert_crf_model import BERT_CRF
from bert_crf_data_processing import generate_label_map

# 配置日志记录和设备
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model_path, tokenizer_path, test_path, output_path):
    """
    使用 BERT+CRF 模型进行预测。
    """
    # 加载模型和分词器
    try:
        model = BERT_CRF.from_pretrained(model_path)
        model.to(device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    try:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

    # 生成标签映射
    label_map = generate_label_map()
    id2label = {v: k for k, v in label_map.items()}  # 标签 ID 到标签名称的映射

    # 处理测试文件
    with open(test_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8', newline='\n') as f_out:

        for line_idx, line in enumerate(f_in):
            line = line.strip()

            # 如果是空行，直接写入一个空行到输出文件
            if not line:
                f_out.write("\n")
                continue

            # 字符级分割
            chars = list(line)
            total_length = len(chars)
            max_chunk_length = 510  # 512 - 2 ([CLS] 和 [SEP])
            all_labels = []

            logger.info(f"Processing line {line_idx + 1} with {total_length} characters")

            # 分块处理长文本
            for chunk_idx in range(0, total_length, max_chunk_length):
                chunk = chars[chunk_idx: chunk_idx + max_chunk_length]

                # 对分块进行分词
                inputs = tokenizer(
                    chunk,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt',
                    return_offsets_mapping=True
                )

                # 将输入移动到设备
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                # 模型预测
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask)

                # 提取预测标签
                batch_tags = outputs['predictions']
                if isinstance(batch_tags, torch.Tensor):
                    batch_tags = batch_tags.cpu().tolist()

                word_ids = inputs.word_ids(batch_index=0)
                valid_tags = []
                for word_id, tag_id in zip(word_ids, batch_tags[0]):
                    if word_id is not None:
                        valid_tags.append((word_id, tag_id))

                # 按 word ID 排序并去重
                valid_tags.sort(key=lambda x: x[0])
                chunk_labels = [id2label.get(tag, 'O') for _, tag in valid_tags]

                # 确保预测标签与字符长度一致
                try:
                    assert len(chunk_labels) == len(chunk)
                except AssertionError:
                    logger.error(f"Alignment error: Predicted tags ({len(chunk_labels)}) != Characters ({len(chunk)})")
                    chunk_labels = ['O'] * len(chunk)  # 如果对齐失败，回退到 'O'

                all_labels.extend(chunk_labels)

            # 最终对齐检查
            try:
                assert len(all_labels) == total_length
            except AssertionError:
                logger.error(f"Final alignment error: Labels ({len(all_labels)}) != Characters ({total_length})")
                all_labels = all_labels[:total_length]  # 截断以匹配字符数

            # 写入结果到输出文件
            for char, label in zip(chars, all_labels):
                f_out.write(f"{char}\t{label}\n")

            # 在每句话后添加一个空行
            f_out.write("\n")

            logger.info(f"Line {line_idx + 1} processed successfully.")


if __name__ == "__main__":
    # 配置路径
    md_dir = "../../../hy-tmp/models/Total_bertlr5e-5_crflr5e-3_cosine" # 修改输出目录
    model_path = md_dir
    tokenizer_path = md_dir
    test_file = "../datasets/test/raw/testset_C.txt"  # 测试文件路径 A | B | C
    output_file = "../results/ablation_CRF_result_C.txt"  # 输出文件路径 A | B | C

    predict(model_path, tokenizer_path, test_file, output_file)