# data_processing.py
import logging
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)


def generate_label_map():
    """生成BIOES标签映射"""
    entities = ['ZD', 'ZZ', 'ZF', 'ZP', 'ZS', 'ZA']
    prefixes = ['B', 'I', 'E', 'S']

    label_map = {'O': 0}
    for entity in entities:
        for prefix in prefixes:
            label_map[f"{prefix}-{entity}"] = len(label_map)

    return label_map


def read_file(file_path, max_seq_length=512):
    """读取并预处理数据"""
    sequences = []
    labels = []
    current_seq = []
    current_labels = []
    label_map = generate_label_map()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t') if '\t' in line else line.split()
                if len(parts) < 2:
                    continue
                token, tag = parts[0], parts[-1]
                tag_id = label_map.get(tag, 0)

                current_seq.append(token)
                current_labels.append(tag_id)

                if len(current_seq) >= max_seq_length - 2:
                    sequences.append(current_seq)
                    labels.append(current_labels)
                    current_seq = []
                    current_labels = []
            else:
                if current_seq:
                    sequences.append(current_seq)
                    labels.append(current_labels)
                    current_seq = []
                    current_labels = []

        if current_seq:
            sequences.append(current_seq)
            labels.append(current_labels)
    return sequences, labels


class NERDataset(Dataset):
    """适配CRF的数据集（填充标签为0）"""

    def __init__(self, sequences, labels, tokenizer, max_len=512):
        assert len(sequences) == len(labels), "数据标签数量不匹配"
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = generate_label_map()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length', # Keep padding for now, but we might test without later
            return_tensors='pt'
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # 修改：填充标签改为 -100
            else:
                assert word_id < len(labels), f"子词索引越界: word_id={word_id}, len(labels)={len(labels)}, idx={idx}"
                aligned_labels.append(labels[word_id])

        aligned_labels = aligned_labels[:self.max_len]
        aligned_labels += [-100] * (self.max_len - len(aligned_labels)) # 修改：填充标签改为 -100

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }


def prepare_datasets(data_path, tokenizer, test_size=0.2):
    """数据准备"""
    sequences, labels = read_file(data_path)
    train_seq, val_seq, train_lbl, val_lbl = train_test_split(
        sequences, labels, test_size=test_size, random_state=42
    )
    return (
        NERDataset(train_seq, train_lbl, tokenizer),
        NERDataset(val_seq, val_lbl, tokenizer),
    )