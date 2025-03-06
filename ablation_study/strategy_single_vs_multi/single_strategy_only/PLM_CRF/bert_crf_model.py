# model.py
from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
from torchcrf import CRF


class BERT_CRF(BertPreTrainedModel):
    """BERT+CRF模型"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        # 网络结构
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
        # 初始化
        self.init_weights()


    def forward(self, input_ids, attention_mask=None, labels=None):
        # BERT编码 (后续的 BERT 层 *需要* attention_mask 参数)
        outputs = self.bert(input_ids, attention_mask=attention_mask)  # BERT 模型整体 forward  *需要* attention_mask
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        # CRF处理
        mask = attention_mask.bool() if attention_mask is not None else None
        tags = self.crf.decode(emissions, mask=mask)

        loss = None
        if labels is not None:
            # 确保labels是LongTensor类型
            labels = labels.long()
            # print(labels)
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')

        return {
            'loss': loss,
            'predictions': tags,
            'attention_mask': mask,
            'logits': emissions  # 将 emissions 添加到返回字典，键名为 'logits'
        }
