# model.py
from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn

class BERT_Softmax(BertPreTrainedModel):
    """BERT+Softmax模型"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        # 网络结构
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 初始化
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 显式设置 ignore_index 为 -100
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 处理padding部分的损失忽略
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

        return {
            'loss': loss,
            'logits': logits,
            'attention_mask': attention_mask
        }