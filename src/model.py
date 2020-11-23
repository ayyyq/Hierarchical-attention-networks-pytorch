import torch
import torch.nn as nn
from transformers import BertModel
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet


class BertHierAttNet(nn.Module):
    def __init__(self, num_classes, bert_path, bert_size=768, word_hidden_size=100, sent_hidden_size=100):
        super(BertHierAttNet, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.linear = nn.Linear(bert_size, word_hidden_size)
        self.word_att_net = WordAttNet(word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        # self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, x, mask):
        # x: [batch, seq_num, seq_len]
        # mask: [batch, seq_num, seq_len]
        # batch is possible to be 1
        batch, seq_num, seq_len = x.shape
        x = x.view(-1, seq_len)
        x = self.bert(x, attention_mask=mask.view(-1, seq_len))[1]  # [batch * seq_num, bert_size]
        x = self.linear(x).view(batch, seq_num, -1)  # [batch, seq_num, word_hidden_size]
        out = self.sent_att_net(x, mask[:, :, 0])
        return out
