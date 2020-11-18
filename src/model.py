import torch
import torch.nn as nn
from transformers import BertModel
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet


class BertHierAttNet(nn.Module):
    def __init__(self, bert_size, word_hidden_size, sent_hidden_size, num_classes, bert_path):
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
        seq_list = []
        x = x.transpose(0, 1)  # [seq_num, batch, seq_len]
        mask = mask.transpose(0, 1)  # [seq_num, batch, seq_len]
        for (s, m) in zip(x, mask):
            # s: [batch, seq_len]
            # m: [batch, seq_len]
            s = self.bert(s, attention_mask=m)[0]  # [batch, seq_len, bert_size]
            s = self.linear(s)  # [batch, seq_len, word_hidden_size]
            s = self.word_att_net(s, m)
            seq_list.append(s)
        s = torch.cat(seq_list, dim=0)  # [seq_num, batch, word_hidden_size]
        s = s.transpose(0, 1)  # [batch, seq_num, word_hidden_size]
        mask = mask[:, :, 0].transpose(0, 1)  # [batch, seq_num]
        out = self.sent_att_net(s, mask)

        return out
