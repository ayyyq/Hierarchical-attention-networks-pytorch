"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul

class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size, word_hidden_size, num_classes):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(sent_hidden_size, word_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(1, sent_hidden_size))

        # self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        self.fc = nn.Linear(sent_hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.sent_weight)
        nn.init.constant_(self.sent_bias, 0.)
        nn.init.xavier_uniform_(self.context_weight)

    def forward(self, x, mask=None):
        # x: [batch, seq_num, hidden_size]

        # f_output, h_output = self.gru(input, hidden_state)
        attn = torch.tanh(F.linear(x, self.word_weight, self.word_bias))  # [batch, seq_num, hidden_size]
        attn = torch.matmul(self.context_weight, attn.transpose(-1, -2))  # [batch, 1, seq_num]
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = torch.matmul(attn, x)

        out = self.fc(attn)
        return out


if __name__ == "__main__":
    abc = SentAttNet()
