"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul
import pandas as pd
import numpy as np
import csv

class WordAttNet(nn.Module):
    def __init__(self, hidden_size):
        super(WordAttNet, self).__init__()
        # self.word_weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.word_bias = nn.Parameter(torch.Tensor(hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(1, hidden_size))

        # self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        # self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self._reset_parameters()

    def _reset_parameters(self):
        # nn.init.xavier_uniform_(self.word_weight)
        # nn.init.constant_(self.word_bias, 0.)
        nn.init.xavier_uniform_(self.context_weight)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, hidden_size]
        # mask: [batch, seq_len]

        # output = self.lookup(input)
        # f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        attn = torch.tanh(x)  # [batch, seq_len, hidden_size]
        attn = torch.matmul(self.context_weight, attn.transpose(-1, -2))  # [batch, 1, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = torch.tanh(torch.matmul(attn, x)).squeeze()

        return attn  # [batch, hidden_size]


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
