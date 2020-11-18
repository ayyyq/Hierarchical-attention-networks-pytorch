"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer
import xml.etree.ElementTree as ET

from src.utils import k_fold_split


class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, index, label2idx, max_seq_num=40, max_seq_len=150):
        super(MyDataset, self).__init__()
        self.texts, self.labels, self.ids = self.process_xml(data_path, dict_path, index, label2idx)
        self.max_seq_num = max_seq_num
        self.max_seq_len = max_seq_len
        self.num_classes = len(label2idx)

    def process_xml(self, data_path, dict_path, index, label2idx):
        # 解析xml文件，并tokenize
        tok = BertTokenizer.from_pretrained(dict_path)

        docs = []  # [doc_num, seq_len], list of list
        texts = []  # [doc_num, seq_num, seq_len], list of list of list
        labels = []  # [doc_num], list
        ids = []  # [doc_num], list

        tree = ET.parse(data_path)
        root = tree.getroot()
        for document_set in root:
            for i in index:
                document = document_set[i]
                # id
                ids.append(document.attrib['id'])

                # label
                label = document.attrib['document_level_value']
                labels.append(label2idx[label])

                # text
                for sentence in document:
                    if sentence.text == '-EOP- .':
                        continue
                    sent = ''
                    for text in sentence.itertext():
                        sent += text
                    sent = sent.replace('-EOP- ', '').lower()

                    # tokenize
                    tok_sent = tok.encode(sent)
                    docs.append(tok_sent)

                texts.append(docs)
                docs = []
        return texts, labels, ids

    def get_label_size(self):
        return

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]  # scalar
        doc = self.texts[index]  # doc, list of list
        id = self.ids[index]
        mask = []

        for sent in doc:
            tmp_mask = [1] * len(sent) + [0] * (self.max_seq_len - len(sent))
            mask.append(tmp_mask)
            if len(sent) < self.max_seq_len:
                sent += [0] * (self.max_seq_len - len(sent))
        if len(doc) < self.max_seq_num:
            mask += [[0] * self.max_seq_len] * (self.max_seq_num - len(doc))
            doc += [[0] * self.max_seq_len] * (self.max_seq_num - len(doc))
        elif len(doc) > self.max_seq_num:
            mask = mask[:self.max_seq_num]
            doc = doc[:self.max_seq_num]

        return torch.tensor(doc), torch.tensor(mask), torch.tensor(label, dtype=torch.long), id


if __name__ == '__main__':
    train_idx, test_idx, label2idx = k_fold_split("../data/dlef_corpus/english.xml")

    test = MyDataset(data_path="../data/dlef_corpus/english.xml", dict_path="../data/bert-base-uncased",
                     index=train_idx[0], label2idx=label2idx)
    doc, masks, label, id = test.__getitem__(index=0)
    print("end")
