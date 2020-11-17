"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer
import xml.etree.ElementTree as ET


class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, max_seq_num=40):
        super(MyDataset, self).__init__()
        self.texts, self.labels, self.num_classes, self.max_seq_num, self.max_seq_len = self.process_xml(data_path,
                                                                                                    dict_path)
        print("num_classes", self.num_classes)
        print("max_seq_num", self.max_seq_num)
        print("max_seq_len", self.max_seq_len)

        if self.max_seq_num > max_seq_num:
            self.max_seq_num = max_seq_num

    def process_xml(self, data_path, dict_path):
        # 解析xml文件，并tokenize
        tok = BertTokenizer.from_pretrained(dict_path)
        max_seq_num = 0
        max_seq_len = 0

        docs = []  # [doc_num, seq_len], list of list
        texts = []  # [doc_num, seq_num, seq_len], list of list of list
        labels = []  # [doc_num], list
        label2idx = {}

        tree = ET.parse(data_path)
        root = tree.getroot()
        for document_set in root:
            for document in document_set:
                # label
                label = document.attrib['document_level_value']
                if label not in label2idx:
                    label2idx[label] = len(label2idx)
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
                    if len(tok_sent) > max_seq_len:
                        max_seq_len = len(tok_sent)
                    docs.append(tok_sent)

                if len(docs) > max_seq_num:
                    max_seq_num = len(docs)
                texts.append(docs)
                docs = []
        return texts, labels, len(label2idx), max_seq_num, max_seq_len

    def get_label_size(self):
        return

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]  # scalar
        doc = self.texts[index]  # doc, list of list
        masks = []

        for sent in doc:
            mask = [1] * len(sent) + [0] * (self.max_seq_len - len(sent))
            masks.append(mask)
            if len(sent) < self.max_seq_len:
                sent += [0] * (self.max_seq_len - len(sent))
        if len(doc) < self.max_seq_num:
            masks += [[0] * self.max_seq_len] * (self.max_seq_num - len(doc))
            doc += [[0] * self.max_seq_len] * (self.max_seq_num - len(doc))
        elif len(doc) > self.max_seq_num:
            masks = masks[:self.max_seq_num]
            doc = doc[:self.max_seq_num]

        return torch.tensor(doc), torch.tensor(masks), torch.tensor(label, dtype=torch.long)


if __name__ == '__main__':
    test = MyDataset(data_path="../data/dlef_corpus/english.xml", dict_path="../data/bert-base-uncased")
    doc, masks, label = test.__getitem__(index=701)
    print("end")
