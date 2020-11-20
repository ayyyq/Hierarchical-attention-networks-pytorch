import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import time
import argparse
import os
from transformers.optimization import AdamW
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'


def k_fold_split(data_path):
    # 划分训练集和测试集
    train_idx, test_idx = [], []
    labels = []
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

    skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    for train, test in skf.split(np.zeros(len(labels)), labels):
        train_idx.append(train)
        test_idx.append(test)
    return train_idx, test_idx, label2idx


class review(torch.utils.data.Dataset):
    def __init__(self, data_path, train_idx, test_idx, label2idx, is_training=True):
        self.tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
        self.is_traing = is_training

        self.train_data, self.train_label, self.train_ids = self.process_xml(data_path, train_idx, label2idx)
        self.test_data, self.test_label, self.test_ids = self.process_xml(data_path, test_idx, label2idx)

        '''with open('data/chnsenticorp/train.tsv',encoding='utf8') as f:
            for sent in f:
                self.train_label.append(int(sent.split('\t',1)[0]))  # [int, int, ...]
                self.train_data.append(sent.split('\t',1)[1].strip())  # [str, str, ...]
        with open('data/chnsenticorp/test.tsv',encoding='utf8') as f:
            for sent in f:
                self.test_label.append(int(sent.split('\t', 1)[0]))
                self.test_data.append(sent.split('\t', 1)[1].strip())'''
        if self.is_traing:
            data = self.tokenizer(self.train_data, return_tensors='pt', padding=True, truncation=True, max_length=512)
            # return_tensors: return torch.Tensor
            # padding: pad to the longest sequence in the batch
            # truncation: truncate to a maximum length
            self.data = data['input_ids']
            self.attention = data['attention_mask']
            self.label = self.train_label
            self.ids = self.train_ids
        else:
            data = self.tokenizer(self.test_data, return_tensors='pt', padding=True, truncation=True, max_length=512)
            self.data = data['input_ids']
            self.attention = data['attention_mask']
            self.label = self.test_label
            self.ids = self.test_ids

    def process_xml(self, data_path, index, label2idx):
        texts = []  # [doc_num], list of list of str
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
                doc = ''
                for sentence in document:
                    if sentence.text == '-EOP- .':
                        continue
                    sent = ''
                    for text in sentence.itertext():
                        sent += text
                    sent = sent.replace('-EOP- ', '').lower()
                    doc += sent + ' '

                texts.append(doc.strip())
        return texts, labels, ids

    def __getitem__(self, index):
        return self.data[index], self.attention[index], self.label[index], self.ids[index]

    def __len__(self):
        return len(self.data)


def get_train_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help = '每批数据的数量')
    parser.add_argument('--nepoch', type=int, default=100, help = '训练的轮次')
    parser.add_argument('--lr', type=float, default=0.5e-4, help = '学习率')
    parser.add_argument('--gpu', type=bool, default=True, help = '是否使用gpu')
    parser.add_argument('--num_workers', type=int, default=2, help='dataloader使用的线程数量')
    parser.add_argument('--data_path', type=str, default='../data/dlef_corpus/english.xml', help='数据路径')

    parser.add_argument('--bert_path', type=str, default="../data/bert-base-uncased")
    parser.add_argument('--output_path', type=str, default="output.txt")
    parser.add_argument('--model_path', type=str, default="../trained_models/model.pt")

    opt = parser.parse_args()
    print(opt)
    return opt


class sentiment(nn.Module):
    def __init__(self, num_classes):
        super(sentiment, self).__init__()
        self.encoder = BertModel.from_pretrained(opt.bert_path)
        self.classificaion = nn.Linear(768, num_classes)

    def forward(self, input, attention):
        feature = self.encoder(input, attention)[1]
        logit = self.classificaion(feature)
        return logit


def get_data(opt, train_idx, test_idx, label2idx):
    trainset = review(opt.data_path, train_idx, test_idx, label2idx, is_training=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=False,
                                              num_workers=opt.num_workers)
    testset = review(opt.data_path, train_idx, test_idx, label2idx, is_training=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=opt.num_workers)
    return trainloader, testloader


def train(epoch, model, trainloader, testloader, optimizer, opt, max_f1):
    print('\ntrain-Epoch: %d' % (epoch + 1))
    model.train()
    start_time = time.time()
    print_step = int(len(trainloader) / 10)
    for batch_idx, (data, attention, label, _) in enumerate(trainloader):
        if opt.gpu:
            data = data.cuda()
            attention = attention.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        logit = model(data, attention)
        loss = nn.functional.cross_entropy(logit , label)
        loss.backward()
        optimizer.step()
        if batch_idx % print_step == 0:
            acc, f1_micro, f1_macro = test(model, testloader, opt)
            print("Epoch:%d [%d|%d] loss:%f Acc:%.3f F1_micro:%.3f F1_macri:%.3f" % (epoch + 1, batch_idx, len(trainloader),
                                                                                     loss.mean(), acc, f1_micro,
                                                                                     f1_macro))
            if f1_micro + f1_macro > max_f1:
                max_f1 = f1_micro + f1_macro
                torch.save({
                    'epoch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, opt.model_path)
    print("time:%.3f" % (time.time() - start_time))
    return max_f1


def test(model, testloader, opt, wr=False):
    f = open(opt.output_path, 'w')
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (data, attention, label, id) in enumerate(testloader):
            if opt.gpu:
                data = data.cuda()
                attention = attention.cuda()
                label = label.cuda()

            logit = model(data, attention)
            _, predicted = torch.max(logit.data,1)
            total += data.size(0)
            correct += predicted.data.eq(label.data).cpu().sum()
            y_true += label.cpu().data.numpy().tolist()
            y_pred += predicted.data.numpy().tolist()

            if wr:
                batch = label.shape[0]
                for i in range(batch):
                    f.write(id[i] + "\t" + str(label[i].item()) + "\t" + str(predicted[i].item()) + "\n")

    acc = (1.0 * correct.numpy()) / total
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    if wr:
        f.write("acc: " + str(acc) + "\n")
        f.write("f1_micro: " + str(f1_micro) + "\n")
        f.write("f1_macro: " + str(f1_macro) + "\n")
    f.close()
    return acc, f1_micro, f1_macro


if __name__=='__main__':
    opt = get_train_args()
    train_idx, test_idx, label2idx = k_fold_split(opt.data_path)
    trainloader, testloader = get_data(opt, train_idx[0], test_idx[0], label2idx)
    model = sentiment(len(label2idx))
    if opt.gpu:
        model = nn.DataParallel(model)
        model.cuda()
    optimizer = AdamW(model.parameters(), lr=opt.lr)

    max_f1 = 0
    for epoch in range(opt.nepoch):
        max_f1 = train(epoch, model, trainloader, testloader, optimizer, opt, max_f1)

    checkpoint = torch.load(opt.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    acc, f1_micro, f1_macro = test(model, testloader, opt, wr=True)
    print("Epoch:%d Acc:%.3f F1_micro:%.3f F1_macro:%.3f" % (checkpoint['epoch'] + 1, acc, f1_micro, f1_macro))
