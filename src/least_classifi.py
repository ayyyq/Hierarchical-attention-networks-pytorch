from transformers import BertTokenizer
import torch
import torch.nn as nn
from transformers import BertModel
import time
import argparse
import os
from transformers.optimization import AdamW
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

class review(torch.utils.data.Dataset):
    def __init__(self,is_training=True):
        self.tokenizer = BertTokenizer.from_pretrained('/home/zhoubo/program/BERT/chinese-macbert')
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.is_traing=is_training
        with open('data/chnsenticorp/train.tsv',encoding='utf8') as f:
            for sent in f:
                self.train_label.append(int(sent.split('\t',1)[0]))
                self.train_data.append(sent.split('\t',1)[1].strip())
        with open('data/chnsenticorp/test.tsv',encoding='utf8') as f:
            for sent in f:
                self.test_label.append(int(sent.split('\t', 1)[0]))
                self.test_data.append(sent.split('\t', 1)[1].strip())
        if self.is_traing:
            data=self.tokenizer(self.train_data, return_tensors='pt', padding=True, truncation=True, max_length=512)
            self.data=data['input_ids']
            self.attention=data['attention_mask']
            self.label=self.train_label
        else:
            data = self.tokenizer(self.test_data, return_tensors='pt', padding=True, truncation=True, max_length=512)
            self.data = data['input_ids']
            self.attention = data['attention_mask']
            self.label = self.test_label
    def __getitem__(self, index):
        return self.data[index], self.attention[index], self.label[index]
    def __len__(self):
        return len(self.data)

def get_train_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=54,help = '每批数据的数量')
    parser.add_argument('--nepoch',type=int,default=20,help = '训练的轮次')
    parser.add_argument('--lr',type=float,default=0.5e-4,help = '学习率')
    parser.add_argument('--gpu',type=bool,default=True,help = '是否使用gpu')
    parser.add_argument('--num_workers',type=int,default=2,help='dataloader使用的线程数量')
    parser.add_argument('--num_labels',type=int,default=2,help='分类类数')
    parser.add_argument('--data_path',type=str,default='./data',help='数据路径')
    opt=parser.parse_args()
    print(opt)
    return opt

class sentiment(nn.Module):
    def __init__(self):
        super(sentiment, self).__init__()
        self.encoder = BertModel.from_pretrained('/home/zhoubo/program/BERT/chinese-macbert')
        self.classificaion = nn.Linear(768, 2)
    def forward(self, input, attention):
        feature=self.encoder(input,attention)[1]
        logit=self.classificaion(feature)
        return logit

def get_data(opt):
    trainset = review(is_training = True)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    testset = review(is_training = False)
    testloader=torch.utils.data.DataLoader(testset,batch_size=50,shuffle=False,num_workers=opt.num_workers)
    return trainloader,testloader

def train(epoch,model,trainloader,testloader,optimizer,opt):
    print('\ntrain-Epoch: %d' % (epoch+1))
    model.train()
    start_time = time.time()
    print_step = int(len(trainloader)/10)
    for batch_idx,(data, attention, label) in enumerate(trainloader):
        if opt.gpu:
            data = data.cuda()
            attention = attention.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        logit = model(data, attention)
        loss=nn.functional.cross_entropy(logit,label)
        loss.backward()
        optimizer.step()
        if batch_idx % print_step == 0:
            acc = test(model, testloader, opt)
            print("Epoch:%d [%d|%d] loss:%f Acc:%.3f" % (epoch + 1, batch_idx, len(trainloader), loss.mean(), acc))
    print("time:%.3f" % (time.time() - start_time))

def test(model,testloader,opt):
    model.eval()
    total=0
    correct=0
    with torch.no_grad():
        for batch_idx,(data, attention, label) in enumerate(testloader):
            if opt.gpu:
                data = data.cuda()
                attention = attention.cuda()
                label = label.cuda()

            logit = model(data, attention)
            _,predicted=torch.max(logit.data,1)
            total+=data.size(0)
            correct+=predicted.data.eq(label.data).cpu().sum()
    acc=(1.0*correct.numpy())/total
    return acc

if __name__=='__main__':
    opt = get_train_args()
    trainloader,testloader = get_data(opt)
    model = sentiment()
    if opt.gpu:
        model = nn.DataParallel(model)
        model.cuda()
    optimizer=AdamW(model.parameters(),lr=opt.lr)
    for epoch in range(opt.nepoch):
        train(epoch,model,trainloader,testloader,optimizer,opt)
