from torch.utils.data import Dataset,DataLoader
import os
from torch.utils.tensorboard import  SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F
import re
import collections
import random
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def read_text(address):    
    with open(address,'r') as f:
        lines=f.readlines()
    return [re.sub('[^A-Za-z]+',' ',line).strip().lower() for line in lines]
def token(lines):
    return [line.split() for line in lines]
def count(tokens):
    token=[token for line in tokens for token in line]
    return collections.Counter(token)
class Vocab():
    def __init__(self,tokens,min_fraq=0):
        counter=count(tokens)
        self.unique_token_counter=sorted(counter.items(),key=lambda x:x[1],reverse=True)
        self.densed_unique_token=['<unknown>']
        self.token2idx=collections.defaultdict(lambda:0)  # 未知词返回0
        for i in self.unique_token_counter:
            if i[1]>=min_fraq:
                self.densed_unique_token.append(i[0])
                self.token2idx[i[0]]=len(self.densed_unique_token)-1
    def __len__(self):
        return len(self.densed_unique_token)
    def __getitem__(self,idx):
        if not isinstance(idx,(list,tuple)):
            return self.densed_unique_token[idx]
        return [self.__getitem__(i) for i in idx]
def load_text_by_word(address,min_freq=0):
    lines=read_text(address)
    lines=[i for i in lines if i]
    vocab=Vocab(token(lines),min_fraq=min_freq)
    corpus=[j for i in lines for j in i.split()]
    return corpus,vocab
corpus,vocab=load_text_by_word('Educated.txt',min_freq=1)
def generate_random_material(corpus,batch_size,num_steps,vocab):
    print(corpus)
    #corpus=corpus[random.randint(0,num_steps-1):]
    corpus=[vocab.token2idx[i] for i in corpus]
    print(corpus)
    batch=(len(corpus)-1)//(batch_size*num_steps)
    trainset=[]
    testset=[]
    while True:
        for i in range(batch):
            trainset.append(torch.tensor([corpus[i*batch_size*num_steps+j*batch_size:i*batch_size*num_steps+j*batch_size+batch_size] for j in range(num_steps)]))
            testset.append(torch.tensor([corpus[i*batch_size*num_steps+j*batch_size+1:1+i*batch_size*num_steps+j*batch_size+batch_size] for j in range(num_steps)]))
        yield trainset,testset
class rnn(torch.nn.Module):
    def __init__(self,rnn_layer,vocab_size,**kwargs):
        super(rnn,self).__init__(**kwargs)
        self.rnn_layer=rnn_layer
        self.vocab_size=vocab_size
        self.num_hiddens=self.rnn_layer.hidden_size
        self.linear=torch.nn.Linear(self.num_hiddens, self.vocab_size)
    def forward(self,inputs,state):
        X=F.one_hot(inputs.T.long(),self.vocab_size).to(torch.float32)
        X=X.to(device)
        state=state.to(device)
        Y,state=self.rnn_layer(X,state)
        output=self.linear(Y.reshape((-1,Y.shape[-1])))
        return output,state
    def begin_state(self,batch_size=1):
        return torch.zeros((self.rnn_layer.num_layers,batch_size,self.num_hiddens),device=device)
    
rnn_layer=torch.nn.RNN(len(vocab),512,2)
net=rnn(rnn_layer,len(vocab))
net=net.to(device)
def predict(prefix,number,vocab,net):
    initial_state=net.begin_state()
    prefix=prefix.lower().split()
    print(prefix)
    output=[vocab.token2idx[prefix[0]]]
    state=initial_state
    for i in range(1,len(prefix)):
        print(i,output[-1])
        _,state=net(torch.tensor([output[-1]],device=device).reshape(1,1),state)
        output.append(vocab.token2idx[prefix[i]])
    for i in range(number):
        out,state=net(torch.tensor([output[-1]],device=device).reshape(1,1),state)
        output.append((torch.argmax(out)))
    return [vocab[i] for i in output]
def grad_clipping(net, theta):
    """裁剪梯度。"""
    # 如果 net 是 nn.Module 的实例（即使用 PyTorch 构建的模型）
    if isinstance(net, torch.nn.Module):
        # 获取所有需要计算梯度的参数列表
        params = [p for p in net.parameters() if p.requires_grad]
    # 如果 net 是自定义的模型（例如上述的 RNNModelScratch）
    else:
        # 获取自定义模型的参数列表
        params = net.params
    # 计算参数梯度的范数，即所有参数梯度平方和的平方根
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    # 如果梯度范数超过指定阈值 theta
    if norm > theta:
        # 对于每个参数
        for param in params:
            # 将参数的梯度值裁剪至指定范围内，保持梯度范数不超过 theta
            param.grad[:] *= theta / norm
def train_epoch(net,trainset,testset,loss,optimizer,batch_size):
    state=net.begin_state(batch_size=batch_size)
    for i in range(len(trainset)):
        state.detach_()
        X=trainset[i]
        Y=testset[i]
        Y=Y.T.reshape(-1).to(device)
        X,state=net(X,state)
        testloss=loss(X,Y.long().to(device))
        optimizer.zero_grad()
        testloss.backward()
        grad_clipping(net,1)
        optimizer.step()
def train_mul_epoch(net,learningrate,corpus,batch_size,num_steps,max_epoch,vocab):
    optimizer=torch.optim.Adam(net.parameters(),lr=learningrate)
    loss=torch.nn.CrossEntropyLoss()
    loss=loss.to(device)
    t=generate_random_material(corpus,batch_size,num_steps,vocab)
    for i in range(max_epoch):
        print(i)
        trainset,testset=next(t)
        #for i in range(len(trainset)):
        train_epoch(net,trainset,testset,loss,optimizer,num_steps)
        print(predict('time is',10,vocab,net))
lr=0.01
batch_size=36
num_steps=30
train_mul_epoch(net,learningrate=lr,corpus=corpus,batch_size=batch_size,num_steps=num_steps,max_epoch=20,vocab=vocab)
torch.save(net,'rnn_predict.pth')