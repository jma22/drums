import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from helpers import *
from classes import *
import math
import random
import time
from torch.utils.tensorboard import SummaryWriter

## Set device
# def weight(epoch):
#     out = 1/(1+math.exp((300-epoch)/60))
#     return out
             
torch.cuda.set_device(2)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

dur=[]
epochs = 1000
train_num = 300
test_num =30
lr = 0.001
name = "rnn"
sessions = ["session1","session2","session3"]
net = rnn().to(device)
optimizer = torch.optim.Adam(net.parameters(),lr = lr)
writerb = SummaryWriter('runs/{}'.format(name))

for epoch in range(epochs):
    t0 = time.time()
    for i in range(train_num):
        stop_reached = False
        step = 0
        net.train()
        ## pick midi
        current = os.getcwd()
        path = os.path.join(current, "data") 
        path = os.path.join(path, "drummer1") 
        ## choose session
        path = os.path.join(path, random.choice(sessions)) 
        ## choose midi
        midi_path = os.path.join(path, random.choice(os.listdir(path))) 
        
        ##input array making
        try:
            in_array = midi_to_input_uncap(midi_path)
        except notLong:
            continue
        
        in_tensor = torch.tensor(in_array).to(device=device, non_blocking=True).float()
        hidden = torch.randn([1,8]).to(device=device, non_blocking=True).float()
        while not stop_reached:
            #forward
            net.zero_grad()
            output, hidden = net(in_tensor[step],hidden)
            train_loss = F.binary_cross_entropy(output,in_tensor[step+1])
            #backward
            train_loss.backward()
            optimizer.step()
            step+=1
            if torch.equal(in_tensor[step+1],torch.ones(9).to(device=device, non_blocking=True).float()):
                stop_reached = True
                
        dur.append(time.time() - t0)
        
    ## eval
    
    loss_total = 0
    count = 0
    for j in range(test_num):
        stop_reached = False
        step = 0
        net.train()
        ## pick midi
        current = os.getcwd()
        path = os.path.join(current, "data") 
        path = os.path.join(path, "drummer1") 
        ## choose session
        path = os.path.join(path, "eval_session") 
        ## choose midi
        midi_path = os.path.join(path, random.choice(os.listdir(path))) 
        
        ##input array making
        try:
            in_array = midi_to_input_uncap(midi_path)
        except notLong:
            continue
        
        in_tensor = torch.tensor(in_array).to(device=device, non_blocking=True).float()
        hidden = torch.randn([1,8]).to(device=device, non_blocking=True).float()
        while not stop_reached:
            net.eval()
            with torch.no_grad():
            #forward
                output, hidden = net(in_tensor[step],hidden)
                test_loss = F.binary_cross_entropy(output,in_tensor[step+1])
            step+=1
            loss_total += test_loss
            if torch.equal(in_tensor[step+1],torch.ones(9).to(device=device, non_blocking=True).float()):
                stop_reached = True
        count+=1
    print("Epoch {:05d} | Train_loss {:.4f} | Test_loss {:.4f}  | Time(s) {:.4f}" .format(epoch, train_loss.item(), loss_total/count, np.mean(dur)))
    writerb.add_scalar('test_loss',loss_total/count,epoch)
    writerb.add_scalar('train_loss',train_loss,epoch)

             
    dur.append(time.time() - t0)
    torch.save(net.state_dict(), "models/{}.pth".format(name))