import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from helpers import *
from classes import *
import random
import time
from torch.utils.tensorboard import SummaryWriter

## Set device

torch.cuda.set_device(3)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

dur=[]
epochs = 1000
train_num = 1000
test_num =100
lr = 0.001
name = "autoencode"
sessions = ["session1","session2","session3"]
net = ae().to(device)
optimizer = torch.optim.Adam(net.parameters(),lr = lr)
writerb = SummaryWriter('runs/{}'.format(name))

for epoch in range(epochs):
    t0 = time.time()
    for i in range(train_num):
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
            in_array = midi_to_input(midi_path)
        except notLong:
            continue
        
        in_tensor = torch.tensor(np.expand_dims(in_array,0)).to(device=device, non_blocking=True).float()
        
        #forward
        optimizer.zero_grad()
        output = net(in_tensor)
        train_loss = F.binary_cross_entropy(output,in_tensor)
        #backward
        train_loss.backward()
        optimizer.step()
        dur.append(time.time() - t0)
        
    ## eval
    loss_total = 0
    count = 0
    for j in range(test_num):
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
            in_array = midi_to_input(midi_path)
        except notLong:
            continue
        in_tensor = torch.tensor(np.expand_dims(in_array,0)).to(device=device, non_blocking=True).float()
        
        #forward
        net.eval()
        with torch.no_grad():
            output = net(in_tensor)
            loss_total+= F.binary_cross_entropy(output,in_tensor)
        count+=1
    print("Epoch {:05d} | Train_loss {:.4f} | Test_loss {:.4f}  | Time(s) {:.4f}" .format(epoch, train_loss.item(), loss_total/count, np.mean(dur)))
    writerb.add_scalar('test_loss',loss_total/count,epoch)
    writerb.add_scalar('train_loss',train_loss,epoch)
    dur.append(time.time() - t0)
    torch.save(net.state_dict(), "models/{}.pth".format(name))
        
        
        