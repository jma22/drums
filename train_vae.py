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
#     out = 1/(100*(1+math.exp((100-epoch)/10)))
#     return out*1e-5
def weight(epoch):
    if epoch%25<13:
        return 1e-4 * (epoch%25)/13
    else:
        return 1e-4
             
torch.cuda.set_device(3)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

dur=[]
epochs = 1000
train_num = 20
test_num =2
lr = 0.001
name = "lstmcyclic2561e-4"
sessions = ["session1","session2","session3"]
decoder = lstmd().to(device)
encoder = lstmm().to(device)
d_optimizer = torch.optim.Adam(decoder.parameters(),lr = lr)
e_optimizer = torch.optim.Adam(encoder.parameters(),lr = lr)
writerb = SummaryWriter('runs/{}'.format(name))

for epoch in range(epochs):
    t0 = time.time()
    for i in range(train_num):
        print(i)
        decoder.train()
        encoder.train()
        ## pick midi
        current = os.getcwd()
        path = os.path.join(current, "data") 
        ##batch
        batch_size = 200
        in_tensor = []
        ground_truth = []

        for j in range(batch_size):
            tpath = os.path.join(path ,"drummer" +str(random.randint(1,9))) 

            
            ## choose session
            tpath = os.path.join(tpath, random.choice(os.listdir(tpath)))
            ## choose midisa
            midi_path = os.path.join(tpath, random.choice(os.listdir(tpath))) 
            ##input array making
            try:
#                 in_array = midi_to_input(midi_path)np.expand_dims(in_array,0)
#                 ground_truth.append(np.expand_dims(midi_to_input(midi_path),0))
                ground_truth.append(midi_to_input(midi_path))


                in_tensor.append(midi_to_input(midi_path).reshape(-1,8*9))
            except notLong:
                continue
        
        in_tensor = torch.tensor(in_tensor).to(device=device, non_blocking=True).float()
        ground_truth = torch.tensor(ground_truth).to(device=device, non_blocking=True).float().unsqueeze(0)

        print(in_tensor.shape)
        thickness = in_tensor.shape[0]
        
        #forward
        d_optimizer.zero_grad()
        e_optimizer.zero_grad()
        mean,var = encoder(in_tensor)
#         print('bruh')
#         print(mean.shape)
#         print(var.shape)
        sample = mean + torch.exp(var / 2) * torch.randn(1,thickness, 256).to(device)
        print(sample.shape)
        output = decoder(sample)
        print(output.shape)
        recon_loss = F.binary_cross_entropy(output,in_tensor) 
        kl_loss = 0.5 * torch.sum(torch.exp(var) + mean**2 - 1. - var)
        train_loss = weight(epoch) * kl_loss + recon_loss
        #backward
        train_loss.backward()
        d_optimizer.step()
        e_optimizer.step()
        dur.append(time.time() - t0)
        
    ## eval
    
    loss_total = 0
    kl_total = 0
    recon_total = 0
    count = 0
    for i in range(test_num):
        ## pick midi
        current = os.getcwd()
        path = os.path.join(current, "data") 
        path = os.path.join(path, "drummer1") 
        ## choose session
        path = os.path.join(path, "eval_session") 
        ## choose midi
        
        ##batch
        batch_size = 50
        in_tensor = []
        ground_truth = []

        for j in range(batch_size):

            
            ## choose midisa
            midi_path = os.path.join(path, random.choice(os.listdir(path))) 
            ##input array making
            try:
#                 in_array = midi_to_input(midi_path)np.expand_dims(in_array,0)
#                 ground_truth.append(np.expand_dims(midi_to_input(midi_path),0))
                ground_truth.append(midi_to_input(midi_path))


                in_tensor.append(midi_to_input(midi_path).reshape(-1,8*9))
            except notLong:
                continue
        
        in_tensor = torch.tensor(in_tensor).to(device=device, non_blocking=True).float()
        ground_truth = torch.tensor(ground_truth).to(device=device, non_blocking=True).float().unsqueeze(0)


        
        #forward
        decoder.eval()
        encoder.eval()
        with torch.no_grad():
            mean,var = encoder(in_tensor)
            sample = mean + torch.exp(var / 2) * torch.randn(1, 256).to(device)
            output = decoder(sample)
            trecon_loss = F.binary_cross_entropy(output,in_tensor) 
            tkl_loss = 0.5 * torch.sum(torch.exp(var) + mean**2 - 1. - var)
            kl_total +=tkl_loss
            recon_total += trecon_loss
            loss_total += (tkl_loss + trecon_loss)
        count+=1
    print("Epoch {:05d} | Train_loss {:.4f} | Test_loss {:.4f}  | Time(s) {:.4f}" .format(epoch, train_loss.item(), loss_total/count, np.mean(dur)))
    writerb.add_scalar('test_loss',loss_total/count,epoch)
    writerb.add_scalar('test_kl_loss',kl_total/count,epoch)
    writerb.add_scalar('test_recon_loss',recon_total/count,epoch)
    writerb.add_scalar('train_loss',train_loss,epoch)
    writerb.add_scalar('train_kl_loss',kl_loss,epoch)
    writerb.add_scalar('weighted_train_kl_loss',kl_loss*weight(epoch),epoch)
    writerb.add_scalar('train_recon_loss',recon_loss,epoch)
    writerb.add_scalar('kl_weight',weight(epoch),epoch)
             
    dur.append(time.time() - t0)
    torch.save(encoder.state_dict(), "models/{}_en.pth".format(name))
    torch.save(decoder.state_dict(), "models/{}_de.pth".format(name))