import torch
import torch.nn as nn
from torch.nn import functional as F

class lstmDis(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=256
        self.rnn = nn.LSTM(9*8,self.hidden,batch_first=True,bidirectional=True)
        self.mean = nn.Linear(self.hidden*2*2,1)
        
    def forward(self,x):
        h0 = torch.randn(2, x.shape[0], self.hidden).to('cuda')
        c0 = torch.randn(2, x.shape[0], self.hidden).to('cuda')
        output, (hn, cn) = self.rnn(x, (h0, c0))
        ans = torch.cat((hn,cn),2)
        ans = ans.permute(1, 0,2) 
        ans = ans.reshape([ans.shape[0],-1])
        ans = self.mean(ans)
        ans = torch.sigmoid(ans)
        return ans
    
# class lstmGen(nn.Module):
#     def __init__(self):
#         super().__init__()
#         hidden = 256
#         self.rnn = nn.LSTM(9*8,hidden)
#         self.h = nn.Linear(hidden,hidden)
#         self.c = nn.Linear(hidden,hidden)
#         self.hout = nn.Linear(hidden,72)
        
#     def forward(self,z,num=8):
#         x = torch.zeros(1,z.shape[1],8*9).to('cpu')
#         out = []
#         hn = self.h(z)
#         cn = self.c(z)
        
#         for i in range(num):
#             x, (hn, cn) = self.rnn(x, (hn, cn))
#             x= torch.sigmoid(self.hout(x))
#             out.append(x.squeeze(0).unsqueeze(1))
            
#         return torch.cat(out,1)

class lstmm(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=256
        self.rnn = nn.LSTM(9*8,self.hidden,batch_first=True)
        self.mean = nn.Linear(self.hidden*2,self.hidden)
        self.var = nn.Linear(self.hidden*2,self.hidden)
        
    def forward(self,x):
        h0 = torch.randn(1, x.shape[0], self.hidden).to('cuda')
        c0 = torch.randn(1, x.shape[0], self.hidden).to('cuda')
        output, (hn, cn) = self.rnn(x, (h0, c0))
        hidden = torch.cat((hn,cn),2)
        mean = self.mean(hidden)
        var = self.var(hidden)
        

        
        return mean,var
    
class lstmd(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 256
        self.rnn = nn.LSTM(9*8,hidden)
        self.h = nn.Linear(hidden,hidden)
        self.c = nn.Linear(hidden,hidden)
        self.hout = nn.Linear(hidden,72)
        
    def forward(self,z,num=8):
        x = torch.zeros(1,z.shape[1],8*9).to('cuda')
        out = []
        hn = self.h(z)
        cn = self.c(z)
        
        for i in range(num):
            x, (hn, cn) = self.rnn(x, (hn, cn))
            x= torch.sigmoid(self.hout(x))
            out.append(x.squeeze(0).unsqueeze(1))
            
        return torch.cat(out,1)
        
   

        
        
        
    
    
class ae(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
                  nn.Linear(576,256),
                  nn.ReLU(),
                  nn.Linear(256,64),
                  nn.ReLU(),
                  nn.Linear(64,8)
                )
        self.decode = nn.Sequential(
                  nn.Linear(8,64),
                  nn.ReLU(),
                  nn.Linear(64,256),
                  nn.ReLU(),
                  nn.Linear(256,576),
                  nn.Sigmoid()
                )


    def forward(self,x):
        hidden = self.encode(x)
        y = self.decode(hidden)
        return y

       
class bigger_v_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decode = nn.Sequential(
                    
                  nn.Linear(32,64),
                  nn.ReLU(),
                  nn.Linear(64,256),
                  nn.ReLU(),
                  nn.Linear(256,576),
                  nn.Sigmoid()
                )


    def forward(self,sample):   
        y = self.decode(sample)
        return y
    
    
        
class conv_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
                  nn.Conv1d(1, 16, 16),
                  nn.LeakyReLU(),
                  nn.Conv1d(16, 64, 16),
                  nn.LeakyReLU(),
                  nn.MaxPool1d(64*8)
                
#                   nn.Linear(256,64),
#                   nn.ReLU(),
#                   nn.Linear(64,64)
                )
        self.meme = nn.Linear(64,64)
        self.mean = nn.Linear(64,32)
        self.var = nn.Linear(64,32)



    def forward(self,x):
        hidden = self.encode(x)
#         print(hidden.squeeze().shape)
        hidden = self.meme(hidden.squeeze())
        mean = self.mean(hidden).unsqueeze(1)
        var = self.var(hidden).unsqueeze(1)
     
        return mean,var
    
class bigger_v_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
                  nn.Linear(576,256),
                  nn.ReLU(),
                  nn.Linear(256,64),
                  nn.ReLU(),
                  nn.Linear(64,64)
                )
        self.mean = nn.Linear(64,32)
        self.var = nn.Linear(64,32)



    def forward(self,x):
        hidden = self.encode(x)
        mean = self.mean(hidden)
        var = self.var(hidden)
     
        return mean,var
class v_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decode = nn.Sequential(
                    
                  nn.Linear(8,64),
                  nn.ReLU(),
                  nn.Linear(64,256),
                  nn.ReLU(),
                  nn.Linear(256,576),
                  nn.Sigmoid()
                )


    def forward(self,sample):   
        y = self.decode(sample)
        return y
    
class v_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
                  nn.Linear(576,256),
                  nn.ReLU(),
                  nn.Linear(256,64),
                  nn.ReLU(),
                  nn.Linear(64,16)
                )
        self.mean = nn.Linear(16,8)
        self.var = nn.Linear(16,8)



    def forward(self,x):
        hidden = self.encode(x)
        mean = self.mean(hidden)
        var = self.var(hidden)
     
        return mean,var
    