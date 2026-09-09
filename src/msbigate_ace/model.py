
from __future__ import annotations
import torch
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, in_dim, hid=128, out_dim=2, dropout=0.3):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid,out_dim))
    def forward(self,x): return self.net(x)

class MultiScaleCNNBranch(nn.Module):
    def __init__(self, in_dim, cnn_hid=96, kernels=(3,5,7), dropout=0.35):
        super().__init__(); self.kernels=list(kernels); self.convs=nn.ModuleList()
        for k in self.kernels:
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_dim,cnn_hid,kernel_size=k,padding=k//2),
                nn.BatchNorm1d(cnn_hid), nn.ReLU(), nn.Dropout(dropout)))
        self.out_dim=len(self.kernels)*cnn_hid*2
    def forward(self,x):
        _,L,_=x.size(); xc=x.transpose(1,2); outs=[]
        for conv in self.convs:
            y=conv(xc); outs.extend([y[:,:,L//2], torch.max(y,dim=2).values])
        return torch.cat(outs,dim=-1)

class MSBiGateAce(nn.Module):
    def __init__(self, input_dim=43, embedding_dim=64, hidden_size=160, lstm_layers=1,
                 cnn_channels=96, kernels=(3,5,7), dropout=0.35):
        super().__init__(); h=hidden_size
        self.out_dim=2*h
        self.input_norm=nn.LayerNorm(input_dim)
        self.input_proj=nn.Sequential(nn.Linear(input_dim,embedding_dim),nn.GELU(),nn.Dropout(dropout))
        self.lstm=nn.LSTM(input_size=embedding_dim,hidden_size=h,num_layers=lstm_layers,batch_first=True,
                          bidirectional=True,dropout=dropout if lstm_layers>1 else 0.0)
        self.lstm_proj=nn.Sequential(nn.Linear(4*h,2*h),nn.ReLU(),nn.Dropout(dropout))
        self.cnn_branch=MultiScaleCNNBranch(embedding_dim,cnn_channels,kernels,dropout)
        self.cnn_proj=nn.Sequential(nn.Linear(self.cnn_branch.out_dim,2*h),nn.ReLU(),nn.Dropout(dropout))
        self.gate_fc=nn.Linear(4*h,2*h); nn.init.constant_(self.gate_fc.bias,2.0)
        self.fuse_norm=nn.LayerNorm(2*h)
        self.head=MLPHead(2*h,128,2,dropout)
    def encode(self,x):
        x=self.input_proj(self.input_norm(x)); hseq,_=self.lstm(x)
        center=hseq[:,hseq.size(1)//2,:]; meanv=hseq.mean(dim=1)
        lstm_repr=self.lstm_proj(torch.cat([center,meanv],dim=-1))
        cnn_repr=self.cnn_proj(self.cnn_branch(x))
        gate=torch.sigmoid(self.gate_fc(torch.cat([lstm_repr,cnn_repr],dim=-1)))
        return self.fuse_norm(gate*lstm_repr+(1.0-gate)*cnn_repr)
    def forward(self,x): return self.head(self.encode(x))


def build_model(cfg):
    return MSBiGateAce(input_dim=int(cfg.get('input_dim',43)), embedding_dim=int(cfg['embedding_dim']),
                       hidden_size=int(cfg['hidden_size']), lstm_layers=int(cfg.get('lstm_layers',1)),
                       cnn_channels=int(cfg['cnn_channels']), kernels=cfg['kernels'], dropout=float(cfg['dropout']))

def count_parameters(model): return sum(p.numel() for p in model.parameters())
