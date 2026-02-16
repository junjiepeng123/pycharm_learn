import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,config.n_embd*3) #这里*3就是把query(x),key(x),value(x)一起完成了 并行化最大限度利用GPU
        self.c_proj = nn.Linear(config.n_embd,config.n_embd) #?

        self.n_head= config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias',torch.tril(torch.ones(config.n_embd,config.n_embd)).view(1,1,config.n_embd,config.n_embd))

    def forward(self,x):
        B,T,C = x.shape
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2) #沿着dim2，每n_embd取出来作为一个新的变量 (B,T,n_embd)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

        att =(q @ k.transpose(-2,-1)) *(1.0 / math.sqrt(k.size(-1))) #缩放基于C//self.n_head 这里计算得到的是注意力分数
        att = att.mask_fill(self.bias[:,:,:T,:T] == 0,float('inf'))
        att = F.softmax(att,dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        #把多头的结果给映射一下
        return y

class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,config.n_embd*4)
        self.gelu = nn.GELU(approximate='tanh') #函数形状很像relu但是在近似0处的梯度不为0
        self.c_proj = nn.Linear(config.n_embd*4,config.n_embd)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x= x + self.attn(self.ln_1(x)) #先标准化；保持残差流的干净
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int =384

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd), # token embeding
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.Modulelist([Block(config) for _ in range(config.n_layer)]), #h就是解码器 总计12个
            ln_f = nn.LayerNorm(config.n_embd)
        )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) #GPT的创新
