import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import math


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
        
    def forward(self, x, seq_ls):
        x = x + self.attn(self.ln_1(x), seq_ls) 
        x = x + self.mlp(self.ln_2(x))
        
        return x
    
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc   = nn.Linear(config.n_embed, config.forward_expansion*config.n_embed)
        self.act    = nn.ReLU()
        self.c_proj = nn.Linear(config.forward_expansion*config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        # key, value, and query for all heads
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embed
        
    def forward(self, x, seq_ls):
        device = x.device
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embed)
        # get query, key, values for all heads in batch ad move head forward to the batch_dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, # of heads, T, head_size) ->(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, # of heads, T, head_size)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, # of heads, T, head_size)
        
        # self-attention; self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q@k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        
        # create different masks for various seqs
        tp_mask = torch.zeros((B, T, T)).long().to(device)
        for i in range(len(seq_ls)):
            tp_mask[i, :, :seq_ls[i]]=1
        tp_mask=tp_mask.unsqueeze(1)
        # print(tp_mask.shape)
        att = att.masked_fill(tp_mask==0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att@v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B, T, C) # reassemble all head outputs side by side
        
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class TransModel(nn.Module):
    def __init__(self, config):
        super(TransModel, self).__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(config.fctor_size, config.n_embed),# feature embedding
            # tpe = nn.Embedding(config.n_type, config.n_embed), # position embedding
            wpe = nn.Embedding(config.block_size, config.n_embed), # position embedding
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
    def forward(self, seqs_info, seq_ls, targets=None):
        device = seqs_info.device
        b, t, fs = seqs_info.size() # b: batch size, t: sequence length
        pos = torch.arange(0,t, dtype=torch.long, device=device).unsqueeze(0) # shape (1,t)
        
        # forward hte model
        tok_emb = self.transformer.wte(seqs_info) # word embedding, shape (b, t, n_embed)
        # typ_emb = self.transformer.tpe(type_info) # nuclei type embedding, shape(b, t, n_embed)
        pos_emb = self.transformer.wpe(pos) # position embedding, shape (1, t, n_embed)
        x = tok_emb+pos_emb# +typ_emb
        x = self.transformer.drop(x)
        # x = self.transformer.drop(tok_emb+fct_emb+pos_emb)
        for block in self.transformer.h:
            x = block(x, seq_ls)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        acc = None
        if targets is not None:
            # select logits and targets
            tp_l = torch.tensor([[0.,0.,0.]]).to(device)
            tp_t = torch.tensor([0]).long().to(device)
            for i in range(len(seq_ls)):
                tp_l = torch.cat((tp_l, logits[i][:seq_ls[i]]))
                tp_t = torch.cat((tp_t, targets[i][:seq_ls[i]]))
            tp_l = tp_l[1:]
            tp_t = tp_t[1:]
            loss = F.cross_entropy(tp_l.view(-1, tp_l.size(-1)), tp_t.view(-1), ignore_index=-1)
            pred = torch.argmax(tp_l, dim=1)
            acc = sum(pred==tp_t)/len(tp_t)
            
        return logits, loss, acc
    
class Config():
    def __init__(self, vocab_size=3, n_embed=30, dropout=0, n_layer=3, block_size=48, 
                 forward_expansion=4, n_head=6, fctor_size=6, n_type=17):
        self.vocab_size=vocab_size # number of vocabulary size
        self.n_embed=n_embed # embed size for word and positions
        self.dropout=dropout # probability values are zeroed
        self.n_layer=n_layer # number of transformer block layers
        self.block_size=block_size # number of words to be considered: 47
        self.forward_expansion=forward_expansion # expansion of features in the linear layer
        self.n_head=n_head # number of heads
        self.fctor_size=fctor_size
        self.n_type=n_type
