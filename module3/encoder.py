from torch.nn import Module
import torch

from module3.feedForward import FeedForward
from module3.longformer_attn import LongformerSelfAttention
from module3.longformer_attn import LongformerConfig
from module3.multiHeadAttention import MultiHeadAttention

class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 encoder_type:str,
                 attention_window:int,
                 device: str,
                 seq_len: int,
                 mask: bool = False,
                 dropout: float = 0.1,):
        super(Encoder, self).__init__()

       
        # config = LongformerConfig(attention_mode="tvm")
        config = LongformerConfig()
        # config.max_position_embeddings=seq_len
        config.hidden_size = d_model
        config.num_attention_heads = h
        config.attention_probs_dropout_prob = 0.1
        config.num_hidden_layers = 1 
        config.attention_window = [attention_window] * 1
        config.attention_dilation = [1] * 1
        self.device = device
        # config.intermediate_size = d_hidden



        self.longformer_attention = LongformerSelfAttention(config=config, layer_id=0,device=device)
        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, device=device, dropout=dropout)
        self.mask = mask
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden).to(device)
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model).to(self.device)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model).to(self.device)
        self.encoder_type = encoder_type

    def forward(self, x, stage):

        # seqlen = x.shape[1]
        # if  seqlen % (self.attention_window * 2) != 0:
        #     padding_len = (self.attention_window * 2 - seqlen % (self.attention_window * 2)) % (self.attention_window * 2)
        #     if padding_len > 0:
        #         padding = torch.zeros(*x.shape[:-2], padding_len, *x.shape[-1:]).to(x.device)
        #         x = torch.cat([x, padding], dim=1)
        # else:
        #     pass

        residual = x
        # stepwise uses longformer, channel wise uses normal attn
        if self.encoder_type == 'channelwise':
            x, score = self.MHA(x, stage)
        elif self.encoder_type == 'stepwise':
            # if stage == 'train' and self.mask == True:
            #     attn_mask = self.mask
            # else:
            #     attn_mask = False
            x, score = self.longformer_attention(x,output_attentions=True)

        # channel wise use longformer, stepwise uses normal attn
        # if self.encoder_type == 'channelwise':
        #     if stage == 'train' and self.mask == True:
        #         attn_mask = self.mask
        #     else:
        #         attn_mask = False
        #     x, score = self.longformer_attention(x, attn_mask)
        # elif self.encoder_type == 'stepwise':
        #     x, score = self.MHA(x, stage)
            

        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score
