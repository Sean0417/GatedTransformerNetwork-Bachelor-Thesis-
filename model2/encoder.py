from torch.nn import Module
import torch

from module2.multiHeadAttention import MultiHeadAttention
from module2.feedForward import FeedForward
from module2.probsparse_attn import AttentionLayer, ProbAttention

class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 encoder_type:str,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, device=device, dropout=dropout)
        self.mask = mask
        self.prob_attn = ProbAttention(mask_flag=mask,attention_dropout = 0.1,output_attention=True)
        self.attn_layer = AttentionLayer(self.prob_attn,d_model=d_model,n_heads=h,d_keys=q, d_values=v)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)
        self.encoder_type =encoder_type

    def forward(self, x, stage):

        residual = x
        # stepwise uses ProbSparse attn, channel wise uses normal attn
        if self.encoder_type == 'stepwise':
            if stage == 'train' and self.mask == True:
                attn_mask = self.mask
            else:
                attn_mask = False
            x, score = self.attn_layer(x, x, x, attn_mask=attn_mask)
            # print('stepwise is used')
        elif self.encoder_type == 'channelwise':
            # print('channelwise is used')
            x, score = self.MHA(x, stage)


        # channelwise uses Probsparse attn, stepwise uses normal attn
        # if self.encoder_type == 'stepwise':
        #     x, score = self.MHA(x, stage)
        #     # print('stepwise is used')
        # elif self.encoder_type == 'channelwise':
        #     # print('channelwise is used')

        #     if stage == 'train' and self.mask == True:
        #         attn_mask = self.mask
        #     else:
        #         attn_mask = False
        #     x, score = self.attn_layer(x, x, x, attn_mask=attn_mask)


        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score
