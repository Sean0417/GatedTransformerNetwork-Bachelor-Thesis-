from torch.nn import Module
import torch
from torch.nn import ModuleList
from model2.encoder import Encoder
import math
import torch.nn.functional as F


class Transformer(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 device: str,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False):
        super(Transformer, self).__init__()
        # stepwise, mask is given in the function, mask depends on the input
        self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device,
                                                  encoder_type='stepwise') for _ in range(N)])
        # channel_wise, mask isn't given here, so it's set to Fault by default.
        self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device,encoder_type='channelwise') for _ in range(N)])

        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model
        # self.stage = stage

    def forward(self, x, stage):
        # step-wise
        # Set mask and pe by default in stepwise encoders
        encoding_1 = self.embedding_channel(x) # x[batchsize, time_step, feature] -> encoding_1[batchsize, time_step, d_model]
        input_to_gather = encoding_1           # x[1000, 128, 14] -> [1000, 128, 512]

        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe

        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage) 

        # channel-wise
        # Not set mask and pe on channelwise encoders.
        encoding_2 = self.embedding_input(x.transpose(-1, -2)) # x[batchsize, time_step, feature] -> encoding_1[batchsize, feature, d_model]
        channel_to_gather = encoding_2                         # x[1000, 128, 14]->[1000, 14, 128] -> [1000, 14, 512]

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage) # [1000, 14, 512]

        # 3-d to 2-d
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1) # stepwise S [1000, d_model*]
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1) # channelwise C

        # gate
        h = self.gate(torch.cat([encoding_1, encoding_2], dim=-1))
        gate = F.softmax(h,dim=-1)
        g1 = gate[:,0:1]
        g2 = gate[:,1:2]
        encoding = torch.cat([encoding_1 * g1, encoding_2*g2], dim=-1) #y

        # output
        output = self.output_linear(encoding)

        return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
