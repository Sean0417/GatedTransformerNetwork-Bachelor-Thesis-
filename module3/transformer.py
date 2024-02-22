from torch.nn import Module
import torch
from torch.nn import ModuleList
from module3.encoder import Encoder
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
                 attention_window:int,
                 device: str,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False):
        super(Transformer, self).__init__()
        self.d_model=d_model
        self.d_hidden=d_hidden
        self.q=q
        self.v=v
        self.h=h
        self.mask=mask
        self.dropout=dropout
        self.device=device
        self.N=N
        self.attention_window = attention_window
        # # stepwise, mask is given in the function, mask depends on the input
        # self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
        #                                           d_hidden=d_hidden,
        #                                           q=q,
        #                                           v=v,
        #                                           h=h,
        #                                           mask=mask,
        #                                           dropout=dropout,
        #                                           device=device,
        #                                           encoder_type='stepwise') for _ in range(N)])
        # # channel_wise, mask isn't given here, so it's set to Fault by default.
        # self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
        #                                           d_hidden=d_hidden,
        #                                           q=q,
        #                                           v=v,
        #                                           h=h,
        #                                           dropout=dropout,
        #                                           device=device,encoder_type='channelwise') for _ in range(N)])

        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model
        # self.stage = stage

    def forward(self, x, stage):
        """
        前向传播
        :param x: 输入
        :param stage: 用于描述此时是训练集的训练过程还是测试集的测试过程  测试过程中均不在加mask机制
        :return: 输出，gate之后的二维向量，step-wise encoder中的score矩阵，channel-wise encoder中的score矩阵，step-wise embedding后的三维矩阵，channel-wise embedding后的三维矩阵，gate
        """
        # step-wise
        # score矩阵为 input， 默认加mask 和 pe
        encoding_1 = self.embedding_channel(x) # x[batchsize, time_step, feature] -> encoding_1[batchsize, time_step, d_model]
        seq1_len = encoding_1.shape[1]
        attn_window_size_1 = int(0.25*seq1_len)
        # print(seq1_len)
        
        
        encoder_list_1 = ModuleList([Encoder(d_model=self.d_model,
                                                  d_hidden=self.d_hidden,
                                                  q=self.q,
                                                  v=self.v,
                                                  h=self.h,
                                                  mask=self.mask,
                                                  dropout=self.dropout,
                                                  device=self.device,
                                                  seq_len=seq1_len,
                                                  attention_window=self.attention_window,
                                                  encoder_type='stepwise') for _ in range(self.N)])
        # for i in encoder_list_1.named_parameters():
        #     print(f"{i[0]} -> {i[1].device}")
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

        for encoder in encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage) 

        # channel-wise
        # score矩阵为channel 默认不加mask和pe
        encoding_2 = self.embedding_input(x.transpose(-1, -2)) # x[batchsize, time_step, feature] -> encoding_1[batchsize, feature, d_model]
        seq2_len = encoding_2.shape[1]
        attn_window_size_2=int(0.5*seq2_len)
        channel_to_gather = encoding_2                         # x[1000, 128, 14]->[1000, 14, 128] -> [1000, 14, 512]

        encoder_list_2 = ModuleList([Encoder(d_model=self.d_model,
                                                  d_hidden=self.d_hidden,
                                                  q=self.q,
                                                  v=self.v,
                                                  h=self.h,
                                                  dropout=self.dropout,
                                                  attention_window=self.attention_window,
                                                  device=self.device,encoder_type='channelwise',seq_len=seq2_len) for _ in range(self.N)])
        
        
        # for i in encoder_list_2.named_parameters():
        #     print(f"{i[0]} -> {i[1].device}")



        for encoder in encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage) # [1000, 14, 512]

        # 三维变二维
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1) # stepwise S [1000, d_model*]
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1) # channelwise C

        # gate
        h = self.gate(torch.cat([encoding_1, encoding_2], dim=-1))
        gate = F.softmax(h,dim=-1)
        g1 = gate[:,0:1]
        g2 = gate[:,1:2]
        encoding = torch.cat([encoding_1 * g1, encoding_2*g2], dim=-1) #y
        # gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        # encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

        # 输出
        output = self.output_linear(encoding)

        return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
