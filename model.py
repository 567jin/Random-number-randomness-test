import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import random
import time
import math


class Conv1D_BND_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1D_BND_Block, self).__init__()
        self.padding = nn.ConstantPad1d(
            (int((kernel_size-1)/2), int(kernel_size/2)), 0)
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X):
        X = self.padding(X)
        X = self.conv1d(X)
        return X


class Layer(nn.Module):
    def __init__(self, layer_parameters):
        super(Layer, self).__init__()
        self.convs = nn.ModuleList()
        for i in layer_parameters:
            conv_bn = Conv1D_BND_Block(i[0], i[1], i[2])
            self.convs.append(conv_bn)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.35)

    def forward(self, x):
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class OS_CNN(nn.Module):
    def __init__(self, filter_sizes=[1, 2, 3, 5, 7, 11, 13, 17, 19, 23]):
        super(OS_CNN, self).__init__()
        self.embedding = nn.Embedding(256, 32)
        self.layer_parameters = []
        self.layer_last = [[640, 256, 1], [640, 256, 2]]
        for fs in filter_sizes:
            self.layer_parameters.append([32, 64, fs])

        layer1 = Layer(layer_parameters=self.layer_parameters)

        self.layer_parameters = []
        for fs in filter_sizes:
            self.layer_parameters.append([640, 64, fs])

        layer2 = Layer(layer_parameters=self.layer_parameters)
        layer3 = Layer(layer_parameters=self.layer_last)
        self.net = nn.Sequential(layer1, layer2, layer3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 256)
        self.init_params()

    def init_params(self):
        # 使用自定义初始化函数初始化权重
        nn.init.uniform_(self.embedding.weight, a=0, b=1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        # print(x.shape)  # 32*32*100
        x = self.net(x)
        # print(x.shape)  # 32*512*100
        x = self.pool(x).squeeze()
        # print(x.shape) # 32*512
        x = self.fc(x)
        # print(x.shape)  # 32*256
        return x


class OS_CNN_Trans(nn.Module):  # 在OS_CNN的基础上增加一层transformer层 并且序列长度变为128
    def __init__(self, filter_sizes=[1, 2, 3, 5, 7, 11, 13, 17, 19, 23]):
        super(OS_CNN_Trans, self).__init__()
        self.embedding = nn.Embedding(256, 32)
        self.layer_parameters = []
        self.layer_last = [[640, 256, 1], [640, 256, 2]]
        for fs in filter_sizes:
            self.layer_parameters.append([32, 64, fs])

        layer1 = Layer(layer_parameters=self.layer_parameters)
        layer3 = Layer(layer_parameters=self.layer_last)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=16,
            dim_feedforward=256,
            dropout=0.2,
            activation="relu",
            batch_first=True)
        self.net = nn.Sequential(layer1, layer3)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=1)  # 可以增加归一化层
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 256)
        self.init_params()

    def init_params(self):
        # 使用自定义初始化函数初始化权重
        nn.init.uniform_(self.embedding.weight, a=0, b=1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        # print(x.shape)  # 1024*32*128 >>>[batch_size, embed_num, seq_len]
        x = self.net(x)
        # print(x.shape)  # 1024*512*128
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)
        x = self.pool(x).squeeze()
        # print(x.shape)  # 1024*512
        x = self.fc(x)
        # print(x.shape)  # 1024*256
        return x


class CNN_Trans(nn.Module):
    """
    Input:
        X: (batch_size, n_channel, n_length)
        Y: (batch_size)
    Output:
        out: (batch_size)
    Pararmetes:
        n_classes: number of classes
    """

    def __init__(self, in_channels=32, out_channels=64, n_len_seg=128, n_classes=256):
        super(CNN_Trans, self).__init__()

        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.embedding = nn.Embedding(256, in_channels)

        # (batch, channels, length)
        self.cnn = nn.Conv1d(in_channels=self.in_channels,
                             out_channels=self.out_channels,
                             kernel_size=17,
                             stride=2)
        # (batch, channels, length)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.out_channels,
            nhead=8,
            dim_feedforward=128,
            dropout=0.2,
            activation="relu",
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=5)  # 可以增加归一化层
        self.dense = nn.Linear(out_channels, n_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # 64*32*100
        # n_channel=32  n_length=100(输入的长度)  n_len_seg=100(分割的长度)  当二者相等时 表示不分割
        self.n_channel, self.n_length = x.shape[-2], x.shape[-1]
        assert (self.n_length % self.n_len_seg ==
                0), "Input n_length should divided by n_len_seg"

        # 输入进行分段处理 可以减少计算复杂度 并人为将时序数据中的长期依赖关系转化为局部依赖关系 有助于提取局部特征
        self.n_seg = self.n_length // self.n_len_seg   # 这里由于输出的序列不长 所以n_seg = 1

        out = x
        # print(out.shape)  # 64*32*100

        # (batch_size, n_channel, n_length) -> (batch_size, n_length, n_channel)
        out = out.permute(0, 2, 1)
        # print(out.shape)  # 64*100*32

        # (batch_size, n_length, n_channel) -> (n_samples*n_seg, n_len_seg, n_channel)
        out = out.view(-1, self.n_len_seg, self.n_channel)
        # print(out.shape)  # 64*100*32

        # (batch_size*n_seg, n_len_seg, n_channel) -> (batch_size*n_seg, n_channel, n_len_seg)
        out = out.permute(0, 2, 1)
        # print(out.shape)  # 64*32*100

        # cnn
        out = self.cnn(out)
        # print(out.shape)  # 64*64*43

        # global avg, (n_samples*n_seg, out_channels)
        out = out.mean(-1)  # 在最后一个维度求平均
        # print(out.shape)  # 64*64

        out = out.view(-1, self.n_seg, self.out_channels)
        # print(out.shape)  # 64*1*64

        out = self.transformer_encoder(out)
        # print(out.shape)  # 64*1*64

        out = out.mean(-2)
        # print(out.shape)  # 64*64

        out = self.dense(out)
        # print(out.shape)  # 64*256

        return out
# model = CNN_Trans().cuda()
# print(model)
# parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # 打印模型参数量
# print('Total parameters: {}'.format(parameters))
# x = torch.randint(0,256,size=(512,128), dtype=torch.long).cuda()
# start_time = time.time()
# out = model(x)
# end_time = time.time()
# print(end_time-start_time)
# max_index = torch.argmax(out, dim=1)
# print(out.shape,out)
# print(max_index.shape,max_index)


class PositionalEmbedding2FCN(nn.Module):
    def __init__(self, channels, seq_len=512):
        super(PositionalEmbedding2FCN, self).__init__()

        # 32*512
        pe = torch.zeros(seq_len, channels).float()
        pe.require_grad = False  # 位置编码是固定的 不需要参与训练

        # 创建了一个形状为 (max_len, 1) 的位置张量 position，表示序列的位置。
        position = torch.arange(0, seq_len).float().unsqueeze(1)
        # print("position的形状: ", position.shape)  # (512,1)
        # 创建了一个形状为 (d_model/2,) 的除法项 div_term，用于计算位置编码中的除法项。
        div_term = (torch.arange(0, channels, 1).float()
                    * -(math.log(10000.0) / channels)).exp()   # 16,
        # print("div_term的形状: ", div_term.shape)

        #  (512,1)*(16,)=512*16   不匹配pe[:,0::2]的形状(32,256)
        pe[:, 0::2] = torch.sin((position * div_term))[0, 0::2]  # 偶数位置的编码
        pe[:, 1::2] = torch.cos((position * div_term))[1, 1::2]  # 奇数位置的编码

        pe = pe.unsqueeze(0)
        # print("pe的形状为: ", pe.shape)
        self.register_buffer('pe', pe)  # 1,32,512

    def forward(self, x):
        # 返回位置编码张量 pe 的子集，截取前 x 的长度部分。这样，每个输入序列的位置都会与其对应的位置编码进行相加，以提供关于位置信息的表示。   这里返回的是x的位置编码 在使用时 还需要加上x  pe的shape是[1,in_channels,max_len]
        return self.pe[:, :x.size(1)]




class FCN(nn.Module):
    """
    embedding_dim: 词嵌入的维度 一般是长度越短嵌入维度越低 即二进制编码中2^3=8 即只需要三位二进制就可以编码长度为8的序列
    p: Dropout的概率
    is_pe: 是否需要位置嵌入 默认False
    """

    def __init__(self, embedding_dim, p=0.1, is_pe=False, is_wordEmbedding=False):
        super(FCN, self).__init__()
        # int到float的转换 利用WordEmbedding nn.Embedding的两个参数 num_embeddings表示索引号最大值 在NLP中指的是字典中字符个数 在随机数序列预测这里指的是种类 8位编码就是有256, embedding_dim 嵌入的维度即一个词用多长的向量表示 根据二进制表示法 一般是1/4句子长度就很足够了 比如256 只需要8位二进制就可以完全编码
        if is_wordEmbedding:
            self.embedding = nn.Embedding(
                num_embeddings=256, embedding_dim=embedding_dim, dtype=torch.float32, device="cuda")
        else:
            self.embedding = None
            embedding_dim = 1  # 不进行词嵌入 通道数为1

        if is_pe:
            self.pe = PositionalEmbedding2FCN(embedding_dim)
        else:
            self.pe = None

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=64,
                      kernel_size=3, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=5, padding="same"),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Conv1d(in_channels=256, out_channels=512,
                      kernel_size=5, padding="same"),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Conv1d(in_channels=512, out_channels=256,
                      kernel_size=3, padding="same"),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p),

            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, padding="same"),
            nn.BatchNorm1d(256)
        )

        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        # self.init_params()

    def init_params(self):
        # 使用自定义初始化函数初始化词嵌入的权重
        nn.init.uniform_(self.embedding.weight, a=0, b=1)

    def forward(self, x):
        if self.embedding != None:
            x = self.embedding(x)   # 128*100*32
        else:
            x = x.float()  # 因为传入的是long类型 torch训练需要float类型
            x = x.unsqueeze(2)  # 不使用wordEmbedding 为了能使用CNN处理 需要额外一个维度
        # print("词嵌入后的x形状: ", x.shape)
        if self.pe != None:
            x = x + self.pe(x)
        # 将输入张量的维度顺序转换为 (batch_size, in_channels, max_length)
        x = x.permute(0, 2, 1)
        # print("转置后x的形状: ", x.shape)  # 128*32*100
        x = self.conv(x)
        x = self.globalavgpool(x)
        # print(x.shape)  # 32*256*1
        x = x.squeeze(2)
        return x


# model = FCN(embedding_dim=32).cuda()
# print(model)
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Total parameters: ", total_params)

# x = torch.randint(0, 256, size=(128, 100), dtype=torch.long).cuda()
# start_time = time.time()
# out = model(x)
# end_time = time.time()
# print(end_time-start_time)
# max_index = torch.argmax(out, dim=1)
# print(out.shape, out)
# print(max_index.shape, max_index)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.embedding = nn.Embedding(
            256, input_size, dtype=torch.float32, device="cuda")
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        # print(inputs.shape)  # 64 1 784
        # input should have dimension (N, C, L)  输出也是 N C L
        inputs = self.embedding(inputs)  # 512*100*32
        inputs = inputs.permute(0, 2, 1)  # 512*32*100
        y1 = self.tcn(inputs)
        # print(y1.shape)  # 64 25 784
        o = self.linear(y1[:, :, -1])  # 表示将y1最后一个时间步的输出作为线性层的输入
        # print(o.shape)  # 64 10
        # return o   # 输出 维度 [batch_size,预测的num_class类别]
        return o


# model = TCN(input_size=32, output_size=256, num_channels=[128, 128, 128, 128],
#             kernel_size=3, dropout=0.2).cuda()
# print(model)
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Total parameters: ", total_params)

# x = torch.randint(0, 256, size=(512, 100), dtype=torch.long).cuda()
# start_time = time.time()
# out = model(x)
# end_time = time.time()
# print(end_time-start_time)
# max_index = torch.argmax(out, dim=1)
# print(out.shape, out)
# print(max_index.shape, max_index)
