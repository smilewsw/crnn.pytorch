import torch.nn as nn
import models.utils as utils
from . import resnet

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, ngpu):
        super(BidirectionalLSTM, self).__init__()
        self.ngpu = ngpu

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = utils.data_parallel(
            self.rnn, input, self.ngpu)  # [T, b, h * 2]

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = utils.data_parallel(
            self.embedding, t_rec, self.ngpu)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, ngpu, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        self.ngpu = ngpu
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = resnet.resnet50()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(2048, nh, nh, ngpu),
            BidirectionalLSTM(nh, nh, nclass, ngpu)
        )

    def forward(self, input):
        # conv features
        # print(input.size())
       
        conv = utils.data_parallel(self.cnn, input, self.ngpu)
        # print(conv)
        b, c, h, w = conv.size()
        # print(conv.size())
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = utils.data_parallel(self.rnn, conv, self.ngpu)

        return output
