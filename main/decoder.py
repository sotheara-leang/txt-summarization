import torch.nn as nn

from main.common.common import *


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTMCell(conf.get('emb-size'), conf.get('dec-hidden-size'))

    '''
        :params
            y               : B, E
            pre_hidden      : B, DH
            pre_cell        : B, DH
        :returns
            hidden          : B, DH
            cell            : B, DH   
    '''
    def forward(self, y, pre_hidden, pre_cell):
        hidden, cell = self.lstm(y, (pre_hidden, pre_cell))
        return hidden, cell
