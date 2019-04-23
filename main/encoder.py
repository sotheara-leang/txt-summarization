import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as f

from main.common.common import *


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(conf.get('emb-size'), conf.get('enc-hidden-size'), num_layers=1, batch_first=True, bidirectional=True)

    '''
        :param
            x       : B, L, E
            x_len   : B
            
        :return
            outputs : B, L, 2EH
            hidden  : 2, B, EH
            cell    : 2, B, EH
    '''
    def forward(self, x, x_len):
        packed_x = rnn.pack_padded_sequence(x, x_len, batch_first=True)

        # outputs   : B, L, 2H
        # hidden    : 2, B, H
        # cell      : 2, B, H
        outputs, (hidden, cell) = self.lstm(packed_x)

        outputs, _ = rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = outputs.contiguous()

        return outputs, (hidden, cell)
