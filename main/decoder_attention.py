import torch.nn as nn
import torch.nn.functional as f

from main.common.common import *


class DecoderAttention(nn.Module):

    def __init__(self):
        super(DecoderAttention, self).__init__()

        self.attn = nn.Bilinear(conf.get('dec-hidden-size'), conf.get('dec-hidden-size'), 1, False)

    '''
        :params
            dec_hidden          : B, DH
            pre_dec_hiddens     : B, T, DH
            
        :returns
            ctx_vector          : B, DH
    '''
    def forward(self, dec_hidden, pre_dec_hiddens):
        if pre_dec_hiddens is None:
            ctx_vector = cuda(t.zeros(dec_hidden.size()))
            attention = cuda(t.zeros(dec_hidden.size()))
        else:
            dec_hidden = dec_hidden.unsqueeze(1).expand(-1, pre_dec_hiddens.size(1), -1).contiguous()   # B, T, DH

            score = self.attn(dec_hidden, pre_dec_hiddens).squeeze(2)   # B, T

            # softmax

            attention = f.softmax(score, dim=1)  # B, T

            # context vector

            ctx_vector = t.bmm(attention.unsqueeze(1), pre_dec_hiddens)  # (B, 1, T) * (B, T, DH)  =>  B, 1, DH
            ctx_vector = ctx_vector.squeeze(1)  # B, DH

        return ctx_vector, attention
