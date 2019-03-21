import torch as t
import torch.nn as nn
import torch.nn.functional as f

from main.common.common import *


class EncoderAttention(nn.Module):

    def __init__(self):
        super(EncoderAttention, self).__init__()

        self.attn = nn.Bilinear(2 * conf.get('hidden-size'), 2 * conf.get('hidden-size'), 1, False)

    '''
        :param
            dec_hidden   : B, 2H
            enc_hiddens  : B, L, 2H
            sum_score    : B, L
        
        :return
            ctx_vector      : B, 2H
            att_dist        : B, L
            temporal_score  : B, L
    '''
    def forward(self, dec_hidden, enc_hiddens, temporal_score):
        dec_hidden = dec_hidden.unsqueeze(1).expand(-1, enc_hiddens.size(1), -1).contiguous()  # B, L, 2H

        score = self.attn(dec_hidden, enc_hiddens).squeeze(2)   # B, L

        # temporal normalization

        exp_score = t.exp(score)    # B, L
        if temporal_score is None:
            score = exp_score
            temporal_score = exp_score
        else:
            score = exp_score / temporal_score
            temporal_score += exp_score

        # normalization

        normalization_factor = score.sum(1, keepdim=True)   # B, L
        att_dist = score / normalization_factor             # B, L

        # context vector

        ctx_vector = t.bmm(att_dist.unsqueeze(1), enc_hiddens)  # B, 1, L * B, L, 2H  =>  B, 1, 2H
        ctx_vector = ctx_vector.squeeze(1)  # B, 2*H

        return ctx_vector, att_dist, temporal_score
