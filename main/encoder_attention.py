import torch.nn as nn

from main.common.common import *


class EncoderAttention(nn.Module):

    def __init__(self):
        super(EncoderAttention, self).__init__()

        self.w_attn = nn.Linear(2 * conf('enc-hidden-size'), conf('dec-hidden-size'), False)

    '''
        :params
            dec_hidden          : B, DH
            enc_hiddens         : B, L, EH
            enc_padding_mask    : B, L
            enc_temporal_score  : B, L

        :returns
            ctx_vector          : B, EH
            attention           : B, L
            enc_temporal_score  : B, L
    '''

    def forward(self, dec_hidden, enc_hiddens, enc_padding_mask, enc_temporal_score):
        score = t.bmm(self.w_attn(enc_hiddens), dec_hidden.unsqueeze(2)).squeeze(2)  # (B, L, DH) *  (B, DH, 1) => (B, L

        # temporal normalization

        exp_score = t.exp(score)

        if enc_temporal_score is None:
            score = exp_score
            enc_temporal_score = exp_score
        else:
            score = exp_score / enc_temporal_score
            enc_temporal_score = enc_temporal_score + exp_score

        # normalization

        score = score * enc_padding_mask.float()

        attention = score / t.sum(score, dim=1).unsqueeze(1)  # B, L

        # context vector

        ctx_vector = t.bmm(attention.unsqueeze(1), enc_hiddens).squeeze(1)  # (B, 1, L) * (B, L, EH)  =>  B, EH

        return ctx_vector, attention, enc_temporal_score

