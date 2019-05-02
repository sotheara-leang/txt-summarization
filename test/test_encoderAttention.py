from unittest import TestCase

from main.common.common import *
from main.encoder_attention import EncoderAttention


class TestEncoderAttention(TestCase):

    def test(self):
        dec_hidden = t.randn(2, 2 * conf.get('hidden-size'))

        enc_hidden = t.randn(2, 3, 2 * conf.get('hidden-size'))

        mask = t.tensor([[1, 1, 1], [1, 1, 0]])

        sum_score = None

        encoder_attention = EncoderAttention()

        context_vector, attention, sum_score = encoder_attention(dec_hidden, enc_hidden, mask, sum_score)

        print(context_vector, context_vector.size())
        print(attention, attention.size())
        print(sum_score, sum_score.size())
