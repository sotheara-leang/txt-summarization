from unittest import TestCase

from test.common import *
from main.decoder_attention import DecoderAttention


class TestDecoderAttention(TestCase):

    def test(self):
        dec_hidden = t.randn(2, 2 * conf('hidden-size'))
        pre_dec_hidden = t.randn(2, 3, 2 * conf('hidden-size'))

        decoder_att = DecoderAttention()

        context_vector, attention = decoder_att(dec_hidden, pre_dec_hidden)

        print(context_vector, context_vector.size())
        print(attention, attention.size())
