from unittest import TestCase

from test.common import *
from main.encoder import Encoder


class TestEncoder(TestCase):

    def test(self):
        x = t.randn(2, 3, conf('hidden-size'))
        seq_len = t.FloatTensor([3, 3])

        print(x)
        print(seq_len)

        encoder = Encoder()

        output, (h_n, c_n) = encoder(x, seq_len)

        print(output, output.size())

        print(h_n, h_n.size())
        print(c_n, c_n.size())

