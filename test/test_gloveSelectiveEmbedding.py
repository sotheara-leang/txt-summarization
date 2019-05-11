from unittest import TestCase

from test.common import *
from main.common.util.file_util import FileUtil
from main.common.simple_vocab import SimpleVocab
from main.common.glove.selective_embedding import GloveSelectiveEmbedding


class TestGloveSelectiveEmbedding(TestCase):

    def test_load_emb_file(self):
        vocab = SimpleVocab(FileUtil.get_file_path(conf('vocab-file')), conf('vocab-size'))

        embedding = GloveSelectiveEmbedding(FileUtil.get_file_path('data/raw/glove.6B.50d.txt'), vocab)

        id = t.tensor([1, 2])

        emb = embedding(id)

        print(emb)
