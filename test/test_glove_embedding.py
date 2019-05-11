from unittest import TestCase

from test.common import *
from main.common.batch import *
from main.common.util.file_util import FileUtil
from main.common.simple_vocab import SimpleVocab
from main.common.glove.embedding import GloveEmbedding
from main.common.glove.selective_embedding import GloveSelectiveEmbedding


class TestGloveEmbedding(TestCase):

    def test(self):
        emb = GloveEmbedding(FileUtil.get_file_path('data/extract/embedding'))

    def test_glove_selective_embedding(self):
        vocab = SimpleVocab(FileUtil.get_file_path(conf('vocab-file')), conf('vocab-size'))

        embedding = GloveSelectiveEmbedding(FileUtil.get_file_path(conf('emb-file')), vocab)

        id = t.tensor([0])

        emb = embedding(id)

        print(emb)


