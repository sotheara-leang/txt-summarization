from unittest import TestCase

from main.common.util.file_util import FileUtil
from main.common.simple_vocab import SimpleVocab
from main.common.glove.embedding import GloveEmbedding
from test.common import *


class TestGloveEmbedding(TestCase):

    def test(self):
        vocab = SimpleVocab(FileUtil.get_file_path(conf('vocab-file')), conf('vocab-size'))

        embedding = GloveEmbedding(FileUtil.get_file_path(conf('emb-file')), vocab)

        id = t.tensor([0])

        emb = embedding(id)

        print(emb)




