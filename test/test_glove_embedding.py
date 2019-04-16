from unittest import TestCase

from main.common.util.file_util import FileUtil
from main.common.glove.embedding import GloveEmbedding


class TestGloveEmbedding(TestCase):

    def test(self):
        emb = GloveEmbedding(FileUtil.get_file_path('data/extract/embedding'))




