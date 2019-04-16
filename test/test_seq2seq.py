from unittest import TestCase

from main.common.common import *
from main.seq2seq import Seq2Seq
from main.common.simple_vocab import SimpleVocab
from main.common.util.file_util import FileUtil
from main.data.giga import *


class TestSeq2Seq(TestCase):

    def test(self):
        data_loader = GigaDataLoader(FileUtil.get_file_path(conf.get('train:article-file')),
                                         FileUtil.get_file_path(conf.get('train:summary-file')), 2)

        vocab = SimpleVocab(FileUtil.get_file_path(conf.get('vocab-file')), conf.get('vocab-size'))

        seq2seq = cuda(Seq2Seq(vocab))

        checkpoint = t.load(FileUtil.get_file_path(conf.get('model-file')))

        seq2seq.load_state_dict(checkpoint['model_state_dict'])

        seq2seq.eval()

        samples = data_loader.read_all()

        article, reference = samples[3]

        summary = seq2seq.summarize(article)

        print('>>> article: ', article)
        print('>>> reference: ', reference)
        print('>>> prediction: ', summary)





