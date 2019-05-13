from unittest import TestCase

from rouge import Rouge

from main.seq2seq import Seq2Seq
from main.common.simple_vocab import *
from main.common.simple_vocab import SimpleVocab
from main.common.util.file_util import FileUtil
from main.data.giga_world import *
from test.common import *


class TestSeq2Seq(TestCase):

    def get_score(self, summary, reference):
        rouge = Rouge()

        summary = summary.split()
        summary = [w for w in summary if w != TK_STOP['word']]

        score = rouge.get_scores(' '.join(summary), reference)[0]["rouge-l"]["f"]

        return score

    def test(self):
        data_loader = GigaWorldDataLoader(FileUtil.get_file_path(conf('train:article-file')),
                                          FileUtil.get_file_path(conf('train:summary-file')), 2)

        vocab = SimpleVocab(FileUtil.get_file_path(conf('vocab-file')), conf('vocab-size'))

        seq2seq = cuda(Seq2Seq(vocab))

        checkpoint = t.load(FileUtil.get_file_path(conf('model-file')))

        seq2seq.load_state_dict(checkpoint['model_state_dict'])

        seq2seq.eval()

        samples = data_loader.read_all()

        article, reference = samples[3]

        summary, attention = seq2seq.evaluate(article)

        score = self.get_score(summary, reference)

        print('>>> article: ', article)
        print('>>> reference: ', reference)
        print('>>> prediction: ', summary)
        print('>>> score: ', score)






