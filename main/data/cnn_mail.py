import glob
import struct
from tensorflow.core.example import example_pb2

from main.common.dataloader import *


class CNNDataLoader(DataLoader):

    def __init__(self, article_file, summary_file, batch_size):
        self.article_file = article_file
        self.summary_file = summary_file

        super(CNNDataLoader, self).__init__(batch_size)

    def reader(self):
        with open(self.article_file, 'r', encoding='utf-8') as art_reader, open(self.summary_file, 'r', encoding='utf-8') as sum_reader:
            while True:
                article = next(art_reader)
                summary = next(sum_reader)

                article = article.strip()
                summary = summary.strip()

                if article == '' or summary == '':
                    continue

                yield article, summary


class CNNProcessedDataLoader(DataLoader):

    def __init__(self, data_path, batch_size, file_num=None):
        self.logger = logger(self)

        self.data_path = data_path
        self.file_num = file_num

        super(CNNProcessedDataLoader, self).__init__(batch_size)

    def reader(self):
        file_list = glob.glob(self.data_path)

        file_list = sorted(file_list)

        for idx, f in enumerate(file_list):
            if self.file_num is not None and idx > self.file_num - 1:
                break

            self.logger.info('reading file: %s', f)

            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes:
                    self.logger.info('done reading file: %s', f)

                    break  # finished reading this file

                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]

                example = example_pb2.Example.FromString(example_str)

                yield self.get_data(example)

    def get_data(self, example):
        ptb_unescape = {'<t>': '', '</t>': ''}

        try:
            article_text = example.features.feature['article'].bytes_list.value[0]  # the article text was saved under the key 'article' in the data files
            abstract_text = example.features.feature['abstract'].bytes_list.value[0]  # the abstract text was saved under the key 'abstract' in the data files
            article_text = article_text.decode()
            abstract_text = abstract_text.decode()

            for abbr, sign in ptb_unescape.items():
                abstract_text = abstract_text.replace(abbr, sign)

            return article_text, abstract_text

        except ValueError as e:
            self.logger.error('Failed to get article or abstract from example')
            raise e
