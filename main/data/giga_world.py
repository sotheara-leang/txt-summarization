from main.common.dataloader import *
from main.common.memory_dataloader import MemoryDataLoader


class GigaWorldDataLoader(DataLoader):

    def __init__(self, article_file, summary_file, batch_size):
        self.article_file = article_file
        self.summary_file = summary_file

        super(GigaWorldDataLoader, self).__init__(batch_size)

    def reader(self):
        with open(self.article_file, 'r', encoding='utf-8') as art_reader, open(self.summary_file, 'r', encoding='utf-8') as sum_reader:
            while True:
                article = next(art_reader)
                summary = next(sum_reader)

                yield article.strip(), summary.strip()


class GigaWorldMemoryDataLoader(MemoryDataLoader):

    def __init__(self, article_file, summary_file, batch_size):
        self.article_file = article_file
        self.summary_file = summary_file

        super(GigaWorldMemoryDataLoader, self).__init__(batch_size)

    def reader(self):
        with open(self.article_file, 'r', encoding='utf-8') as art_reader, open(self.summary_file, 'r', encoding='utf-8') as sum_reader:
            while True:
                article = next(art_reader)
                summary = next(sum_reader)

                yield article.strip(), summary.strip()



