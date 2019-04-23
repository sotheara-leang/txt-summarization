from main.common.dataloader import *


class GigaWorldDataLoader(DataLoader):

    def __init__(self, article_file, summary_file, batch_size):
        super(GigaWorldDataLoader, self).__init__(batch_size)

        self.article_file = article_file
        self.summary_file = summary_file

    def reader(self):
        with open(self.article_file, 'r', encoding='utf-8￿') as art_reader, open(self.summary_file, 'r', encoding='utf-8￿') as sum_reader:
            while True:
                article = next(art_reader)
                summary = next(sum_reader)

                yield article.strip(), summary.strip()



