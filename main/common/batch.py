import torch as t
from main.common.vocab import *


class Batch(object):

    def __init__(self, articles, articles_len, summaries, summaries_len, oovs):
        self.articles = articles
        self.articles_len = articles_len
        self.summaries = summaries
        self.summaries_len = summaries_len
        self.oovs = oovs

        self.max_ovv_len = max([len(ovv) for ovv in oovs])

        self.oov_extra_zero = t.zeros(self.articles.size(0), self.max_ovv_len)


class BatchInitializer(object):

    def __init__(self, vocab, max_enc_steps):
        self.vocab = vocab
        self.max_enc_steps = max_enc_steps

    def init(self, samples):
        pad_token_id = self.vocab.word2id(PAD_TOKEN)

        articles, summaries = list(zip(*samples))

        articles_len = [len(a.split()) for a in articles]
        summaries_len = [len(s.split()) + 1 for s in summaries]     # 1 for STOP_DECODING

        max_article_len = max(articles_len)
        max_summary_len = max(summaries_len)

        enc_articles = []
        enc_summaries = []
        oovs = []

        # article
        for article in articles:
            art_words = article.split()
            if len(art_words) > self.max_enc_steps:  # truncate
                art_words = art_words[:self.max_enc_steps]

            enc_article, article_oovs = article2ids(art_words, self.vocab)

            while len(enc_article) < max_article_len:
                enc_article.append(pad_token_id)

            enc_articles.append(enc_article)
            oovs.append(article_oovs)

        # summary
        for summary in summaries:
            summary_words = summary.split()
            if len(summary_words) > self.max_enc_steps:  # truncate
                summary_words = summary_words[:self.max_enc_steps]

            enc_summary = summary2ids(summary_words, self.vocab, oovs) + [self.vocab.word2id(STOP_DECODING)]
            while len(enc_summary) < max_summary_len:
                enc_summary.append(pad_token_id)

            enc_summaries.append(enc_summary)

        # covert to tensor
        enc_articles = t.tensor(enc_articles)
        articles_len = t.tensor(articles_len)

        enc_summaries = t.tensor(enc_summaries)
        summaries_len = t.tensor(summaries_len)

        # sort tensor
        articles_len, indices = articles_len.sort(0, descending=True)
        enc_articles = enc_articles[indices]

        return Batch(enc_articles, articles_len, enc_summaries, summaries_len, oovs)
