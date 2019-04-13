import argparse
import os
import collections
import re
import spacy

from main.common.util.file_util import FileUtil

nlp = spacy.load("en_core_web_sm")

ptb_unescape = {'-lrb-': '(', '-rrb-': ')', '-lcb-': '{', '-rcb-': '}', '<t>': '', '</t>': ''}


def count_samples(file_in):
    counter = 0
    with open(file_in, 'r') as reader:
        while reader.readline() != '':
            counter += 1

    return counter


def generate_vocab(files_in, dir_out, fname, max_vocab):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    reach_max_vocab = False
    vocab_counter = collections.Counter()

    for file in files_in:
        with open(file, 'r') as reader:

            # build vocab
            for line in reader:

                if line == '':
                    break

                doc = nlp(line.lower())

                words = []
                for lexeme in doc:

                    if lexeme.is_digit \
                            or lexeme.is_title \
                            or lexeme.like_email \
                            or lexeme.like_url \
                            or lexeme.like_num \
                            or lexeme.is_space \
                            or lexeme.is_space\
                            or lexeme.text in ptb_unescape.keys() \
                            or lexeme.text == '<t>':

                        continue

                    words.append(lexeme.text)

                vocab_counter.update(words)

                if max_vocab > 0 and len(vocab_counter) >= max_vocab:
                    reach_max_vocab = True
                    break

        if reach_max_vocab is True:
            break

    output_fname = 'vocab.txt' if fname is None else fname

    # write vocab
    with open(dir_out + '/' + output_fname, 'w') as writer:
        for i, token in enumerate(vocab_counter):
            if max_vocab > 0 and i >= max_vocab:
                break

            count = vocab_counter[token]
            writer.write(token + ' ' + str(count) + '\n')


def extract_samples(file_in, start_index, end_index, dir_out, fname):
    path, filename = os.path.split(file_in)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    counter = 0

    output_fname = filename if fname is None else fname

    with open(file_in, 'r') as reader, open(dir_out + '/' + output_fname, 'w') as writer:
        while counter <= end_index:
            line = reader.readline()

            if line == '':
                break

            if counter < start_index:
                counter += 1
                continue

            # for article
            delimiter = line.find('--')
            if delimiter != -1:
                line = line[line.find('--') + 2:]

            for abbr, sign in ptb_unescape.items():
                line = line.replace(abbr, sign)

            line = line.strip()

            if line == '':
                continue

            writer.write(line + '\n')

            counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--opt', type=str, default="gen-vocab")
    parser.add_argument('--file', '--names-list', nargs="*")
    parser.add_argument('--dir_out', type=str, default="extract")
    parser.add_argument('--max_vocab', type=int, default="-1")
    parser.add_argument('--sindex', type=int, default="0")
    parser.add_argument('--eindex', type=int, default="999")
    parser.add_argument('--fname', type=str)

    args = parser.parse_args()

    if args.opt == 'gen-vocab':
        generate_vocab([FileUtil.get_file_path('data/raw/cnn-article.txt')], args.dir_out, args.fname, args.max_vocab)
    elif args.opt == 'count':
        print(count_samples(args.file))
    elif args.opt == 'extract':
        extract_samples(args.file[0], args.sindex, args.eindex, args.dir_out, args.fname)
