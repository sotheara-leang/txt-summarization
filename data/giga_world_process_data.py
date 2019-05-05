import os
import argparse
import collections
import re
import string
import tqdm

ptb_unescape = {'<unk>': '[UNK]'}


def count_samples(file_in):
    counter = 0
    with open(file_in, 'r', encoding='utf-8') as reader:
        for line in tqdm.tqdm(reader):

            if line == '':
                break

            counter += 1

    return counter


def count_max_sample_len(file_name):
    lengths = []

    with open(file_name, 'r', encoding='utf-8') as reader:
        for line in tqdm.tqdm(reader):

            if line == '':
                break

            lengths.append(len(line.split()))

    return max(lengths)


def extract_samples(file_in, start_index, end_index, dir_out, fname):
    counter = 1

    samples = []

    with open(file_in[0], 'r', encoding='utf-8') as art_reader, open(file_in[1], 'r', encoding='utf-8') as sum_reader:
        for article in tqdm.tqdm(art_reader):

            if article == '' or (end_index > 0 and counter > end_index):
                break

            if counter < start_index:
                counter += 1
                continue

            article = article.strip()
            summary = next(sum_reader).strip()

            for abbr, sign in ptb_unescape.items():
                article = article.replace(abbr, sign)
                summary = summary.replace(abbr, sign)

            if article == '' or summary == '':
                continue

            samples.append((article, summary))

            counter += 1

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    _, art_fname = os.path.split(file_in[0])
    _, sum_fname = os.path.split(file_in[1])

    art_output_fname = art_fname if fname is None else fname[0]
    sum_output_fname = sum_fname if fname is None else fname[1]

    samples = sorted(samples, key=lambda sample: len(sample[0]), reverse=False)

    with open(dir_out + '/' + art_output_fname, 'w', encoding='utf-8') as art_writer, \
            open(dir_out + '/' + sum_output_fname, 'w', encoding='utf-8') as sum_writer:
        for sample in samples:
            art_writer.write(sample[0] + '\n')
            sum_writer.write(sample[1] + '\n')


def generate_vocab(files_in, dir_out, fname, max_vocab):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    reach_max_vocab = False
    vocab_counter = collections.Counter()

    for file in files_in:
        with open(file, 'r', encoding='utf-8') as reader:

            # build vocab
            for line in tqdm.tqdm(reader):

                if line == '':
                    break

                tokens = line.split()
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty

                vocab_counter.update(tokens)

                if max_vocab > 0 and len(vocab_counter) >= max_vocab:
                    reach_max_vocab = True
                    break

        if reach_max_vocab is True:
            break

    output_fname = 'vocab.txt' if fname is None else fname

    with open(dir_out + '/' + output_fname, 'w', encoding='utf-8') as writer:
        vocab_counter = sorted(vocab_counter.items(), key=lambda e: e[1], reverse=True)

        for i, element in enumerate(vocab_counter):
            if max_vocab > 0 and i >= max_vocab:
                break

            token = element[0]
            count = element[1]

            writer.write(token + ' ' + str(count) + '\n')


def generate_glove_vocab(files_in, dir_out, fname, max_vocab, glove_file):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    reach_max_vocab = False
    vocab_counter = collections.Counter()

    glove_vocab = get_glove_vocab(glove_file)

    for file in files_in:
        with open(file, 'r', encoding='utf-8') as reader:

            # build vocab
            for line in tqdm.tqdm(reader):

                if line == '':
                    break

                tokens = line.split()

                valid_tokens = []
                for token in tokens:
                    token = token.strip()
                    # if token not in glove_vocab or not valid_token(token):
                    if token not in glove_vocab:
                        continue

                    valid_tokens.append(token)

                vocab_counter.update(valid_tokens)

                if max_vocab > 0 and len(vocab_counter) >= max_vocab:
                    reach_max_vocab = True
                    break

        if reach_max_vocab is True:
            break

    output_fname = 'vocab.txt' if fname is None else fname

    with open(dir_out + '/' + output_fname, 'w', encoding='utf-8') as writer:
        vocab_counter = sorted(vocab_counter.items(), key=lambda e: e[1], reverse=True)

        for i, element in enumerate(vocab_counter):
            if max_vocab > 0 and i >= max_vocab:
                break

            token = element[0]
            count = element[1]

            writer.write(token + ' ' + str(count) + '\n')


def get_glove_vocab(file_in):
    with open(file_in, 'r') as r:
        words = {}
        for line in r:
            line = line.split()
            word = line[0]

            words[word] = ""
        return words


def valid_token(token):
    return token in string.punctuation or \
           (token != ''
            and re.match('^[a-z]+|[a-z]+(-[a-z]+)+|\'[a-z]+|``|\'\'$', token)
            and not token.endswith('#')
            and not token.endswith('.com'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--opt', type=str, default="extract")
    parser.add_argument('--file', nargs="*")
    parser.add_argument('--dir_out', type=str, default="extract")
    parser.add_argument('--max_vocab', type=int, default="-1")
    parser.add_argument('--sindex', type=int, default="1")
    parser.add_argument('--eindex', type=int, default="1000")
    parser.add_argument('--fname', nargs="*")
    parser.add_argument('--glove_file', type=str)

    args = parser.parse_args()

    if args.opt == 'gen_vocab':
        generate_vocab(args.file, args.dir_out, args.fname[0] if args.fname is not None else None, args.max_vocab)
    elif args.opt == 'count':
        print(count_samples(args.file[0]))
    elif args.opt == 'max_len':
        print(count_max_sample_len(args.file[0]))
    elif args.opt == 'extract':
        extract_samples(args.file, args.sindex, args.eindex, args.dir_out, args.fname)
    elif args.opt == 'gen_glove_vocab':
        generate_glove_vocab(args.file, args.dir_out, args.fname[0] if args.fname is not None else None, args.max_vocab, args.glove_file)
