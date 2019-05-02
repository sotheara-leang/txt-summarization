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

    with open(file_name, 'r') as reader:
        for line in tqdm.tqdm(reader):

            if line == '':
                break

            lengths.append(len(line.split()))

    return max(lengths)


def extract_samples(file_in, start_index, end_index, dir_out, fname):
    path, filename = os.path.split(file_in)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    counter = 1

    output_fname = filename if fname is None else fname

    with open(file_in, 'r', encoding='utf-8') as reader, open(dir_out + '/' + output_fname, 'w', encoding='utf-8') as writer:
        for line in tqdm.tqdm(reader):

            if line == '' or counter > end_index:
                break

            if counter < start_index:
                counter += 1
                continue

            line = line.strip()

            for abbr, sign in ptb_unescape.items():
                line = line.replace(abbr, sign)

            if line == '':
                continue

            writer.write(line + '\n')

            counter += 1


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

                valid_tokens = []
                for token in tokens:
                    token = token.strip()
                    if not valid_token(token):
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
                    if token not in glove_vocab or not valid_token(token):
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
    parser.add_argument('--file', '--names-list', nargs="*")
    parser.add_argument('--dir_out', type=str, default="extract")
    parser.add_argument('--max_vocab', type=int, default="-1")
    parser.add_argument('--sindex', type=int, default="1")
    parser.add_argument('--eindex', type=int, default="1000")
    parser.add_argument('--fname', type=str)
    parser.add_argument('--glove_file', type=str)

    args = parser.parse_args()

    if args.opt == 'gen_vocab':
        generate_vocab(args.file, args.dir_out, args.fname, args.max_vocab)
    elif args.opt == 'count':
        print(count_samples(args.file[0]))
    elif args.opt == 'max_len':
        print(count_max_sample_len(args.file[0]))
    elif args.opt == 'extract':
        extract_samples(args.file[0], args.sindex, args.eindex, args.dir_out, args.fname)
    elif args.opt == 'gen_glove_vocab':
        generate_glove_vocab(args.file, args.dir_out, args.fname, args.max_vocab, args.glove_file)
