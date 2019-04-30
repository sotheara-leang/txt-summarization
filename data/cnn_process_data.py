import argparse
import os
import collections
import tqdm
import string
import re

ptb_unescape = {'<t>': '', '</t>': ''}


def count_samples(file_in):
    counter = 0
    with open(file_in, 'r', encoding='utf-8') as reader:
        while reader.readline() != '':
            counter += 1

    return counter


def extract_samples(file_in, start_index, end_index, dir_out, fname):
    path, filename = os.path.split(file_in)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    counter = 1

    output_fname = filename if fname is None else fname

    with open(file_in, 'r', encoding='utf-8') as reader, open(dir_out + '/' + output_fname, 'w', encoding='utf-8') as writer:
        while counter <= end_index:
            line = reader.readline()

            if line == '':
                break

            if counter < start_index:
                counter += 1
                continue

            for abbr, sign in ptb_unescape.items():
                line = line.replace(abbr, sign)

            line = line.strip()

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


def valid_token(token):
    return token in string.punctuation or \
           (token != ''
            and token != '<t>'
            and token not in ptb_unescape.keys()
            and re.match('^[a-z]+|[a-z]+(-[a-z]+)+|\'[a-z]+$', token))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--opt', type=str, default="extract")
    parser.add_argument('--file', '--names-list', nargs="*")
    parser.add_argument('--dir_out', type=str, default="extract")
    parser.add_argument('--max_vocab', type=int, default="-1")
    parser.add_argument('--sindex', type=int, default="1")
    parser.add_argument('--eindex', type=int, default="1000")
    parser.add_argument('--fname', type=str)

    args = parser.parse_args()

    if args.opt == 'gen-vocab':
        generate_vocab(args.file, args.dir_out, args.fname, args.max_vocab)
    elif args.opt == 'count':
        print(count_samples(args.file[0]))
    elif args.opt == 'extract':
        extract_samples(args.file[0], args.sindex, args.eindex, args.dir_out, args.fname)
