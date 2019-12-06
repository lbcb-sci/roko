import pysam
from Bio import SeqIO
from labels import *
import argparse
from multiprocessing import Pool
import gen
from data import DataWriter

ENCODED_UNKNOWN = encoding[UNKNOWN]

GROUP_SIZE = 10000
NUM_WORKERS = 6
MAX_INS = 3


def generate_regions(ref, ref_name, window=100_000, overlap=300):
    length = len(ref)
    i = 0

    while i < length:
        end = i + window
        yield Region(ref_name, i, min(end, length))

        if end >= length:
            break
        else:
            i = end - overlap


def is_in_region(pos, aligns):
    for a in aligns:
        if a.start <= pos < a.end:
            return True
    return False


def generate_train(args):
    bam_X, bam_Y, ref, region = args

    alignments = get_aligns(bam_Y, ref_name=region.name, start=region.start, end=region.end)
    filtered = filter_aligns(alignments)

    print(f'Finished generating labels for {region.name}:{region.start}-{region.end}.')

    if not filtered:
        print('No alignments.')
        return None

    positions, examples, labels = [], [], []

    for a in filtered:
        pos_labels = dict()
        n_pos = set()

        t_pos, t_labels = get_pos_and_labels(a, ref, region)
        for p, l in zip(t_pos, t_labels):
            if l == ENCODED_UNKNOWN:
                n_pos.add(p)
            else:
                pos_labels[p] = l

        pos_sorted = sorted(list(pos_labels.keys()))
        region_string = f'{region.name}:{pos_sorted[0][0]+1}-{pos_sorted[-1][0]}'

        result = gen.generate_features(bam_X, str(ref), region_string)

        for P, X in zip(*result):
            Y = []
            to_yield = True

            for p in P:
                assert is_in_region(p[0], filtered)

                if p in n_pos:
                    to_yield = False
                    break

                try:
                    y_label = pos_labels[p]
                except KeyError:
                    if p[1] != 0:
                        y_label = encoding[GAP]
                    else:
                        raise KeyError(f'No label mapping for position {p}.')

                Y.append(y_label)

            if to_yield:
                positions.append(P)
                examples.append(X)
                labels.append(Y)

    print(f'Finished generating examples for {region.name}:{region.start}-{region.end}.')
    return region.name, positions, examples, labels


def generate_infer(args):
    bam_X, ref, region = args

    region_string = f'{region.name}:{region.start+1}-{region.end}'
    result = gen.generate_features(bam_X, ref, region_string)

    positions, examples = [], []

    for P, X in zip(*result):
        positions.append(P)
        examples.append(X)

    print(f'Finished generating examples for {region.name}:{region.start}-{region.end}.')
    return region.name, positions, examples, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ref', type=str)
    parser.add_argument('X', type=str)
    parser.add_argument('--Y', type=str, default=None)
    parser.add_argument('o', type=str)
    parser.add_argument('--t', type=int, default=1)
    args = parser.parse_args()

    inference = False if args.Y else True
    size = 0

    with open(args.ref, 'r') as handle:
        refs = [(str(r.id), str(r.seq)) for r in SeqIO.parse(handle, 'fasta')]

    with DataWriter(args.o, inference) as data:
        data.write_contigs(refs)

        func = generate_infer if inference else generate_train

        arguments = []
        for n, r in refs:
            for region in generate_regions(r, n):
                a = (args.X, r, region) if inference else (args.X, args.Y, r, region)
                arguments.append(a)

        print(f'Data generation started, number of jobs: {len(arguments)}.')

        with Pool(processes=args.t) as pool:
            finished = 0
            for result in pool.imap(func, arguments):
                if not result:
                    continue
                c, p, x, y = result
                data.store(c, p, x, y)
                finished += 1

                if finished % 10 == 0:
                    print('Writing to disk started')
                    data.write()
                    print('Writing to disk finished')

            print('Writing to disk started')
            data.write()
            print('Writing to disk finished')


if __name__ == '__main__':
    main()

