import pysam
import itertools
from collections import namedtuple
from typing import Type

GAP = '*'
UNKNOWN = 'N'
ALPHABET = 'ACGT' + GAP + UNKNOWN
encoding = {v: i for i, v in enumerate(ALPHABET)}
decoding = {v: k for k, v in encoding.items()}

AlignPos = namedtuple('AlignPos', ('qpos', 'qbase', 'rpos', 'rbase'))
Region = namedtuple('Region', ('name', 'start', 'end'))


class TargetAlign:
    def __init__(self, align, start, end, keep=True):
        self.align = align
        self.start = start
        self.end = end
        self.keep = keep


def get_aligns(bam, ref_name=None, start=0, end=None):
    """This function filters and sort aligns for the given bam file.

    Function removes secondary and unmapped aligns and sorts them by starting position.

    :param bam: A string representing path to a bam file
    :param ref_name: A string representing the name of a reference sequence
    :param start: An integer representing the start of a region
    :param end: An integer represeting the end of a region
    :return: A list of filtered and sorted aligns, wrapped in TargetAlign class
    """

    filtered = []

    with pysam.AlignmentFile(bam, 'rb', index_filename=bam + '.bai') as f:
        for r in f.fetch(ref_name, start, end):
            if r.reference_name != ref_name:
                raise ValueError

            if r.reference_end <= start or r.reference_start >= end:
                continue

            if not r.is_unmapped and not r.is_secondary:
                filtered.append(TargetAlign(r, r.reference_start, r.reference_end, True))

    filtered.sort(key=lambda e: e.align.reference_start)
    return filtered


def get_overlap(first, second):
    if second.start < first.end:
        return second.start, first.end
    else:
        return None


def filter_aligns(aligns, len_threshold=2., ol_threshold=0.5, min_len=1000, start=0, end=None):
    """ This function filters aligns based on they length and overlap ratio.

    This function removes or clips the given sequences based on their length and overlap ratio. Length ratio is
    calculated as a ratio between a longer sequence and a shorter sequence. Overlap ratio is calculated as a ratio
    between an overlap and shorter sequence. Based on this two ratios we distinguish for different cases:

    1) LEN_RATIO < LEN_THRESHOLD and OL_RATIO >= OL_THRESHOLD: both sequences are removed
    2) LEN_RATIO < LEN_THRESHOLD and OL_RATIO < OL_THRESHOLD: the start of the second sequence becomes the end of the
    first sequence and vice-versa, removing ambiguity in the process
    3) LEN_RATIO >= LEN_THRESHOLD and OL_RATIO >= OL_THRESHOLD: only the shorter sequence is removed
    4) LEN_RATIO >= LEN_THRESHOLD and OL_RATIO < OL_THRESHOLD: the end of the longer sequence becomes the start for
    the shorter one

    :param aligns: A list of the TargetAlign objects
    :param len_threshold: A float representing the threshold for the length ratio
    :param ol_threshold: A float representing the threshold for the overlap fraction
    :param min_len: An integer representing the minimal alignment length

    :param start: An integer representing the region start
    :param end: An integer representing the region end
    :return: A list of the filtered and clipped TargetAlign objects
    """

    for i, j in itertools.combinations(aligns, 2):
        first, second = sorted((i, j), key=lambda r: r.align.reference_start)

        ol = get_overlap(first, second)
        if ol is None:
            continue
        ol_start, ol_end = ol

        shorter, longer = sorted((i, j), key=lambda r: r.align.reference_length)
        len_ratio = longer.align.reference_length / shorter.align.reference_length
        ol_fraction = (ol_end - ol_start) / shorter.align.reference_length

        if len_ratio < len_threshold:
            if ol_fraction >= ol_threshold:
                shorter.keep = False
                longer.keep = False
            else:
                first.end = ol_start
                second.start = ol_end
        else:
            if ol_fraction >= ol_threshold:
                shorter.keep = False
            else:
                second.start = ol_end

        if start > 0 or end is not None:
            for a in aligns:
                if start > 0:
                    a.start = max(start, a.start)
                if end is not None:
                    a.end = min(end, a.end)

    filtered = [a for a in aligns if (a.keep and a.end - a.start >= min_len)]
    filtered.sort(key=lambda e: e.start)
    return filtered


def get_pairs(align, ref):
    """This function returns AlignPos information for the given alignment.

    This function yields the pair of (POS, BASE) for the reference and the read.

    :param align: An align
    :param ref: Reference
    :return: An AlignPos object containing the alignment information for the specific position
    """

    query = align.query_sequence
    if query is None:
        raise StopIteration()

    for qp, rp in align.get_aligned_pairs():
        rb = ref[rp] if rp is not None else None
        qb = query[qp] if qp is not None else None
        yield AlignPos(qp, qb, rp, rb)


def get_pos_and_labels(align: TargetAlign, ref, region):
    """Retruns the positions and the labels for the specific alignment.

    This function returns the positions represented as (POS, INS_NUM) pair and the corresponding labels represented
    as BASE

    :param align: An alignment
    :param ref: Reference
    :return: Two list containing positions and labels for the specified alignment
    """

    start, end = region.start, region.end
    if start is None:
        start = 0
    if end is None:
        end = float('inf')
    start, end = max(start, align.start), min(end, align.end)

    all_pos = []
    all_labels = []

    pairs = get_pairs(align.align, ref)
    cur_pos, ins_count = None, 0

    def p(e):
        return e.rpos is None or (e.rpos < start)

    for pair in itertools.dropwhile(p, pairs):
        if (pair.rpos == align.align.reference_end or
                (pair.rpos is not None and pair.rpos >= end)):
            break

        if pair.rpos is None:
            ins_count += 1
        else:
            ins_count = 0
            cur_pos = pair.rpos
        pos = (cur_pos, ins_count)
        all_pos.append(pos)

        label = pair.qbase.upper() if pair.qbase else GAP
        try:
            label = encoding[label]
        except KeyError:
            label = encoding[UNKNOWN]

        all_labels.append(label)

    return all_pos, all_labels
