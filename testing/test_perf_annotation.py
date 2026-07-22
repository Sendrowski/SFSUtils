"""
Pin the coding sequence lookup of :class:`sfsutils.annotation.DegeneracyAnnotation` down to the
positional index it is built on.

The index answers three questions per CDS transition: the coding sequence reaching the current
position, and the neighbouring coding sequences of the same transcript a codon spanning a CDS
boundary is completed from. A linear scan over the coding sequences of the contig answers the same
questions, and serves as the reference here: the two must agree at every position of every contig.
"""
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from sfsutils.annotation import DegeneracyAnnotation, _CDSIndex
from sfsutils.io_handlers import DummyVariant


class _Handler:
    """
    Stand-in for the file handler, serving coding sequences from a frame held in memory.
    """

    def __init__(self, cds: pd.DataFrame):
        """
        :param cds: The coding sequences.
        """
        self._cds = cds

    @staticmethod
    def get_aliases(chrom: str):
        """
        :param chrom: The contig.
        :return: The aliases of the contig.
        """
        return [chrom]


def _make_cds(records) -> pd.DataFrame:
    """
    Build a coding sequence frame in the layout produced by the GFF handler.

    :param records: Tuples of seqid, start, end, strand, phase and parent.
    :return: The frame, sorted by seqid and start.
    """
    df = pd.DataFrame(records, columns=['seqid', 'start', 'end', 'strand', 'phase', 'parent'])

    return df.sort_values(by=['seqid', 'start']).reset_index(drop=True)


def _scan(on_contig: pd.DataFrame, pos: int):
    """
    Locate the coding sequence and its neighbours by scanning the contig, mirroring the semantics the
    index has to reproduce.

    :param on_contig: The coding sequences of one contig.
    :param pos: The 1-based position.
    :return: The coding sequence, the preceding one and the following one, any of which may be ``None``.
    """
    cds = on_contig[on_contig.end >= pos]

    if cds.empty:
        return None, None, None

    cd = cds.iloc[0]

    if pd.notna(cd.parent):
        neighbours = on_contig[on_contig.parent == cd.parent]
    else:
        neighbours = on_contig

    following = neighbours[neighbours.start > cd.end]
    preceding = neighbours[neighbours.end < cd.start]

    return (
        cd,
        preceding.iloc[-1] if not preceding.empty else None,
        following.iloc[0] if not following.empty else None
    )


def _same(a, b) -> bool:
    """
    Compare two coding sequences.

    :param a: The first coding sequence.
    :param b: The second coding sequence.
    :return: Whether both are absent or hold the same values.
    """
    if a is None or b is None:
        return a is None and b is None

    return list(a.values) == list(b.values)


# a layout with interleaved transcripts on both strands, transcripts nested inside the introns of
# others, coding sequences sharing a start position, and one transcript without a parent
CDS = _make_cds([
    ('ctg1', 10, 24, '+', 0, 'tx1'),
    ('ctg1', 40, 54, '+', 0, 'tx1'),
    ('ctg1', 80, 94, '+', 0, 'tx1'),
    ('ctg1', 10, 30, '-', 0, 'tx2'),
    ('ctg1', 60, 70, '-', 0, 'tx2'),
    ('ctg1', 45, 50, '+', 1, 'tx3'),
    ('ctg1', 96, 99, '-', 2, 'tx3'),
    ('ctg2', 5, 9, '-', 0, 'tx4'),
    ('ctg2', 20, 40, '-', 0, 'tx4'),
    ('ctg2', 22, 26, '+', 0, np.nan),
    ('ctg2', 30, 31, '+', 0, 'tx5'),
    ('ctg3', 1, 3, '+', 0, 'tx6'),
])


@pytest.mark.parametrize('contig', ['ctg1', 'ctg2', 'ctg3'])
def test_index_agrees_with_scan_at_every_position(contig):
    """
    The index reproduces the scan at every position of the contig.
    """
    on_contig = CDS[CDS.seqid == contig]
    index = _CDSIndex(on_contig)

    for pos in range(1, 110):
        cd, cd_prev, cd_next = _scan(on_contig, pos)

        row = index.locate(pos)

        if cd is None:
            assert row is None
            continue

        assert _same(index.get(row), cd)

        parent = index.get(row).parent
        parent = None if pd.isna(parent) else parent

        row_prev = index.locate_prev(row, parent)
        row_next = index.locate_next(row, parent)

        assert _same(None if row_prev is None else index.get(row_prev), cd_prev)
        assert _same(None if row_next is None else index.get(row_next), cd_next)


def test_index_agrees_with_scan_on_random_layouts():
    """
    The index reproduces the scan on randomly drawn layouts.
    """
    rng = np.random.default_rng(42)

    for _ in range(20):
        starts = rng.integers(1, 200, size=30)
        records = [
            (
                'ctg1',
                int(start),
                int(start + rng.integers(1, 30)),
                '+' if rng.random() < 0.5 else '-',
                int(rng.integers(0, 3)),
                f'tx{rng.integers(0, 4)}'
            ) for start in starts
        ]

        on_contig = _make_cds(records)
        index = _CDSIndex(on_contig)

        for pos in range(1, 240):
            cd, cd_prev, cd_next = _scan(on_contig, pos)

            row = index.locate(pos)

            if cd is None:
                assert row is None
                continue

            assert _same(index.get(row), cd)
            assert _same(
                None if index.locate_prev(row, cd.parent) is None else index.get(index.locate_prev(row, cd.parent)),
                cd_prev
            )
            assert _same(
                None if index.locate_next(row, cd.parent) is None else index.get(index.locate_next(row, cd.parent)),
                cd_next
            )


# reference contig, 1-based: 'G' at 13, 'T' at 14, 'T' at 30 and 'C' at 20
CONTIG = ''.join('G' if i == 12 else 'T' if i in (13, 29) else 'C' if i == 19 else 'A' for i in range(60))


def _make_annotation(cds: pd.DataFrame) -> DegeneracyAnnotation:
    """
    Create an annotation backed by the given coding sequences.

    :param cds: The coding sequences.
    :return: The annotation.
    """
    ann = DegeneracyAnnotation()
    ann._handler = _Handler(cds)
    ann._contig = CONTIG
    ann._fetch_contig = lambda v: None  # the reference contig is pre-injected

    return ann


def test_codon_spanning_cds_boundary_uses_same_transcript():
    """
    A codon reaching past the end of its coding sequence is completed from the next coding sequence of
    the same transcript, skipping an unrelated transcript in between.
    """
    cds = _make_cds([
        ('ctg1', 10, 14, '+', 0, 'tx1'),
        ('ctg1', 20, 25, '-', 0, 'tx2'),
        ('ctg1', 30, 40, '+', 1, 'tx1'),
    ])

    ann = _make_annotation(cds)

    v = DummyVariant(ref='T', pos=14, chrom='ctg1')
    ann.annotate_site(v)

    assert ann._cd_next.start == 30
    assert ann._cd_next.parent == 'tx1'

    # third codon base taken from position 30 ('T'), not from the unrelated transcript at position 20 ('C')
    assert v.INFO['Degeneracy_Info'] == '1,+,GTT'
    assert v.INFO['Degeneracy'] == 0


def test_codon_spanning_cds_boundary_backward():
    """
    The same holds on the minus strand, where the codon is completed from the preceding coding sequence.
    """
    cds = _make_cds([
        ('ctg1', 10, 14, '-', 0, 'tx1'),
        ('ctg1', 20, 25, '+', 0, 'tx2'),
        ('ctg1', 30, 40, '-', 0, 'tx1'),
    ])

    ann = _make_annotation(cds)

    v = DummyVariant(ref='T', pos=30, chrom='ctg1')
    ann.annotate_site(v)

    assert ann._cd_prev.start == 10
    assert ann._cd_prev.parent == 'tx1'

    # third codon base taken from position 14, which yields 'TAA'; the unrelated transcript ending at
    # position 25 would yield 'TAT'
    assert v.INFO['Degeneracy_Info'] == '1,-,TAA'


def test_codon_without_same_transcript_neighbour_is_skipped():
    """
    A codon reaching past the end of the only coding sequence of its transcript is not completed from a
    neighbouring transcript.
    """
    cds = _make_cds([
        ('ctg1', 10, 14, '+', 0, 'tx1'),
        ('ctg1', 20, 25, '+', 0, 'tx2'),
    ])

    ann = _make_annotation(cds)

    v = DummyVariant(ref='T', pos=14, chrom='ctg1')
    ann.annotate_site(v)

    assert ann._cd_next.start == ann._pos_mock
    assert v in ann.errors
    assert v.INFO['Degeneracy'] == '.'


def test_lookup_advances_across_contigs_and_survives_rewind():
    """
    Walking several contigs and rewinding in between yields the same coding sequences.
    """
    ann = _make_annotation(CDS)

    walked = []
    for contig in ['ctg1', 'ctg2', 'ctg3']:
        for pos in range(1, 100):
            try:
                ann._fetch_cds(DummyVariant(ref='A', pos=pos, chrom=contig))
                walked.append((contig, pos, ann._cd.start, ann._cd.end))
            except LookupError:
                pass

    assert len(walked) > 50

    ann._rewind()

    again = []
    for contig in ['ctg1', 'ctg2', 'ctg3']:
        for pos in range(1, 100):
            try:
                ann._fetch_cds(DummyVariant(ref='A', pos=pos, chrom=contig))
                again.append((contig, pos, ann._cd.start, ann._cd.end))
            except LookupError:
                pass

    assert walked == again


def test_lookup_cost_does_not_scale_with_genome(monkeypatch):
    """
    The cost per CDS transition is set by the contig, not by the number of coding sequences in the genome.

    The work done is counted rather than timed: the index searched holds the coding sequences of the current
    contig alone, and each transition costs a fixed number of lookups into it, both independent of how many
    coding sequences sit on unrelated contigs.
    """
    counts = Counter()

    for name in ('locate', 'locate_next', 'locate_prev'):
        original = getattr(_CDSIndex, name)

        def counting(self, *args, _name=name, _original=original, **kwargs):
            counts[_name] += 1

            return _original(self, *args, **kwargs)

        monkeypatch.setattr(_CDSIndex, name, counting)

    def work(n_padding: int) -> tuple:
        """
        :param n_padding: The number of coding sequences on unrelated contigs.
        :return: The lookups per CDS transition and the size of the index searched.
        """
        records = [('ctg1', 100 * i + 1, 100 * i + 60, '+', 0, f'tx{i}') for i in range(500)]
        records += [('pad', 10 * i + 1, 10 * i + 5, '+', 0, f'p{i}') for i in range(n_padding)]

        ann = _make_annotation(_make_cds(records))

        sites = [DummyVariant(ref='A', pos=100 * i + 30, chrom='ctg1') for i in range(500)]

        # the first call builds the index for the contig
        ann._fetch_cds(sites[0])

        counts.clear()
        for site in sites:
            ann._fetch_cds(site)

        # the first site still sits on the coding sequence the warm-up call fetched, so it is no transition
        return {name: n / (len(sites) - 1) for name, n in counts.items()}, len(ann._indexes['ctg1'].cds)

    small = work(1000)
    large = work(200000)

    # one lookup of each kind per transition, into an index of the 500 coding sequences of 'ctg1' only
    assert small == ({'locate': 1.0, 'locate_next': 1.0, 'locate_prev': 1.0}, 500)
    assert large == small
