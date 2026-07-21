"""
Regression tests for the annotation defects found by the ninth release-readiness scan: the coding
sequence lookup handing out pandas rows, whose per-field cost dominated the degeneracy annotation, and
``max_sites=0`` annotating the whole input. Kept fast and unmarked so they run in the default suite.
"""

import logging

import numpy as np
import pandas as pd
import pytest

import sfsutils as su
from sfsutils.annotation import DegeneracyAnnotation, _CDSIndex, _CDSRecord
from sfsutils.filtration import CodingSequenceFiltration
from sfsutils.io_handlers import DummyVariant

CDS = pd.DataFrame(
    [
        ('ctg1', 10, 24, '+', 0, 'tx1'),
        ('ctg1', 40, 54, '+', 1, 'tx1'),
        ('ctg1', 45, 50, '-', 2, 'tx2'),
        ('ctg1', 80, 94, '+', 0, 'tx2'),
    ],
    columns=['seqid', 'start', 'end', 'strand', 'phase', 'parent']
)


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

    def _require_gff(self, name: str):
        """
        :param name: The name of the requiring component.
        """
        pass


class _CountingContig:
    """
    Reference contig counting how often a base is read off it.
    """

    def __init__(self, seq: str):
        """
        :param seq: The sequence.
        """
        self.seq = seq
        self.n_reads = 0
        self.id = 'ctg1'

    def __getitem__(self, item):
        """
        :param item: The position.
        :return: The base.
        """
        self.n_reads += 1

        return self.seq[item]


class TestCDSRecord:
    """
    Test the record the coding sequence index hands out.
    """

    def test_record_holds_the_fields_of_the_frame_row(self):
        """The record carries the same values as the row it stands in for."""
        index = _CDSIndex(CDS)

        for row in range(len(CDS)):
            record = index.get(row)

            assert isinstance(record, _CDSRecord)
            assert list(record.values) == list(CDS.iloc[row].values)

            for field in ['seqid', 'start', 'end', 'strand', 'phase', 'parent']:
                assert getattr(record, field) == CDS.iloc[row][field]

    def test_record_rejects_unknown_fields(self):
        """The record holds exactly the fields of a coding sequence."""
        record = _CDSIndex(CDS).get(0)

        with pytest.raises(AttributeError):
            record.attributes = 'Parent=tx1'

    def test_index_without_transcripts_yields_records(self):
        """A frame without a parent column still yields records, with no transcript."""
        index = _CDSIndex(CDS.drop(columns=['parent']))

        assert index.get(0).parent is None

    def test_annotation_agrees_with_pandas_rows(self, monkeypatch):
        """The records reproduce the annotations the pandas rows produced."""
        seq = ''.join('ACGT'[i % 4] for i in range(120))

        def annotate(patched: bool):
            """
            :param patched: Whether to hand out pandas rows instead of records.
            :return: The annotations of every position.
            """
            if patched:
                monkeypatch.setattr(_CDSIndex, 'get', lambda self, row: self.cds.iloc[row])
            else:
                monkeypatch.undo()

            ann = DegeneracyAnnotation()
            ann._handler = _Handler(CDS)
            ann._contig = seq
            ann._fetch_contig = lambda v: None

            out = []
            for pos in range(1, 100):
                v = DummyVariant(ref=seq[pos - 1], pos=pos, chrom='ctg1')
                ann.annotate_site(v)
                out.append(dict(v.INFO))

            return out

        assert annotate(patched=True) == annotate(patched=False)

    def test_filtration_reads_the_records(self):
        """The coding sequence filtration keeps working on the records."""
        f = CodingSequenceFiltration()
        f._setup(_Handler(CDS))

        kept = [pos for pos in range(1, 100) if f.filter_site(DummyVariant(ref='A', pos=pos, chrom='ctg1'))]

        assert kept == list(range(10, 25)) + list(range(40, 55)) + list(range(80, 95))

    def test_reference_base_is_not_read_for_a_discarded_message(self):
        """The debug message is not assembled unless debug messages are emitted."""
        contig = _CountingContig(''.join('ACGT'[i % 4] for i in range(120)))

        def n_reads(level: int) -> int:
            """
            :param level: The log level of the annotation.
            :return: The number of bases read off the contig.
            """
            ann = DegeneracyAnnotation()
            ann._handler = _Handler(CDS)
            ann._contig = contig
            ann._fetch_contig = lambda v: None
            ann._logger.setLevel(level)

            contig.n_reads = 0
            ann.annotate_site(DummyVariant(ref=contig.seq[14], pos=15, chrom='ctg1'))

            return contig.n_reads

        try:
            assert n_reads(logging.WARNING) < n_reads(logging.DEBUG)
        finally:
            DegeneracyAnnotation()._logger.setLevel(logging.NOTSET)


class TestMaxSites:
    """
    Test the site limit of the annotator.
    """

    def test_non_positive_max_sites_is_rejected(self):
        """A limit of zero or less is not a valid number of sites to annotate."""
        for max_sites in [0, -1]:
            with pytest.raises(ValueError, match='max_sites'):
                su.Annotator(
                    source='resources/genome/betula/biallelic.subset.10000.vcf.gz',
                    output='scratch/annotator.max_sites.vcf',
                    max_sites=max_sites
                )

    def test_max_sites_is_honoured(self, tmp_path):
        """The annotator stops after the requested number of sites."""
        out = str(tmp_path / 'out.vcf')

        su.Annotator(
            source='resources/genome/betula/biallelic.subset.10000.vcf.gz',
            output=out,
            max_sites=7
        ).annotate()

        with open(out) as fh:
            assert len([line for line in fh if not line.startswith('#')]) == 7
