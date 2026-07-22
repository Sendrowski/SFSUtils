"""
INFO encoding and contig lengths of the VCF-Zarr stores our writer produces. The type of an INFO field
is taken from the chunk it first appears in, so a chunk carrying no value at all for a field, and the
promotion of a field to a wider encoding, are the cases that decide whether absence survives the round
trip. The declared length of a contig is the region a spectrum extrapolates over, so the store has to
record what the source declares rather than the span of the variants that reach it.
"""
import random

import numpy as np
import pytest

from sfsutils.io_handlers import Variant, VariantWriter, ZarrVariantReader, ZarrVariantWriter

SAMPLES = ["s1"]


def _variant(pos, info):
    """A biallelic phased SNP at ``pos`` carrying ``info``."""
    return Variant(ref="A", pos=pos, chrom="1", gt_bases=["A|T"], alt=["T"], is_snp=True, info=dict(info))


def _roundtrip(path, infos, chunk=None):
    """Write one variant per INFO dict and read the INFO values back."""
    writer = ZarrVariantWriter(str(path), samples=SAMPLES, seqnames=["1"], info_ancestral="AA")

    if chunk is not None:
        writer._variant_chunk = chunk

    for i, info in enumerate(infos):
        writer.write(_variant(i + 1, info))

    writer.close()

    return [dict(variant.INFO) for variant in ZarrVariantReader(str(path))]


def test_field_absent_on_every_variant_of_its_first_chunk(tmp_path):
    """A field whose only values are the VCF missing marker, as DegeneracyAnnotation writes at a
    non-coding site, is written without a usable value and reads back as absent."""
    read = _roundtrip(tmp_path / "empty.vcz", [{"Degeneracy": "."}, {"Degeneracy": "."}])

    assert [entry.get("Degeneracy") for entry in read] == [None, None]


def test_field_absent_on_every_variant_of_a_later_chunk(tmp_path):
    """An integer field no variant of a later chunk carries widens to the encoding that can mark an
    absent value, keeping the integers already written."""
    infos = [{"DP": 11 + i} for i in range(4)] + [{}] * 4

    read = _roundtrip(tmp_path / "late.vcz", infos, chunk=4)

    assert [entry.get("DP") for entry in read] == [11, 12, 13, 14, None, None, None, None]


def test_flag_promoted_to_string_keeps_an_unset_flag_absent(tmp_path):
    """A Flag widened to a string by a later chunk carrying a string: the variants that never carried
    the flag stay absent instead of reading back as the string 'False'."""
    infos = [{"F": True}, {}, {"F": True}, {"F": "x"}, {"F": "x"}, {"F": "x"}]

    read = _roundtrip(tmp_path / "flag.vcz", infos, chunk=3)

    assert [entry.get("F") for entry in read] == ["True", None, "True", "x", "x", "x"]


def test_integer_encoder_marks_an_absent_value(tmp_path):
    """The integer branch of the encoder marks an absent value with the sentinel the reader maps back
    to absent rather than failing on it."""
    writer = ZarrVariantWriter(str(tmp_path / "int.vcz"), samples=SAMPLES, seqnames=["1"])

    encoded = writer._encode([1, writer._missing, 3], "int")

    assert list(encoded) == [1, -1, 3]


def test_declared_contig_length_survives_the_round_trip(tmp_path):
    """A store written from a source declaring a contig length records that length, not the last
    position written, so the region a spectrum extrapolates over is not understated."""
    reader = _FakeReader(seqnames=["chr1", "chr2"], lengths={"chr1": 1_000_000})

    writer = VariantWriter.open(str(tmp_path / "declared.vcz"), reader)
    writer.write(_variant_on("chr1", 49800))
    writer.write(_variant_on("chr2", 700))
    writer.close()

    assert ZarrVariantReader(str(tmp_path / "declared.vcz")).contig_lengths == {"chr1": 1_000_000,
                                                                               "chr2": 700}


def test_undeclared_contig_length_falls_back_to_the_observed_span(tmp_path):
    """A source declaring no lengths leaves the store carrying the last position written, which the
    reader documents as a lower bound."""
    writer = ZarrVariantWriter(str(tmp_path / "observed.vcz"), samples=SAMPLES, seqnames=["1"])
    writer.write(_variant(500, {}))
    writer.close()

    assert ZarrVariantReader(str(tmp_path / "observed.vcz")).contig_lengths == {"1": 500}


def test_declared_length_reaches_the_store_through_a_vcf_source(tmp_path):
    """The ##contig header of a VCF input reaches the output store."""
    pytest.importorskip("cyvcf2")

    from sfsutils.filtration import Filterer, SNPFiltration

    vcf = tmp_path / "in.vcf"
    with open(vcf, "w") as handle:
        handle.write("##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000000>\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n")
        for i in range(50):
            handle.write(f"chr1\t{100 + i * 100}\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\n")

    out = str(tmp_path / "sub.vcz")
    Filterer(source=str(vcf), output=out, filtrations=[SNPFiltration()]).filter()

    assert ZarrVariantReader(out).contig_lengths == {"chr1": 1_000_000}


def _variant_on(chrom, pos):
    """A biallelic phased SNP at ``pos`` on ``chrom``."""
    return Variant(ref="A", pos=pos, chrom=chrom, gt_bases=["A|T"], alt=["T"], is_snp=True, info={})


class _FakeReader:
    """A stand-in cyvcf2 VCF exposing only what the writer takes from an input."""

    def __init__(self, seqnames, lengths):
        self.samples = SAMPLES
        self.seqnames = list(seqnames)
        self.seqlens = [lengths.get(name, 0) for name in seqnames]


def _matches(written, read):
    """Whether a value read back is the value written, allowing for the wider encoding a later chunk
    may have promoted the field to."""
    if isinstance(written, bool):
        return read is True or read == "True"

    if isinstance(written, str):
        return read == written

    if isinstance(read, str):
        try:
            return float(read) == float(written)
        except ValueError:
            return False

    return read is not None and float(read) == float(written)


#: Seeds of the encoder fuzz. Two thirds of them draw a chunk in which a field appears but carries no
#: usable value, the shape the encoder used to fail on, so this many hits it about thirty times over.
FUZZ_SEEDS = 50


def _fuzz_case(tmp_path, seed):
    """
    Round-trip one randomly drawn sequence of INFO fields.

    :param tmp_path: Directory to write the store into.
    :param seed: Seed of the draw.
    :return: The written INFO mappings and the ones read back.
    """
    rng = random.Random(seed)

    def value():
        choice = rng.randrange(5)

        if choice == 0:
            return True
        if choice == 1:
            return rng.randrange(-1000, 1000)
        if choice == 2:
            return rng.uniform(-10, 10)
        if choice == 3:
            return rng.choice(["A", "T", "coding", "x"])

        return rng.choice([None, "."])

    n = rng.randrange(1, 20)
    infos = [{key: value() for key in rng.sample(["F", "DP", "AA"], rng.randrange(0, 4))}
             for _ in range(n)]

    return infos, _roundtrip(tmp_path / f"fuzz{seed}.vcz", infos, chunk=rng.randrange(1, 6))


def test_info_encoding_round_trip_fuzz(tmp_path):
    """Across chunk sizes and sequences of types, a value written reads back as itself and a variant
    carrying no value reads back as carrying none. A boolean False is excluded: an unset VCF Flag is
    absent from INFO, so it is not a value a source can carry."""
    for seed in range(FUZZ_SEEDS):
        infos, read = _fuzz_case(tmp_path, seed)

        assert len(read) == len(infos), f"seed {seed}"

        for written, got in zip(infos, read):
            for key in ("F", "DP", "AA"):
                expected = written.get(key)

                if expected is None or expected == ".":
                    assert got.get(key) is None, f"seed {seed}: {key} {written} -> {got}"
                else:
                    assert _matches(expected, got.get(key)), f"seed {seed}: {key} {written} -> {got}"


def test_info_dtypes_cover_every_encoding_the_writer_reaches():
    """Every encoding the writer settles on is one the store has a dtype for."""
    assert set(ZarrVariantWriter._info_dtypes) == {"bool", "int", "float", "str"}
    assert np.dtype(ZarrVariantWriter._info_dtypes["int"]) == np.int64
