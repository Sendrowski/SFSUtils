"""
Regression tests for the filtration and annotation defects found by the tenth release-readiness scan: an
annotation reused across inputs keeping the first input's reference, the CpG filtration reading past the
end of a short FASTA record, sites-only input reaching the masked filtrations with no genotype to judge by,
the coding sequence filtration carrying its processed count across passes, a second ``filter()`` /
``annotate()`` writing a header and raising, and a coding sequence with an undefined phase.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2026-07-22"

import logging

import pytest

import sfsutils as su
from sfsutils.annotation import _CDSIndex
from sfsutils.filtration import CodingSequenceFiltration, CpGFiltration, PolyAllelicFiltration, SNPFiltration

HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1,length=1000>\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
)

#: A coding sequence spanning one short contig, in frame and on the forward strand
GFF = "##gff-version 3\n1\tx\tCDS\t1\t18\t.\t+\t0\tID=c1;Parent=t1\n"

#: The same coding sequence with the phase left undefined, which GFF3 allows
GFF_NO_PHASE = "##gff-version 3\n1\tx\tCDS\t1\t18\t.\t+\t.\tID=c1;Parent=t1\n"


def write_vcf(path, rows, samples, header=HEADER):
    """
    Write a minimal VCF holding the given data rows, omitting the genotype columns for no samples.

    :param path: The path to write to.
    :param rows: The data rows, each a sequence of the eight fixed columns followed by the genotypes.
    :param samples: The sample names, empty for a sites-only file.
    :param header: The header to write.
    :return: The path as a string.
    """
    columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]

    if samples:
        columns += ["FORMAT", *samples]

    path.write_text(header + "#" + "\t".join(columns) + "\n" + "".join("\t".join(r) + "\n" for r in rows))

    return str(path)


def written(path):
    """
    Read back the data lines of a VCF.

    :param path: The path to read.
    :return: The data lines, split on tabs.
    """
    with open(path) as fh:
        return [line.rstrip("\n").split("\t") for line in fh if not line.startswith("#")]


def degeneracies(path):
    """
    Read back the degeneracy annotation of every record of a VCF.

    :param path: The path to read.
    :return: The ``Degeneracy`` and ``Degeneracy_Info`` values per record.
    """
    return [
        tuple(f for f in record[7].split(";") if f.startswith("Degeneracy"))
        for record in written(path)
    ]


@pytest.fixture
def short_fasta(tmp_path):
    """
    A FASTA whose only record is far shorter than the contig length the VCF header declares.

    :param tmp_path: The temporary directory.
    :return: The path to the FASTA.
    """
    path = tmp_path / "short.fa"
    path.write_text(">1\nACGTACGTAC\n")

    return str(path)


class TestCpGShortContig:
    """
    A reference sequence not reaching a site must not abort the run, whatever the reference base is.
    """

    @pytest.mark.parametrize("ref", ["C", "G"])
    def test_site_beyond_end_of_contig_is_retained(self, tmp_path, short_fasta, ref, caplog):
        """
        Both reference bases have to survive a FASTA record that stops short of the site.
        """
        vcf = write_vcf(tmp_path / "in.vcf",
                        [["1", "500", ".", ref, "A", ".", ".", f"AA={ref}", "GT", "0/1"]], ["s1"])

        out = str(tmp_path / f"out_{ref}.vcf")

        f = su.Filterer(source=vcf, output=out, fasta=short_fasta, filtrations=[CpGFiltration()])

        with caplog.at_level(logging.WARNING, logger="sfsutils"):
            f.filter()

        assert f.n_filtered == 0
        assert len(written(out)) == 1
        assert "beyond the end of contig 1" in caplog.text

    def test_short_contig_warns_once(self, tmp_path, short_fasta, caplog):
        """
        The warning is per contig, not per site.
        """
        rows = [["1", str(pos), ".", "G", "A", ".", ".", "AA=G", "GT", "0/1"] for pos in (500, 600, 700)]
        vcf = write_vcf(tmp_path / "in.vcf", rows, ["s1"])

        out = str(tmp_path / "out.vcf")

        f = su.Filterer(source=vcf, output=out, fasta=short_fasta, filtrations=[CpGFiltration()])

        with caplog.at_level(logging.WARNING, logger="sfsutils"):
            f.filter()

        assert len(written(out)) == 3
        assert caplog.text.count("beyond the end of contig") == 1

    def test_cpg_context_within_the_sequence_is_unchanged(self, tmp_path):
        """
        The bounds check must not change the verdict where the sequence does reach the site. ``ACGTACGTAC``
        has a CpG at positions 2 (C) and 3 (G), and the terminal bases have no neighbour to be typed by.
        """
        fasta = tmp_path / "ref.fa"
        fasta.write_text(">1\nACGTACGTAC\n")

        assert CpGFiltration._is_cpg("ACGTACGTAC", 2, "C") is True
        assert CpGFiltration._is_cpg("ACGTACGTAC", 3, "G") is True
        assert CpGFiltration._is_cpg("ACGTACGTAC", 6, "C") is True
        assert CpGFiltration._is_cpg("ACGTACGTAC", 10, "C") is False
        assert CpGFiltration._is_cpg("GCGTACGTAC", 1, "G") is False
        assert CpGFiltration._is_cpg("ACGTACGTAC", 11, "G") is None
        assert CpGFiltration._is_cpg("ACGTACGTAC", 0, "C") is None


class TestSitesOnlyInput:
    """
    With no sample to carry an allele, the masked filtrations have to fall back to the declared alleles
    rather than judging every site monomorphic (dropping all of them) or bi-allelic (keeping all of them).
    """

    ROWS = [
        ["1", "10", ".", "A", "T", ".", ".", "AA=A"],
        ["1", "20", ".", "A", "T,G", ".", ".", "AA=A"],
        ["1", "30", ".", "A", ".", ".", ".", "AA=A"],
    ]

    def test_snp_filtration_keeps_declared_snps(self, tmp_path, caplog):
        """
        The two SNPs survive and the monomorphic site does not.
        """
        vcf = write_vcf(tmp_path / "sites.vcf", self.ROWS, [])
        out = str(tmp_path / "out.vcf")

        f = su.Filterer(source=vcf, output=out, filtrations=[SNPFiltration()])

        with caplog.at_level(logging.WARNING, logger="sfsutils"):
            f.filter()

        assert [r[1] for r in written(out)] == ["10", "20"]
        assert f.n_filtered == 1
        assert "falls back to the alleles declared" in caplog.text

    def test_poly_allelic_filtration_drops_declared_poly_allelic_sites(self, tmp_path):
        """
        The tri-allelic site is dropped rather than kept.
        """
        vcf = write_vcf(tmp_path / "sites.vcf", self.ROWS, [])
        out = str(tmp_path / "out.vcf")

        f = su.Filterer(source=vcf, output=out, filtrations=[PolyAllelicFiltration()])
        f.filter()

        assert [r[1] for r in written(out)] == ["10", "30"]
        assert f.n_filtered == 1

    def test_verdicts_match_the_same_data_with_samples(self, tmp_path):
        """
        The fallback has to reach the verdict samples carrying every declared allele would produce.
        """
        genotypes = [["0/0", "0/1"], ["0/1", "2/2"], ["0/0", "0/0"]]
        with_samples = [row + ["GT", *gt] for row, gt in zip(self.ROWS, genotypes)]

        typed = write_vcf(tmp_path / "typed.vcf", with_samples, ["s1", "s2"])
        sites = write_vcf(tmp_path / "sites.vcf", self.ROWS, [])

        for filtration in (SNPFiltration, PolyAllelicFiltration):
            kept = []

            for name, source in (("typed", typed), ("sites", sites)):
                out = str(tmp_path / f"{filtration.__name__}_{name}.vcf")

                su.Filterer(source=source, output=out, filtrations=[filtration()]).filter()

                kept.append([r[1] for r in written(out)])

            assert kept[0] == kept[1]

    def test_samples_are_still_used_where_there_are_any(self, tmp_path):
        """
        An alternate allele no sample carries must keep making a site monomorphic, as before.
        """
        rows = [["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/0"]]
        vcf = write_vcf(tmp_path / "typed.vcf", rows, ["s1"])
        out = str(tmp_path / "out.vcf")

        f = su.Filterer(source=vcf, output=out, filtrations=[SNPFiltration()])
        f.filter()

        assert written(out) == []
        assert f.n_filtered == 1


class TestCodingSequenceFiltrationRewind:
    """
    The processed count belongs to one pass, as every other counter of the filtration does.
    """

    def test_n_processed_is_reset_on_rewind(self):
        """
        A rewind restores the count the warning about a mismatched GFF is guarded by.
        """
        f = CodingSequenceFiltration()
        f.n_processed = 17

        f._rewind()

        assert f.n_processed == 0

    def test_mismatched_gff_warns_on_every_pass(self, tmp_path, caplog):
        """
        A shared filtration must warn about a GFF whose contigs do not match on the second pass too.
        """
        gff = tmp_path / "other.gff"
        gff.write_text("##gff-version 3\nother\tx\tCDS\t1\t18\t.\t+\t0\tID=c1;Parent=t1\n")

        rows = [["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/1"]]
        vcf = write_vcf(tmp_path / "in.vcf", rows, ["s1"])

        filtration = CodingSequenceFiltration()

        for i in range(2):
            caplog.clear()

            out = str(tmp_path / f"out{i}.vcf")

            with caplog.at_level(logging.WARNING, logger="sfsutils"):
                su.Filterer(source=vcf, output=out, gff=str(gff), filtrations=[filtration]).filter()

            # the teardown rewinds the filtration, so the count is back at the start of the next pass
            assert filtration.n_processed == 0
            assert "No subsequent coding sequence found" in caplog.text


class TestRepeatedPasses:
    """
    A second pass over an input has to produce what the first one did, rather than a header-only file.
    """

    def test_filter_can_be_called_twice(self, tmp_path):
        """
        The second ``filter()`` writes the same records and reports its own count.
        """
        rows = [["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/1"],
                ["1", "20", ".", "A", "T,G", ".", ".", "AA=A", "GT", "1/2"]]

        vcf = write_vcf(tmp_path / "in.vcf", rows, ["s1"])

        f = su.Filterer(source=vcf, output=str(tmp_path / "out1.vcf"), filtrations=[PolyAllelicFiltration()])
        f.filter()

        first, n_first = written(str(tmp_path / "out1.vcf")), f.n_filtered

        f.output = str(tmp_path / "out2.vcf")
        f.filter()

        assert written(str(tmp_path / "out2.vcf")) == first
        assert f.n_filtered == n_first

    def test_filter_releases_the_reader_when_setup_fails(self, tmp_path):
        """
        Discarding the reader of a previous pass must release it rather than leak it, so that a setup
        failure still leaves nothing open.
        """

        class RecordingReader:
            """A stand-in reader that records whether it was closed."""

            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        rows = [["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/1"]]
        vcf = write_vcf(tmp_path / "in.vcf", rows, ["s1"])

        # a tree sequence cannot be written to, so the writer the setup opens raises
        f = su.Filterer(source=vcf, output=str(tmp_path / "out.trees"), filtrations=[])

        reader = RecordingReader()
        f.__dict__["_reader"] = reader

        with pytest.raises(ValueError):
            f.filter()

        assert reader.closed

    def test_filter_recovers_from_a_failed_pass(self, tmp_path):
        """
        A pass that raised must not stop the next one from reading the input from the start.
        """
        rows = [["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/1"]]
        vcf = write_vcf(tmp_path / "in.vcf", rows, ["s1"])

        f = su.Filterer(source=vcf, output=str(tmp_path / "out.trees"), filtrations=[])

        with pytest.raises(ValueError):
            f.filter()

        f.output = str(tmp_path / "out.vcf")
        f.filter()

        assert len(written(f.output)) == 1

    def test_annotate_can_be_called_twice(self, tmp_path):
        """
        The second ``annotate()`` writes the same records rather than a header and a raise.
        """
        gff = tmp_path / "cds.gff"
        gff.write_text(GFF)

        fasta = tmp_path / "ref.fa"
        fasta.write_text(">1\nATGGTTTTTCGTTATTAA\n")

        rows = [["1", "6", ".", "T", "C", ".", ".", "AA=T", "GT", "0/1"]]
        vcf = write_vcf(tmp_path / "in.vcf", rows, ["s1"])

        a = su.Annotator(source=vcf, output=str(tmp_path / "out1.vcf"), fasta=str(fasta), gff=str(gff),
                         annotations=[su.DegeneracyAnnotation()])
        a.annotate()

        first = degeneracies(str(tmp_path / "out1.vcf"))

        a.output = str(tmp_path / "out2.vcf")
        a.annotate()

        assert degeneracies(str(tmp_path / "out2.vcf")) == first
        assert first == [("Degeneracy=4", "Degeneracy_Info=2,+,GTT")]


class TestSharedAnnotationAcrossInputs:
    """
    An annotation reused across inputs must annotate each of them against its own reference.
    """

    #: The two genomes differ only in the second codon, ``GTT`` (4-fold at its third position) against
    #: ``ATT`` (2-fold), so a stale reference sequence books a selected site as a neutral one
    GENOMES = {"A": ("ATGGTTTTTCGTTATTAA", ("Degeneracy=4", "Degeneracy_Info=2,+,GTT")),
               "B": ("ATGATTTTTCGTTATTAA", ("Degeneracy=2", "Degeneracy_Info=2,+,ATT"))}

    def _annotate(self, tmp_path, annotation, name, tag):
        """
        Annotate one of the two genomes with the given annotation.

        :param tmp_path: The temporary directory.
        :param annotation: The annotation to apply.
        :param name: The genome to annotate.
        :param tag: A tag distinguishing the output of this call.
        :return: The degeneracy annotation of the single record.
        """
        gff = tmp_path / "cds.gff"
        gff.write_text(GFF)

        fasta = tmp_path / f"{name}.fa"
        fasta.write_text(f">1\n{self.GENOMES[name][0]}\n")

        rows = [["1", "6", ".", "T", "C", ".", ".", "AA=T", "GT", "0/1"]]
        vcf = write_vcf(tmp_path / f"{name}.vcf", rows, ["s1"])

        out = str(tmp_path / f"{tag}{name}.vcf")

        su.Annotator(source=vcf, output=out, fasta=str(fasta), gff=str(gff),
                     annotations=[annotation]).annotate()

        return degeneracies(out)

    def test_reused_annotation_matches_fresh_ones(self, tmp_path):
        """
        Running one annotation over both genomes must give what two fresh annotations give.
        """
        shared = su.DegeneracyAnnotation()

        for name in ("A", "B"):
            expected = [self.GENOMES[name][1]]

            assert self._annotate(tmp_path, shared, name, "shared") == expected
            assert self._annotate(tmp_path, su.DegeneracyAnnotation(), name, "fresh") == expected

    def test_counts_do_not_accumulate_across_inputs(self, tmp_path):
        """
        The reported counts belong to the input just annotated.
        """
        shared = su.DegeneracyAnnotation()

        for name in ("A", "B"):
            self._annotate(tmp_path, shared, name, "shared")

            assert shared.n_annotated == 1
            assert shared.n_skipped == 0
            assert shared.mismatches == []
            assert shared.errors == []


class TestUndefinedPhase:
    """
    GFF3 allows a coding sequence to leave the phase undefined, which must not abort the annotation.
    """

    def test_phases_are_parsed_once(self):
        """
        An undefined phase is read as no offset into the first codon.
        """
        assert list(_CDSIndex._parse_phases(["0", "1", "2", ".", None])) == [0, 1, 2, 0, 0]

    def test_annotation_survives_an_undefined_phase(self, tmp_path, caplog):
        """
        The site is annotated as it would be at phase 0, with a warning naming the coding sequences.
        """
        gff = tmp_path / "cds.gff"
        gff.write_text(GFF_NO_PHASE)

        fasta = tmp_path / "ref.fa"
        fasta.write_text(">1\nATGGTTTTTCGTTATTAA\n")

        rows = [["1", "6", ".", "T", "C", ".", ".", "AA=T", "GT", "0/1"]]
        vcf = write_vcf(tmp_path / "in.vcf", rows, ["s1"])

        out = str(tmp_path / "out.vcf")

        with caplog.at_level(logging.WARNING, logger="sfsutils"):
            su.Annotator(source=vcf, output=out, fasta=str(fasta), gff=str(gff),
                         annotations=[su.DegeneracyAnnotation()]).annotate()

        assert degeneracies(out) == [("Degeneracy=4", "Degeneracy_Info=2,+,GTT")]
        assert "leave the phase undefined" in caplog.text

    def test_debug_logging_survives_an_undefined_phase(self, tmp_path, caplog):
        """
        The debug message about the located coding sequence is what raised, so it is exercised too.
        """
        gff = tmp_path / "cds.gff"
        gff.write_text(GFF_NO_PHASE)

        fasta = tmp_path / "ref.fa"
        fasta.write_text(">1\nATGGTTTTTCGTTATTAA\n")

        rows = [["1", "6", ".", "T", "C", ".", ".", "AA=T", "GT", "0/1"]]
        vcf = write_vcf(tmp_path / "in.vcf", rows, ["s1"])

        out = str(tmp_path / "out.vcf")

        with caplog.at_level(logging.DEBUG, logger="sfsutils"):
            su.Annotator(source=vcf, output=out, fasta=str(fasta), gff=str(gff),
                         annotations=[su.DegeneracyAnnotation()]).annotate()

        assert "Found coding sequence: 1:1-18" in caplog.text
        assert degeneracies(out) == [("Degeneracy=4", "Degeneracy_Info=2,+,GTT")]
