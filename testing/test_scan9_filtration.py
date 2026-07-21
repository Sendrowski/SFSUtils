"""
Regression tests for the filtration defects found by the ninth release-readiness scan: a mask selecting
every sample being dropped (which made a verdict depend on samples nobody asked about), the deviant
outgroup filtration reading multi-character alleles as characters (which made its verdict depend on the
backend), a missing FASTA contig aborting the run, ``max_sites=0`` filtering everything, and the existing
outgroup filtration re-binning a site once per outgroup.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2026-07-22"

import os
import subprocess
import sys
import time

import numpy as np
import pytest

import sfsutils as su
from sfsutils.filtration import CpGFiltration
from sfsutils.io_handlers import MultiHandler, SiteAlleles, ZarrVariantReader
from sfsutils.settings import Settings

HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1,length=100000>\n"
    "##contig=<ID=2,length=100000>\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
)

#: Sites mixing ploidies, multi-allelic records, MNPs, indels and missing calls, so that the numeric and
#: the string paths and the backends can be held against each other on the awkward cases
MIXED_ROWS = [
    ["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/0", "0/1", "1/1", "./."],
    ["1", "11", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/0", "0/1", "0/0", "0/0"],
    ["1", "12", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/1", "2/2", "0/0", "./."],
    ["1", "13", ".", "AT", "GC", ".", ".", "AA=AT", "GT", "1/1", "1/1", "0/0", "0/0"],
    ["1", "14", ".", "AT", "GC,AA", ".", ".", "AA=AT", "GT", "2", "0/0", "0/0", "1/1"],
    ["1", "15", ".", "A", "AT,ATT", ".", ".", "AA=A", "GT", "2", "0/0", "1/1", "1/1"],
    ["1", "16", ".", "ACG", "A", ".", ".", "AA=ACG", "GT", "0/1", "0/0", "0/0", "1/1"],
    ["1", "17", ".", "A", "C,G,T", ".", ".", "AA=A", "GT", "1/1", "2/2", "3/3", "0/0"],
    ["1", "18", ".", "A", "T", ".", ".", "AA=A", "GT", "0/0", "0/0", "0/0", "0/0"],
    ["1", "19", ".", "A", "<NON_REF>,C", ".", ".", "AA=A", "GT", "0/0", "1/1", "2/2", "./."],
    ["1", "20", ".", "A", "T,G", ".", ".", "AA=A", "GT", "./.", "./.", "./.", "./."],
    ["1", "21", ".", "N", "C,G", ".", ".", "AA=N", "GT", "0/0", "1/1", "0/0", "2/2"],
    ["1", "22", ".", "C", "G", ".", ".", "AA=C", "GT", "1", "0/1", "0/0", "1/1"],
]

MIXED_SAMPLES = ["s1", "s2", "s3", "s4"]

#: The msprime fixtures holding the same data in all three backends
TRIO = [
    ("resources/msprime/two_epoch.vcf", "vcf"),
    ("resources/msprime/two_epoch.vcz", "zarr"),
    ("resources/msprime/two_epoch.trees", "tskit"),
]


def write_vcf(path, rows, samples, header=HEADER):
    """
    Write a minimal VCF holding the given data rows.

    :param path: The path to write to.
    :param rows: The data rows, each a sequence of the nine fixed columns followed by the genotypes.
    :param samples: The sample names.
    :param header: The header to write.
    :return: The path as a string.
    """
    columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *samples]

    path.write_text(header + "#" + "\t".join(columns) + "\n" + "".join("\t".join(r) + "\n" for r in rows))

    return str(path)


def to_vcz(vcf, store):
    """
    Convert a VCF to a VCF-Zarr store.

    :param vcf: The path to the VCF.
    :param store: The path to write the store to.
    :return: The path to the store.
    """
    pytest.importorskip("bio2zarr")

    subprocess.run([sys.executable, "-m", "bio2zarr", "vcf2zarr", "convert", vcf, store],
                   capture_output=True, check=True)

    return store


def read(source):
    """
    Stream the sites of a source through the backend its extension selects.

    :param source: The path to the source.
    :return: The list of sites.
    """
    if source.endswith(".vcz") or source.endswith(".zarr"):
        # an explicit chunk size, so the reader's own default cannot vary the comparison
        return list(ZarrVariantReader(source, chunk_size=1000))

    if source.endswith(".trees"):
        import tskit

        from sfsutils.io_handlers import TskitVariantReader

        return list(TskitVariantReader(tskit.load(source)))

    from cyvcf2 import VCF

    return list(VCF(source))


def samples_of(source):
    """
    The sample names of a source.

    :param source: The path to the source.
    :return: The sample names.
    """
    if source.endswith(".vcz") or source.endswith(".zarr"):
        return list(ZarrVariantReader(source, chunk_size=1000).samples)

    if source.endswith(".trees"):
        import tskit

        from sfsutils.io_handlers import TskitVariantReader

        return list(TskitVariantReader(tskit.load(source)).samples)

    from cyvcf2 import VCF

    return list(VCF(source).samples)


def setup(filtration, samples):
    """
    Set the filtration up against the given sample names.

    :param filtration: The filtration.
    :param samples: The sample names.
    :return: The filtration.
    """
    handler = type("_Handler", (), {"_reader": type("_Reader", (), {"samples": list(samples)})()})()

    filtration._setup(handler)

    return filtration


def verdicts(source, build):
    """
    The verdicts a freshly built filtration reaches on every site of a source.

    :param source: The path to the source.
    :param build: A callable taking the sample names and returning the filtration.
    :return: The verdicts, one per site.
    """
    samples = samples_of(source)
    filtration = setup(build(), samples)

    return [filtration.filter_site(v) for v in read(source)]


class TestAllTrueMaskIsHonoured:
    """
    A mask selecting every sample must decide a site exactly as no mask does, so that a sample belonging
    to no requested population cannot flip a verdict (C11).
    """

    pops = {"p1": ["s1", "s2"], "p2": ["s3", "s4"]}

    def _total(self, vcf):
        """
        The total mass of the joint spectra of a parse dropping the poly-allelic sites.

        :param vcf: The path to the VCF.
        :return: The total mass.
        """
        Settings.disable_pbar = True

        spectra = su.Parser(source=vcf, n=4, pops=self.pops, filtrations=[su.PolyAllelicFiltration()]).parse()

        return float(sum(np.asarray(s.data).sum() for s in spectra.to_dict().values()))

    def test_unparsed_sample_does_not_change_the_verdict(self, tmp_path):
        """The populations carry two alleles, so the sites are kept whether or not a fifth sample exists."""
        genotypes = ["0/0", "0/1", "0/0", "0/0"]

        four = write_vcf(
            tmp_path / "four.vcf",
            [["1", str(10 + i), ".", "A", "T,G", ".", ".", "AA=A", "GT", *genotypes] for i in range(20)],
            MIXED_SAMPLES
        )

        five = write_vcf(
            tmp_path / "five.vcf",
            [["1", str(10 + i), ".", "A", "T,G", ".", ".", "AA=A", "GT", *genotypes, "0/0"] for i in range(20)],
            MIXED_SAMPLES + ["x1"]
        )

        assert self._total(four) == self._total(five) == 20

    def test_naming_every_sample_agrees_with_naming_none(self, tmp_path):
        """The two ways of asking for every sample reach the same verdict site by site."""
        vcf = write_vcf(tmp_path / "mixed.vcf", MIXED_ROWS, MIXED_SAMPLES)

        for build in (su.PolyAllelicFiltration, su.SNPFiltration):
            named = verdicts(vcf, lambda: build(include_samples=list(MIXED_SAMPLES)))
            unnamed = verdicts(vcf, build)

            assert named == unnamed
            assert len(named) == len(MIXED_ROWS)

    def test_declared_but_uncalled_allele_is_not_counted(self, tmp_path):
        """A third allele the ``ALT`` field declares but no sample carries leaves the site bi-allelic."""
        vcf = write_vcf(tmp_path / "declared.vcf", [
            ["1", "10", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/0", "0/1"],
            ["1", "11", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/1", "2/2"],
        ], ["s1", "s2"])

        assert verdicts(vcf, su.PolyAllelicFiltration) == [True, False]

    def test_filterer_keeps_the_declared_but_uncalled_allele(self, tmp_path):
        """The filterer without a samples restriction reaches the same verdict as the parser."""
        Settings.disable_pbar = True

        vcf = write_vcf(tmp_path / "declared.vcf", [
            ["1", "10", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/0", "0/1"],
            ["1", "11", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/1", "2/2"],
        ], ["s1", "s2"])

        out = str(tmp_path / "filtered.vcf")

        su.Filterer(source=vcf, output=out, filtrations=[su.PolyAllelicFiltration()]).filter()

        assert [v.POS for v in read(out)] == [10]

    def test_snp_filtration_judges_from_the_genotypes(self, tmp_path):
        """A record declaring an alternate allele no sample carries is not polymorphic."""
        vcf = write_vcf(tmp_path / "mono.vcf", [
            ["1", "10", ".", "A", "T", ".", ".", "AA=A", "GT", "0/0", "0/0"],
            ["1", "11", ".", "A", "T", ".", ".", "AA=A", "GT", "0/0", "0/1"],
        ], ["s1", "s2"])

        assert verdicts(vcf, su.SNPFiltration) == [False, True]


class TestDeviantOutgroupAlleles:
    """
    A multi-character allele must be majority-counted as one allele on every backend (C12).
    """

    def _build(self):
        """
        The filtration under test.

        :return: The filtration.
        """
        return su.DeviantOutgroupFiltration(["s1"], ingroups=["s2", "s3", "s4"], retain_monomorphic=False)

    def test_mnp_counts_as_one_allele(self, tmp_path):
        """The outgroup's ``GC`` call weighs one allele per haplotype, not one per base."""
        vcf = write_vcf(tmp_path / "mnp.vcf", [
            ["1", "10", ".", "AT", "GC", ".", ".", "AA=AT", "GT", "1/1", "0/0", "0/0", "0/0"],
            ["1", "11", ".", "AT", "GC", ".", ".", "AA=AT", "GT", "1/1", "1/1", "1/1", "0/0"],
        ], MIXED_SAMPLES)

        assert verdicts(vcf, self._build) == [False, True]

    def test_agrees_across_backends_on_mnps_and_indels(self, tmp_path):
        """The verdicts of the VCF and of the store built from it match site by site."""
        vcf = write_vcf(tmp_path / "mixed.vcf", MIXED_ROWS, MIXED_SAMPLES)
        vcz = to_vcz(vcf, str(tmp_path / "mixed.vcz"))

        from_vcf = verdicts(vcf, self._build)
        from_vcz = verdicts(vcz, self._build)

        assert from_vcf == from_vcz
        assert len(from_vcf) == len(MIXED_ROWS)

        # the haploid call of a later allele at a multi-character site is where the two used to part
        assert from_vcf[4] is False


class TestCrossBackendEquivalence:
    """
    Every masked and outgroup filtration must reach the same verdicts on the same data, whichever backend
    presents it.
    """

    @staticmethod
    def _builders(samples):
        """
        The filtrations compared, each as a zero-argument builder.

        :param samples: The sample names of the source.
        :return: The named builders.
        """
        half = list(samples[:max(1, len(samples) // 2)])

        return {
            "snp": su.SNPFiltration,
            "snp-masked": lambda: su.SNPFiltration(include_samples=half),
            "poly": su.PolyAllelicFiltration,
            "poly-masked": lambda: su.PolyAllelicFiltration(include_samples=half),
            "deviant": lambda: su.DeviantOutgroupFiltration([samples[0]], retain_monomorphic=False),
            "existing": lambda: su.ExistingOutgroupFiltration([samples[0], samples[-1]], n_missing=1),
        }

    def test_mixed_fixture_agrees_between_vcf_and_zarr(self, tmp_path):
        """Mixed ploidy, multi-allelic records, MNPs, indels and missing calls agree on both backends."""
        vcf = write_vcf(tmp_path / "mixed.vcf", MIXED_ROWS, MIXED_SAMPLES)
        vcz = to_vcz(vcf, str(tmp_path / "mixed.vcz"))

        compared = 0
        for name, build in self._builders(MIXED_SAMPLES).items():
            from_vcf = verdicts(vcf, build)
            from_vcz = verdicts(vcz, build)

            assert from_vcf == from_vcz, name
            assert len(from_vcf) == len(MIXED_ROWS)

            compared += len(from_vcf)

        assert compared == 6 * len(MIXED_ROWS)

    def test_msprime_trio_agrees_on_every_backend(self):
        """The committed VCF, store and tree sequence hold the same data and must be judged alike."""
        available = [(path, name) for path, name in TRIO if os.path.exists(path)]

        if len(available) < 2:
            pytest.skip("the msprime fixtures are not available")

        samples = samples_of(available[0][0])
        reference = None

        for path, name in available:
            assert samples_of(path) == samples, name

            for label, build in self._builders(samples).items():
                got = verdicts(path, build)

                if reference is None:
                    reference = {}

                assert reference.setdefault(label, got) == got, f"{name} disagrees on {label}"

        assert sum(len(v) for v in reference.values()) > 1000


class TestMissingContig:
    """
    A contig the FASTA carries no sequence for must not abort the run (C13).
    """

    def _fasta(self, tmp_path):
        """
        A single-contig reference.

        :param tmp_path: The temporary directory.
        :return: The path to the FASTA.
        """
        path = tmp_path / "ref.fasta"
        path.write_text(">1\n" + "ACGT" * 25 + "\n")

        return str(path)

    def test_parse_survives_an_absent_contig(self, tmp_path, caplog):
        """The sites on the absent contig are kept and the rest of the parse completes."""
        Settings.disable_pbar = True

        vcf = write_vcf(tmp_path / "two_contigs.vcf", [
            ["1", "2", ".", "C", "T", ".", ".", "AA=C", "GT", "0/0", "0/1"],
            ["2", "2", ".", "C", "T", ".", ".", "AA=C", "GT", "0/0", "0/1"],
            ["2", "3", ".", "G", "A", ".", ".", "AA=G", "GT", "0/0", "0/1"],
        ], ["s1", "s2"])

        f = CpGFiltration()
        handler = MultiHandler(source=vcf, fasta=self._fasta(tmp_path))
        f._setup(handler)

        sites = read(vcf)

        # the reference reads ACGTACGT..., so position 2 of contig 1 is the only CpG site
        assert [f.filter_site(v) for v in sites] == [False, True, True]

        # the absent contig is warned about once, not once per site
        assert f._missing_contigs == {"2"}

        handler._reader.close()

    def test_parser_completes_over_an_absent_contig(self, tmp_path):
        """A whole parse runs through rather than being discarded by the one absent scaffold."""
        Settings.disable_pbar = True

        vcf = write_vcf(tmp_path / "two_contigs.vcf", [
            ["1", "5", ".", "A", "T", ".", ".", "AA=A", "GT", "0/0", "0/1"],
            ["2", "2", ".", "C", "T", ".", ".", "AA=C", "GT", "0/0", "0/1"],
        ], ["s1", "s2"])

        sfs = su.Parser(source=vcf, n=2, fasta=self._fasta(tmp_path), filtrations=[CpGFiltration()]).parse()

        assert float(np.asarray(sfs.data).sum()) == 2


class TestMaxSites:
    """
    ``max_sites`` must bound the output rather than being reached only on an exact hit (C10).
    """

    def test_zero_is_rejected(self, tmp_path):
        """A non-positive bound would silently mean no bound at all."""
        vcf = write_vcf(tmp_path / "mixed.vcf", MIXED_ROWS, MIXED_SAMPLES)

        for value in (0, -1):
            with pytest.raises(ValueError, match="max_sites must be positive"):
                su.Filterer(source=vcf, output=str(tmp_path / "out.vcf"), max_sites=value)

    def test_bound_is_honoured(self, tmp_path):
        """The output stops at the bound."""
        Settings.disable_pbar = True

        vcf = write_vcf(tmp_path / "mixed.vcf", MIXED_ROWS, MIXED_SAMPLES)
        out = str(tmp_path / "out.vcf")

        su.Filterer(source=vcf, output=out, max_sites=3, filtrations=[su.NoFiltration()]).filter()

        assert len(read(out)) == 3


class TestExistingOutgroupBatching:
    """
    The batched missing-outgroup count must reach the verdicts the per-outgroup count reached (P7).
    """

    @staticmethod
    def _reference(filtration, variant, n_missing):
        """
        Count the missing outgroups one call into the numeric view at a time.

        :param filtration: The filtration holding the outgroup rows.
        :param variant: The site.
        :param n_missing: The number of missing outgroups required to fail.
        :return: The verdict.
        """
        site = SiteAlleles.from_site(variant)
        rows = filtration._outgroup_rows

        return sum(site.n_called(rows[i:i + 1]) == 0 for i in range(len(rows))) < n_missing

    @pytest.mark.parametrize("n_missing", [1, 2, 3])
    def test_verdicts_are_unchanged(self, tmp_path, n_missing):
        """Every site of the mixed fixture reaches the verdict of the per-outgroup count."""
        vcf = write_vcf(tmp_path / "mixed.vcf", MIXED_ROWS, MIXED_SAMPLES)

        f = setup(su.ExistingOutgroupFiltration(list(MIXED_SAMPLES), n_missing=n_missing), MIXED_SAMPLES)

        compared = 0
        for v in read(vcf):
            assert f.filter_site(v) == self._reference(f, v, n_missing)
            compared += 1

        assert compared == len(MIXED_ROWS)

    def test_batching_is_faster_than_one_call_per_outgroup(self):
        """The cost of a site no longer grows with a re-binning per outgroup."""
        if not os.path.exists(TRIO[0][0]):
            pytest.skip("the msprime fixture is not available")

        sites = read(TRIO[0][0])
        samples = samples_of(TRIO[0][0])

        f = setup(su.ExistingOutgroupFiltration(list(samples), n_missing=1), samples)

        start = time.perf_counter()
        for v in sites:
            f.filter_site(v)
        batched = time.perf_counter() - start

        start = time.perf_counter()
        for v in sites:
            self._reference(f, v, 1)
        per_outgroup = time.perf_counter() - start

        assert batched < per_outgroup
