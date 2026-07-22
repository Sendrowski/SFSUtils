"""
Regression tests for the filtration defects found by the eighth release-readiness scan: the outgroup
filtrations reading the genotype strings (which hide a haploid call of a later allele), the poly-allelic
verdict disagreeing between a parser and a filterer, and the per-site scan over the whole coding sequence
frame. Kept fast and unmarked so they run in the default suite.
"""

import numpy as np
import pandas as pd
import pytest

import sfsutils as su
from sfsutils.annotation import DegeneracyAnnotation
from sfsutils.filtration import CodingSequenceFiltration
from sfsutils.io_handlers import MultiHandler, DummyVariant
from sfsutils.settings import Settings

HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=1,length=100000>\n"
    '##INFO=<ID=AA,Number=1,Type=String,Description="ancestral allele">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n'
)


def _write_vcf(path, rows, samples):
    """
    Write a minimal VCF holding the given data rows.

    :param path: The path to write to.
    :param rows: The data rows, each a sequence of the nine fixed columns followed by the genotypes.
    :param samples: The sample names.
    :return: The path as a string.
    """
    columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *samples]

    path.write_text(HEADER + "#" + "\t".join(columns) + "\n" + "".join("\t".join(r) + "\n" for r in rows))

    return str(path)


def _sites(source):
    """
    Read every site of the given source.

    :param source: The variant source.
    :return: The handler and its sites.
    """
    handler = MultiHandler(source=source)

    return handler, list(handler._reader)


def _setup(filtration, samples):
    """
    Set the filtration up against the given sample names.

    :param filtration: The filtration.
    :param samples: The sample names.
    :return: The filtration.
    """
    handler = type("_Handler", (), {"_reader": type("_Reader", (), {"samples": list(samples)})()})()

    filtration._setup(handler)

    return filtration


class TestHaploidLaterAlleleOutgroups:
    """
    A haploid call of the third or a later allele is rendered as a missing genotype wherever the site's
    maximum ploidy is two, so the outgroup filtrations must decide from the numeric calls.
    """

    samples = ("a", "b", "c", "d")

    def test_existing_keeps_haploid_later_allele_call(self, tmp_path):
        """The outgroup carries the second alternate allele as a haploid call, so it is not missing."""
        vcf = _write_vcf(tmp_path / "haploid.vcf", [
            ["1", "1", ".", "T", "C,A", ".", ".", ".", "GT", "2", "0/0", "0/0", "0/1"],
        ], self.samples)

        handler, sites = _sites(vcf)

        # the genotype strings render the outgroup's call as './.'
        assert sites[0].gt_bases[0] == "./."

        f = _setup(su.ExistingOutgroupFiltration(["a"], n_missing=1), self.samples)

        assert f.filter_site(sites[0])

        handler._reader.close()

    def test_deviant_reads_haploid_later_allele_call(self, tmp_path):
        """The outgroup's major base is the second alternate allele, which the ingroup shares."""
        vcf = _write_vcf(tmp_path / "deviant.vcf", [
            ["1", "1", ".", "T", "C,A", ".", ".", ".", "GT", "2", "2/2", "2/2", "2/2"],
            ["1", "2", ".", "T", "C,A", ".", ".", ".", "GT", "2", "0/0", "0/0", "0/0"],
        ], self.samples)

        handler, sites = _sites(vcf)

        f = _setup(su.DeviantOutgroupFiltration(["a"], ingroups=["b", "c", "d"]), self.samples)

        # the ingroup is fixed for the same allele the outgroup carries, then for the reference
        assert [f.filter_site(v) for v in sites] == [True, False]

        handler._reader.close()

    def test_verdicts_agree_across_encodings(self, tmp_path):
        """A mixed-ploidy input reaches the same verdicts read as a VCF and as a VCF-Zarr store."""
        pytest.importorskip("bio2zarr")
        pytest.importorskip("zarr")

        from bio2zarr import vcf as bio2zarr_vcf

        Settings.disable_pbar = True

        samples = ["a", "b", "c", "d", "e", "o1", "o2"]
        rng = np.random.default_rng(42)
        rows = []

        for i in range(200):
            alt = ["C", "A", "G"][:rng.integers(1, 4)]
            n = len(alt) + 1
            genotypes = []

            for _ in samples:
                if rng.random() < 0.12:
                    genotypes.append("./." if rng.random() < 0.5 else ".")
                elif rng.random() < 0.35:
                    genotypes.append(str(rng.integers(0, n)))
                else:
                    genotypes.append(f"{rng.integers(0, n)}/{rng.integers(0, n)}")

            rows.append(["1", str(i * 10 + 1), ".", "T", ",".join(alt), ".", ".", ".", "GT", *genotypes])

        vcf = _write_vcf(tmp_path / "mixed.vcf", rows, samples)
        vcz = str(tmp_path / "mixed.vcz")

        bio2zarr_vcf.convert([vcf], vcz)

        def verdicts(source, make):
            handler, sites = _sites(source)
            f = make()
            f._setup(handler)
            out = [f.filter_site(v) for v in sites]
            handler._reader.close()

            return out

        for make in [
            lambda: su.ExistingOutgroupFiltration(["o1", "o2"], n_missing=1),
            lambda: su.DeviantOutgroupFiltration(["o1", "o2"], ingroups=["a", "b", "c", "d", "e"]),
        ]:
            assert verdicts(vcf, make) == verdicts(vcz, make)

    def test_deviant_retains_monomorphic_and_dummy_sites(self, tmp_path):
        """The monomorphic shortcut and the dummy target site keep their verdicts."""
        vcf = _write_vcf(tmp_path / "mono.vcf", [
            ["1", "1", ".", "T", ".", ".", ".", ".", "GT", "./.", "0/0", "0/0", "0/0"],
        ], self.samples)

        handler, sites = _sites(vcf)

        f = _setup(su.DeviantOutgroupFiltration(["a"], ingroups=["b", "c", "d"]), self.samples)

        assert f.filter_site(sites[0])

        # without the shortcut the missing outgroup fails the strict-mode test
        g = _setup(su.DeviantOutgroupFiltration(["a"], ingroups=["b", "c", "d"], retain_monomorphic=False),
                   self.samples)

        assert not g.filter_site(sites[0])

        dummy = DummyVariant(ref="A", pos=1, chrom="1", n_samples=4)

        assert g.filter_site(dummy)
        assert _setup(su.ExistingOutgroupFiltration(["a"]), self.samples).filter_site(dummy)

        handler._reader.close()

    def test_deviant_counts_multi_character_alleles_as_one(self, tmp_path):
        """An MNP is majority-counted as one allele per haplotype rather than one per base."""
        vcf = _write_vcf(tmp_path / "mnp.vcf", [
            ["1", "1", ".", "AT", "GC", ".", ".", ".", "GT", "1/1", "1/1", "0/0", "0/0"],
        ], self.samples)

        handler, sites = _sites(vcf)

        f = _setup(su.DeviantOutgroupFiltration(["a"], ingroups=["b", "c", "d"], retain_monomorphic=False),
                   self.samples)

        # the outgroup's majority allele is GC, the ingroup's is AT
        assert not f.filter_site(sites[0])

        handler._reader.close()


class TestPolyAllelicAgreement:
    """
    A parser and a filterer must reach the same poly-allelic verdict, which is decided by the alleles the
    included samples actually carry rather than by the ``ALT`` field.
    """

    samples = ("s1", "s2")

    def test_parser_and_filterer_agree(self, tmp_path):
        """A site declaring three alleles of which the samples carry three is dropped on both paths."""
        Settings.disable_pbar = True

        rows = [["1", str(10 + i), ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/1", "2/2"] for i in range(20)]
        vcf = _write_vcf(tmp_path / "polyallelic.vcf", rows, self.samples)

        direct = su.Parser(source=vcf, n=4, filtrations=[su.PolyAllelicFiltration()]).parse()

        assert direct.data.sum().sum() == 0

        out = str(tmp_path / "filtered.vcf.gz")

        su.Filterer(source=vcf, output=out, filtrations=[su.PolyAllelicFiltration()]).filter()

        assert su.Parser(source=out, n=4).parse().data.sum().sum() == 0

    def test_all_true_mask_and_no_mask_agree(self, tmp_path):
        """Naming every sample and naming none of them reach the same verdict."""
        vcf = _write_vcf(tmp_path / "mask.vcf", [
            ["1", "10", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/0", "1/1"],
            ["1", "11", ".", "A", "T,G", ".", ".", "AA=A", "GT", "0/1", "2/2"],
        ], self.samples)

        handler, sites = _sites(vcf)

        f = su.PolyAllelicFiltration(include_samples=list(self.samples))
        f._setup(handler)

        assert f._samples_mask.tolist() == [True, True]

        g = su.PolyAllelicFiltration()
        g._setup(handler)

        assert g._samples_mask is None

        # the third allele is only declared at the first site and carried at the second
        assert [f.filter_site(v) for v in sites] == [True, False]
        assert [g.filter_site(v) for v in sites] == [True, False]

        # a genuine restriction is kept
        h = su.PolyAllelicFiltration(include_samples=["s1"])
        h._setup(handler)

        assert h._samples_mask is not None and h._samples_mask.tolist() == [True, False]

        handler._reader.close()


class TestCodingSequenceIndex:
    """
    The coding sequence lookup must not scan the whole frame per advancing site, and must reach the
    verdicts the scan reached.
    """

    def _scan(self, filtration, v):
        """
        Reach the verdict by scanning the whole coding sequence frame.

        :param filtration: The filtration holding the handler and the cursor.
        :param v: The variant.
        :return: The verdict.
        """
        aliases = filtration._handler.get_aliases(v.CHROM)
        cds = filtration._handler._cds

        if filtration.cd is None or filtration.cd.seqid not in aliases or v.POS > filtration.cd.end:
            filtration.cd = pd.Series({
                'seqid': v.CHROM,
                'start': DegeneracyAnnotation._pos_mock,
                'end': DegeneracyAnnotation._pos_mock
            })

            found = cds[cds['seqid'].isin(aliases) & (cds['end'] >= v.POS)]

            if not found.empty:
                filtration.cd = found.iloc[0]

        return filtration.cd.seqid in aliases and filtration.cd.start <= v.POS <= filtration.cd.end

    def test_matches_the_scan_on_a_real_gff(self):
        """Every site of a real input reaches the verdict the scan reaches."""
        Settings.disable_pbar = True

        handler = MultiHandler(
            source='resources/genome/betula/all.polarized.subset.10000.vcf.gz',
            gff='resources/genome/betula/genome.gff.gz'
        )

        indexed = CodingSequenceFiltration()
        indexed._setup(handler)

        scanned = CodingSequenceFiltration()
        scanned._setup(handler)

        n = kept = 0
        for v in handler._reader:
            verdict = indexed.filter_site(v)
            assert verdict == self._scan(scanned, v)
            n += 1
            kept += bool(verdict)

        assert n > 0 and kept > 0

        handler._reader.close()

    def test_lookup_does_not_scan_the_frame(self):
        """The cost of an advance is independent of the number of coding sequences on other contigs."""
        Settings.disable_pbar = True

        handler = MultiHandler(
            source='resources/genome/betula/all.polarized.subset.10000.vcf.gz',
            gff='resources/genome/betula/genome.gff.gz'
        )

        f = CodingSequenceFiltration()
        f._setup(handler)

        cds = handler._cds
        seqid = cds.seqid.value_counts().index[0]
        positions = cds[cds.seqid == seqid].start.to_numpy()[:200]

        # one site per coding sequence, so the cursor advances on every one of them
        for pos in positions:
            f.filter_site(DummyVariant(ref='A', pos=int(pos), chrom=seqid))

        index = f._get_index(seqid, handler.get_aliases(seqid))

        # the index covers only the contig it was built for
        assert len(index.cds) < len(cds)
        assert set(index.cds.seqid) <= set(handler.get_aliases(seqid))

        handler._reader.close()
