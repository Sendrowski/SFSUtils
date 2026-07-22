"""
Filtrations and a filterer to apply them.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-11"

import functools
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Dict, Set

import numpy as np
import pandas as pd

from .annotation import DegeneracyAnnotation, _CDSIndex
from .io_handlers import get_major_base, MultiHandler, get_called_bases, get_distinct_called_bases, \
    get_distinct_called_alleles, DummyVariant, SiteAlleles, \
    Site, VariantReader, VCFHandler, \
    VariantWriter

# get logger
logger = logging.getLogger('sfsutils')

# cyvcf2 genotype codes as returned by ``Variant.gt_types`` for a VCF opened with the default ``gts012=False``
HOM_REF, HET, UNKNOWN, HOM_ALT = 0, 1, 2, 3


def _count_filtered(func: Callable) -> Callable:
    """
    Decorator that increases ``self.n_filtered`` by 1 if the decorated function returns False.
    """

    @functools.wraps(func)
    def wrapper(self, variant):
        """
        Wrapper function.

        :param self: Self.
        :param variant: The variant to filter.
        :return: The result of the decorated function.
        """
        result = func(self, variant)
        if not result:
            self.n_filtered += 1
        return result

    return wrapper


class Filtration(ABC):
    """
    Base class for filtering sites based on certain criteria.
    """

    #: The number of sites that didn't pass the filter.
    n_filtered: int = 0

    def __init__(self):
        """
        Initialize filtration.
        """
        #: The logger.
        self._logger = logger.getChild(self.__class__.__name__)

        #: The handler.
        self._handler: MultiHandler | None = None

    @abstractmethod
    @_count_filtered
    def filter_site(self, variant: Site) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        pass

    def _setup(self, handler: MultiHandler):
        """
        Perform any necessary pre-processing. This method is called before the actual filtration.

        :param handler: The handler.
        """
        self._handler = handler

    def _rewind(self):
        """
        Rewind the filtration.
        """
        self.n_filtered = 0

    def _teardown(self):
        """
        Perform any necessary post-processing. This method is called after the actual filtration.
        """
        self._logger.info(f"Filtered out {self.n_filtered} sites.")


class MaskedFiltration(Filtration, ABC):
    """
    Filter sites based on a samples mask. A mask of ``None`` selects every sample rather than turning the
    filtration into a test on the ``ALT`` field, so that naming every sample and naming none of them reach
    the same verdict, and a sample belonging to no requested population cannot change one.

    Where the input carries no samples at all, as a sites-only VCF or store does, the genotypes cannot
    settle anything and the verdict is taken from the declared alleles instead, with one warning on setup.
    """

    #: Whether the declared alleles stand in for the genotypes, held at class level so a restored instance
    #: predating this attribute has one.
    _use_declared_alleles: bool = False

    def __init__(
            self,
            use_parser: bool = True,
            include_samples: List[str] | None = None,
            exclude_samples: List[str] | None = None
    ):
        """
        Create a new filtration instance.

        :param use_parser: Whether to use the samples mask from the parser, if used together with parser.
        :param include_samples: The samples to include, defaults to all samples.
        :param exclude_samples: The samples to exclude, defaults to no samples.
        """
        super().__init__()

        #: Whether to use the samples mask from the parser, if used together with parser.
        self.use_parser: bool = use_parser

        #: The samples to include.
        self.include_samples: List[str] | None = include_samples

        #: The samples to exclude.
        self.exclude_samples: List[str] | None = exclude_samples

        #: The samples mask.
        self._samples_mask: np.ndarray | None = None

        #: Whether the verdict is taken from the declared alleles because no sample is left to judge by.
        self._use_declared_alleles: bool = False

    def _prepare_samples_mask(self):
        """
        Prepare the samples mask.
        """
        from .parser import Parser

        if self.use_parser and isinstance(self._handler, Parser):

            # use samples mask from parser
            self._samples_mask = self._handler._samples_mask

        elif self.include_samples is None and self.exclude_samples is None:

            # no samples mask
            self._samples_mask = None

        else:

            samples = np.array(self._handler._reader.samples)

            def _check(names, label):
                """Reject names that are absent from the input rather than quietly ignoring them."""
                missing = sorted(set(names) - set(samples.tolist()))

                if missing:
                    raise ValueError(f'The following {label} samples are not present in the input: {missing}.')

            # determine samples to include
            if self.include_samples is None:

                mask = np.ones(len(samples)).astype(bool)
            else:
                _check(self.include_samples, 'included')
                mask = np.isin(samples, self.include_samples)

            # determine samples to exclude
            if self.exclude_samples is not None:
                _check(self.exclude_samples, 'excluded')
                mask &= ~np.isin(samples, self.exclude_samples)

            self._samples_mask = mask

    def _select(self, values) -> np.ndarray:
        """
        Restrict per-sample values to the effective sample set.

        :param values: The per-sample values.
        :return: The values of the selected samples, which are all of them where no mask restricts them.
        """
        values = np.asarray(values)

        return values if self._samples_mask is None else values[self._samples_mask]

    def _setup(self, handler: MultiHandler):
        """
        Prepare the samples mask and determine whether any sample is left to judge a site by.

        :param handler: The handler.
        """
        super()._setup(handler)

        # prepare samples mask
        self._prepare_samples_mask()

        self._use_declared_alleles = self._count_effective_samples() == 0

        # a sites-only input carries no genotypes at all, and judging from none of them would make every
        # site look monomorphic and bi-allelic at once, so the declared alleles are all there is to go on
        if self._use_declared_alleles:
            self._logger.warning(f'No sample is available to judge sites by, so {self.__class__.__name__} '
                                 f'falls back to the alleles declared in the REF and ALT fields.')

    def _count_effective_samples(self) -> int:
        """
        Count the samples the filtration judges a site by.

        :return: The number of samples the mask selects, or all of them where no mask restricts them.
        """
        if self._samples_mask is not None:
            return int(np.asarray(self._samples_mask).sum())

        return len(self._handler._reader.samples)


class SNPFiltration(MaskedFiltration):
    """
    Only keep SNPs. Note that this entails discarding mono-morphic sites, monomorphism being judged from
    the alleles the included samples actually carry rather than from the ``ALT`` field.
    """

    @_count_filtered
    def filter_site(self, variant: Site) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is an SNP that is polymorphic among the included samples,
            ``False`` otherwise. Poly-allelic SNPs are retained; use :class:`PolyAllelicFiltration` to
            drop those.
        """
        # an indel or MNP is never an SNP, whatever its called bases look like: the masked test below
        # counts genotype characters, so without this an ``A -> AT`` indel would pass as polymorphic
        if not variant.is_snp:
            return False

        # a dummy site carries a single ancestral allele by construction and no genotypes to judge it by
        if isinstance(variant, DummyVariant):
            return True

        # with no sample to carry them, an alternate allele is polymorphic iff the input declares one
        if self._use_declared_alleles:
            return len(variant.ALT) > 0

        # check whether the variant is polymorphic among the included samples, which are all of them where
        # no mask restricts them. Building and re-splitting the genotype strings dominates this check, so
        # settle it from the numeric calls where the backend provides them, and decode the bases only when
        # it does not. Multi-character alleles are left to the bases, which count a genotype character at a
        # time and so read an ``AT`` call as two
        site = SiteAlleles.from_site(variant)

        if site is not None and site.single_character:
            return len(site.distinct(self._samples_mask)) > 1

        # the numeric genotype codes settle most samples on their own where they are available
        types = getattr(variant, 'gt_types', None)

        if types is not None:
            types = self._select(types)

            # a heterozygote makes the site polymorphic on its own
            if (types == HET).any():
                return True

            called = types[types != UNKNOWN]

            if called.size == 0:
                return False

            # a mixture of reference and non-reference homozygotes is polymorphic, all-reference is not
            if (called == HOM_REF).any():
                return bool((called != HOM_REF).any())

            # every call is a homozygous alternate, which may still cover two different alternate alleles,
            # so fall through to comparing the actual bases

        return len(get_distinct_called_bases(self._select(variant.gt_bases))) > 1


class SNVFiltration(Filtration):
    """
    Only keep single site variants (discard indels and MNPs but keep monomorphic sites).
    """

    @_count_filtered
    def filter_site(self, variant: Site) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is kept, ``False`` otherwise.
        """
        return np.all([alt in ['A', 'C', 'G', 'T'] for alt in [variant.REF] + variant.ALT])


class PolyAllelicFiltration(MaskedFiltration):
    """
    Filter out poly-allelic sites.
    """

    @_count_filtered
    def filter_site(self, variant: Site) -> bool:
        """
        Filter site. A site is poly-allelic where three or more distinct alleles are called among the
        included samples, which are all of them unless ``include_samples`` / ``exclude_samples`` or a
        parser's populations restrict them. An alternate allele that the ``ALT`` field declares but no
        included sample carries therefore does not make a site poly-allelic.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is not poly-allelic, ``False`` otherwise.
        """
        # with at most one alternate allele no subset of samples can carry three alleles, so the site is kept
        # without decoding any genotype. This settles the monomorphic and bi-allelic bulk of a typical input
        if len(variant.ALT) < 2:
            return True

        # a dummy site carries no genotypes to count, so the declared alleles are all there is to go on
        if isinstance(variant, DummyVariant):
            return False

        # likewise where the input carries no sample that could leave an alternate allele uncarried
        if self._use_declared_alleles:
            return False

        # count the alleles called among the included samples. Building and splitting the genotype strings
        # dominates this check, so read the numeric calls where the backend provides them
        site = SiteAlleles.from_site(variant)

        if site is not None:
            return len(site.distinct(self._samples_mask)) < 3

        # the numeric genotype codes still settle the homozygous reference samples without decoding them
        types = getattr(variant, 'gt_types', None)

        if types is None:
            return len(get_distinct_called_alleles(self._select(variant.gt_bases))) < 3

        # a homozygous reference call has every called allele equal to the reference, so it contributes the
        # reference allele and nothing else and need not be decoded. Any other code, including a partially
        # missing call, may carry an alternate allele that the code does not identify, so it is decoded
        hom_ref = np.asarray(types) == HOM_REF
        included = self._samples_mask if self._samples_mask is not None else np.ones(hom_ref.shape, dtype=bool)

        alleles = get_distinct_called_alleles(np.asarray(variant.gt_bases)[included & ~hom_ref])

        if (included & hom_ref).any():
            # route the reference through the same helper so it is subject to the same allele validity test
            alleles = alleles | get_distinct_called_alleles([variant.REF or ''])

        return len(alleles) < 3


class AllFiltration(Filtration):
    """
    Filter out all sites. Only useful for testing purposes.
    """

    @_count_filtered
    def filter_site(self, variant: Site) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``False``.
        """
        return False


class NoFiltration(Filtration):
    """
    Do not filter out any sites. Only useful for testing purposes.
    """

    @_count_filtered
    def filter_site(self, variant: Site) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True``.
        """
        return True


class CodingSequenceFiltration(Filtration):
    """
    Filter out sites that are not in coding sequences. This filter should find frequent use when parsing
    spectra for which only sites in coding sequences should be considered.
    By using it, the annotation and parsing of unnecessary sites can be avoided which increases the speed.
    Note that we assume here that within contigs, sites in the GFF file are sorted by position in ascending order.

    For this filtration to work, we require a GFF file (passed to :class:`~sfsutils.parser.Parser` or
    :class:`~sfsutils.filtration.Filterer`).
    """

    #: The coding sequence index per contig, rebuilt on setup so a restored instance has one.
    _indexes: Dict[str, _CDSIndex] = {}

    def __init__(self):
        """
        Create a new filtration instance.
        """
        Filtration.__init__(self)

        #: The coding sequence enclosing the current variant or the closest one downstream.
        self.cd: Optional[pd.Series] = None

        #: The number of processed sites.
        self.n_processed: int = 0

        #: The coding sequence index per contig.
        self._indexes: Dict[str, _CDSIndex] = {}

    def _setup(self, handler: MultiHandler):
        """
        Touch the GFF file to load it.

        :param handler: The handler.
        """
        # require GFF file
        handler._require_gff(self.__class__.__name__)

        # the indexes are tied to the handler's coding sequences
        self._indexes = {}

        # setup GFF handler
        super()._setup(handler)

        # load coding sequences
        _ = handler._cds

    def _get_index(self, chrom: str, aliases: List[str]) -> _CDSIndex:
        """
        Get the coding sequence index for the given contig.

        :param chrom: The contig of the current variant.
        :param aliases: The aliases of the contig.
        :return: The index.
        """
        if chrom not in self._indexes:
            cds = self._handler._cds

            self._indexes[chrom] = _CDSIndex(cds[cds.seqid.isin(aliases)])

        return self._indexes[chrom]

    def _rewind(self):
        """
        Rewind the filtration.
        """
        super()._rewind()

        # reset coding sequence
        self.cd = None

        # the count guards the warning about a GFF whose contigs do not match the input, which must be
        # reachable on every pass rather than only on the first one of an instance's lifetime
        self.n_processed = 0

    @_count_filtered
    def filter_site(self, v: Site) -> bool:
        """
        Filter site by whether it is in a coding sequence.

        :param v: The variant to filter.
        :return: ``True`` if the variant is in a coding sequence, ``False`` otherwise.
        """
        aliases = self._handler.get_aliases(v.CHROM)

        # if self.cd is None or not on the same chromosome or ends before the variant
        if self.cd is None or self.cd.seqid not in aliases or v.POS > self.cd.end:

            # initialize mock coding sequence
            self.cd = pd.Series({
                'seqid': v.CHROM,
                'start': DegeneracyAnnotation._pos_mock,
                'end': DegeneracyAnnotation._pos_mock
            })

            # find the first coding sequence reaching the variant, by binary search: scanning the whole
            # frame here costs a pass over every coding sequence of the input for each advancing site
            index = self._get_index(v.CHROM, aliases)

            row = index.locate(v.POS)

            if row is not None:
                self.cd = index.get(row)

                if self.cd.start == v.POS:
                    self._logger.debug(f'Found coding sequence for {v.CHROM}:{v.POS}.')
                else:
                    self._logger.debug(f'Found coding sequence downstream of {v.CHROM}:{v.POS}.')

            if self.n_processed == 0 and self.cd.start == DegeneracyAnnotation._pos_mock:
                self._logger.warning(f'No subsequent coding sequence found on the same contig as the first variant. '
                                     f'Please make sure this is the correct GFF file with contig names matching '
                                     f'the input. You can use the aliases parameter to match contig names.')

        self.n_processed += 1

        # check whether the variant is in the current coding sequence
        if self.cd is not None and self.cd.seqid in aliases and self.cd.start <= v.POS <= self.cd.end:
            return True

        return False


class DeviantOutgroupFiltration(Filtration):
    """
    Filter out sites where the major allele of the specified outgroup samples differs from the major
    allele of the ingroup samples.
    """

    def __init__(
            self,
            outgroups: List[str],
            ingroups: List[str] = None,
            strict_mode: bool = True,
            retain_monomorphic: bool = True
    ):
        """
        Construct DeviantOutgroupFiltration.

        :param outgroups: The name of the outgroup samples to consider.
        :param ingroups: The name of the ingroup samples to consider, defaults to all samples but the outgroups.
        :param strict_mode: Whether to filter out sites where no outgroup sample is present, defaults to ``True``.
        :param retain_monomorphic: Whether to retain monomorphic sites, defaults to ``True``, which is faster.
        """
        super().__init__()

        #: The ingroup samples.
        self.ingroups: List[str] | None = ingroups

        #: The outgroup samples.
        self.outgroups: List[str] = outgroups

        #: Whether to filter out sites where no outgroup sample is present.
        self.strict_mode: bool = strict_mode

        #: Whether to retain monomorphic sites.
        self.retain_monomorphic: bool = retain_monomorphic

        #: The samples found in the input.
        self.samples: Optional[np.ndarray] = None

        #: The ingroup mask.
        self.ingroup_mask: Optional[np.ndarray] = None

        #: The outgroup mask.
        self.outgroup_mask: Optional[np.ndarray] = None

    def _setup(self, handler: MultiHandler):
        """
        Touch the reader to load the samples.

        :param handler: The handler.
        """
        super()._setup(handler)

        # create samples array
        self.samples: np.ndarray = np.array(handler._reader.samples)

        # create ingroup and outgroup masks
        self._create_masks()

    def _create_masks(self):
        """
        Create ingroup and outgroup masks based on the samples.
        """

        # create outgroup masks
        self.outgroup_mask: np.ndarray = np.isin(self.samples, self.outgroups)

        # make sure all outgroups are present
        if self.outgroup_mask.sum() != len(self.outgroups):
            raise ValueError(f'Not all outgroup samples are present in the input: {self.outgroups}')

        # create ingroup mask
        if self.ingroups is None:
            self.ingroup_mask = ~self.outgroup_mask
        else:
            self.ingroup_mask = np.isin(self.samples, self.ingroups)

            # make sure all ingroups are present, as an unmatched name would silently shrink the ingroup
            # and, in the extreme, filter out every polymorphic site
            if self.ingroup_mask.sum() != len(self.ingroups):
                raise ValueError(f'Not all ingroup samples are present in the input: {self.ingroups}')

    @staticmethod
    def _get_major_allele(site: SiteAlleles, mask: np.ndarray) -> str | None:
        """
        Get the majority allele among the selected samples from the numeric calls. A multi-character
        allele is counted once per haplotype carrying it rather than once per base, so an ``AT`` call
        weighs the same as an ``A`` one.

        :param site: The numeric view of the site's genotypes.
        :param mask: The boolean samples mask.
        :return: The majority allele, or ``None`` where no selected sample is called.
        """
        counts = site.counts(mask)

        if not counts:
            return None

        n_max = max(counts.values())
        tied = {allele for allele, count in counts.items() if count == n_max}

        if len(tied) == 1:
            return next(iter(tied))

        # break the tie by the first haplotype carrying one of the tied alleles, which is the order in
        # which the genotype strings would present the bases
        for code in np.asarray(site.indices)[mask].ravel():
            if 0 <= code < len(site.alleles) and site.alleles[code] in tied:
                return site.alleles[code]

    @_count_filtered
    def filter_site(self, variant: Site) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        # keep monomorphic sites if requested
        if not variant.is_snp and self.retain_monomorphic:
            return True

        # a dummy target site is all-ancestral by construction, so its ingroup and outgroup major bases agree
        # and this filter's own criterion keeps it. Dropping it would leave the TargetSiteCounter with no
        # sampled monomorphic sites at all, making n_target_sites a no-op.
        if isinstance(variant, DummyVariant):
            return True

        # the genotype strings render a haploid call of the third or a later allele as missing wherever the
        # site's maximum ploidy is two, and they render an MNP or an indel as several characters rather than
        # one allele, both of which the backends disagree on. Read the numeric calls where the backend
        # provides them, whatever the alleles look like
        site = SiteAlleles.from_site(variant)

        if site is not None:
            ingroup_base = self._get_major_allele(site, self.ingroup_mask)
            outgroup_base = self._get_major_allele(site, self.outgroup_mask)
        else:
            # get major base among ingroup samples
            ingroup_base = get_major_base(variant.gt_bases[self.ingroup_mask])

            # get major base among outgroup samples
            outgroup_base = get_major_base(variant.gt_bases[self.outgroup_mask])

        # filter out if no outgroup base is present and strict mode is enabled
        if outgroup_base is None:
            return not self.strict_mode

        # filter out if outgroup base is different from ingroup base
        return ingroup_base == outgroup_base


class ExistingOutgroupFiltration(Filtration):
    """
    Filter out sites for which at least ``n_missing`` of the specified outgroup samples have no called base.
    """

    #: The row of each outgroup sample, rebuilt on setup so a restored instance has one.
    _outgroup_rows: Optional[np.ndarray] = None

    def __init__(self, outgroups: List[str], n_missing: int = 1):
        """
        Construct ExistingOutgroupFiltration.

        :param outgroups: The names of the outgroup samples considered.
        :param n_missing: The number of outgroup samples that need to be missing to fail the filter.
        """
        super().__init__()

        #: The outgroup samples.
        self.outgroups: List[str] = outgroups

        #: Minimum number of missing outgroups required to filter out a site.
        self.n_missing: int = n_missing

        #: The samples found in the input.
        self.samples: Optional[np.ndarray] = None

        #: The outgroup mask.
        self.outgroup_mask: Optional[np.ndarray] = None

        #: The row of each outgroup sample.
        self._outgroup_rows: Optional[np.ndarray] = None

    def _setup(self, handler: MultiHandler):
        """
        Touch the reader to load the samples.

        :param handler: The handler.
        """
        super()._setup(handler)

        # create samples array
        self.samples: np.ndarray = np.array(handler._reader.samples)

        # create outgroup mask
        self._create_mask()

    def _create_mask(self):
        """
        Create outgroup mask based on the samples.
        """
        self.outgroup_mask: np.ndarray = np.isin(self.samples, self.outgroups)

        # make sure all outgroups are present, as an unmatched name would turn this filtration into a no-op
        if self.outgroup_mask.sum() != len(self.outgroups):
            raise ValueError(f'Not all outgroup samples are present in the input: {self.outgroups}')

        # the outgroups are counted one sample at a time, so hold their rows to select them individually
        self._outgroup_rows = np.flatnonzero(self.outgroup_mask)

    @_count_filtered
    def filter_site(self, variant: Site) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        # keep dummy variants
        if isinstance(variant, DummyVariant):
            return True

        # the genotype strings render a haploid call of the third or a later allele as missing wherever the
        # site's maximum ploidy is two, which would count a called outgroup as absent, so read the numeric
        # calls where the backend provides them
        site = SiteAlleles.from_site(variant)

        if site is not None:
            rows = self._outgroup_rows if self._outgroup_rows is not None else np.flatnonzero(self.outgroup_mask)

            indices = np.asarray(site.indices)

            if indices.ndim == 1:
                indices = indices[:, None]

            # settle every outgroup in one pass over their rows: asking the view sample by sample re-bins
            # the whole row for each of them, which costs a bincount per outgroup and per site
            codes = indices[rows]
            called = site._called[:len(site.alleles)]

            in_range = (codes >= 0) & (codes < called.size)
            valid = np.zeros(codes.shape, dtype=bool)
            valid[in_range] = called[codes[in_range]]

            # count how many outgroups have no called haplotype
            missing_count = int((~valid.any(axis=1)).sum())

            return missing_count < self.n_missing

        # get outgroup genotypes
        outgroups = variant.gt_bases[self.outgroup_mask]

        # count how many outgroups have no called base
        missing_count = sum(len(get_called_bases(outgroup)) == 0 for outgroup in outgroups)

        # filter out if at least n outgroups are missing
        return missing_count < self.n_missing


class BiasedGCConversionFiltration(Filtration):
    """
    Only retain A<->T and G<->C substitutions (which are unaffected
    by biased gene conversion, see [CITGB]_).

    Mono-allelic sites are always retained, and we assume sites are at most bi-allelic. Note that the number of
    mutational target sites is reduced by this filtration.

    .. [CITGB] Pouyet et al., 'Background selection and biased
        gene conversion affect more than 95% of the human genome and bias demographic inferences.',
        Elife, 7:e36317, 2018
    """

    @_count_filtered
    def filter_site(self, variant: Site) -> bool:
        """
        Remove bi-allelic sites that are not A<->T or G<->C mutations.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        if variant.is_snp and len(variant.ALT) > 0:
            return (variant.REF, variant.ALT[0]) in [('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')]

        return True


class CpGFiltration(Filtration):
    """
    Filter out sites whose reference base is in a CpG dinucleotide context. CpG sites are hypermutable
    (the cytosine is prone to deamination), so they are commonly excluded to
    avoid mutation-rate heterogeneity. A site is in CpG context iff:

    - the reference base is ``C`` and the next base on the same strand is ``G``, or
    - the reference base is ``G`` and the previous base on the same strand is ``C``.

    Like :class:`CodingSequenceFiltration`, this filtration requires a FASTA reference (passed to
    :class:`~sfsutils.parser.Parser` or :class:`~sfsutils.filtration.Filterer`), which is used for the
    ``±1`` base lookup. Sites on a contig the FASTA carries no sequence for, and sites the FASTA sequence
    does not reach, cannot be typed and are kept, with one warning per contig.
    """

    #: The contigs the FASTA carries no sequence for, held at class level so a restored instance has one.
    _missing_contigs: Set[str] = set()

    #: The contigs whose FASTA sequence is too short, held at class level so a restored instance has one.
    _short_contigs: Set[str] = set()

    def __init__(self):
        """
        Create a new filtration instance.
        """
        super().__init__()

        #: The contigs the FASTA carries no sequence for, warned about once each.
        self._missing_contigs: Set[str] = set()

        #: The contigs whose FASTA sequence does not reach every site, warned about once each.
        self._short_contigs: Set[str] = set()

    def _setup(self, handler: MultiHandler):
        """
        Require a FASTA file on the parent handler.

        :param handler: The handler.
        """
        # require FASTA file
        handler._require_fasta(self.__class__.__name__)

        self._missing_contigs = set()
        self._short_contigs = set()

        super()._setup(handler)

    @staticmethod
    def _is_cpg(contig, pos: int, ref: str) -> Optional[bool]:
        """
        Whether the reference base at ``pos`` (1-based) sits in a CpG dinucleotide context.

        :param contig: The reference sequence record for the variant's contig.
        :param pos: The 1-based position of the reference base.
        :param ref: The reference base.
        :return: Whether the site is in CpG context, or ``None`` where the FASTA sequence does not reach
            the position and the context cannot be determined.
        """
        i = pos - 1  # 0-based index of the reference base

        # a FASTA record shorter than the contig the input declares leaves the site itself outside the
        # sequence, which no neighbour lookup can be based on and which Bio.SeqRecord raises over
        if not 0 <= i < len(contig):
            return None

        # a site at either end of the sequence has no neighbour on that side, so it is not in CpG context
        if ref == 'C':
            return i + 1 < len(contig) and str(contig[i + 1]).upper() == 'G'

        return i - 1 >= 0 and str(contig[i - 1]).upper() == 'C'

    @_count_filtered
    def filter_site(self, variant: Site) -> bool:
        """
        Filter site by whether its reference base is in a CpG context.

        :param variant: The variant to filter.
        :return: ``True`` if the site should be kept (not CpG), ``False`` otherwise.
        """
        ref = (variant.REF or '').upper()

        # only C/G reference bases can be in CpG context
        if ref not in ('C', 'G'):
            return True

        # fetch the variant's contig from the handler's FASTA (cached across calls). A contig the FASTA does
        # not carry leaves the context unknown, which must not abort the whole run over one absent scaffold,
        # so the site is kept, as every other consumer of get_contig treats a missing contig as a skip
        try:
            contig = self._handler.get_contig(self._handler.get_aliases(variant.CHROM))
        except LookupError as e:
            if variant.CHROM not in self._missing_contigs:
                self._missing_contigs.add(variant.CHROM)
                self._logger.warning(f'Retaining sites on contig {variant.CHROM} unchecked: {e}')

            return True

        cpg = self._is_cpg(contig, variant.POS, ref)

        # a site the sequence does not reach is treated like one on an absent contig, keeping it rather
        # than aborting the whole run over a reference that is truncated or does not match the input
        if cpg is None:
            if variant.CHROM not in self._short_contigs:
                self._short_contigs.add(variant.CHROM)
                self._logger.warning(f'Retaining sites beyond the end of contig {variant.CHROM} unchecked: the '
                                     f'FASTA sequence is {len(contig)} bases long. Are you sure the reference '
                                     f'matches the input?')

            return True

        return not cpg


class ContigFiltration(Filtration):
    """
    Filter out sites that are not on the specified contigs.
    """

    def __init__(self, contigs: List[str]):
        """
        Construct ContigFiltration.

        :param contigs: The contigs to retain.
        """
        super().__init__()

        #: The contigs to retain.
        self.contigs: List[str] = contigs

    @_count_filtered
    def filter_site(self, variant: Site) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is on one of the specified contigs, ``False`` otherwise.
        """
        # match through the handler's aliases (get_aliases returns [CHROM] when unaliased), so a
        # ``chr21`` vs ``21`` naming difference does not silently drop every site
        aliases = self._handler.get_aliases(variant.CHROM) if self._handler is not None else [variant.CHROM]
        return any(alias in self.contigs for alias in aliases)


class Filterer(MultiHandler):
    """
    Filter the input using a list of filtrations.

    Example usage:

    ::

        import sfsutils as su

        # only keep variants in coding sequences
        f = su.Filterer(
            source="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
                "1000_genomes_project/release/20181203_biallelic_SNV/"
                "ALL.chr21.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
            gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
            output='sapiens.chr21.coding.vcf.gz',
            filtrations=[su.CodingSequenceFiltration()],
            aliases=dict(chr21=['21'])
        )

        f.filter()

    """

    def __init__(
            self,
            source: "str | os.PathLike | 'tskit.TreeSequence' | VariantReader | Iterable[Site] | None" = None,
            output: str = None,
            gff: str | None = None,
            fasta: str | None = None,
            filtrations: List[Filtration] = [],
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {},
            vcf: "str | os.PathLike | 'tskit.TreeSequence' | VariantReader | Iterable[Site] | None" = None
    ):
        """
        Create a new filter instance.

        :param source: The variant source: a VCF file (gzipped or a URL), a VCF-Zarr store (a ``.vcz`` or
            ``.zarr`` directory), a tskit tree sequence (a ``.trees`` file or an in-memory
            ``tskit.TreeSequence``), or a pre-built :class:`~sfsutils.io_handlers.VariantReader` / iterable of
            sites. Read through the same streamed site interface as all handlers.
        :param output: The output file.
        :param gff: The GFF file, possibly gzipped or a URL. This argument is required for some filtrations.
        :param fasta: The FASTA reference file, possibly gzipped or a URL. This argument is required for
            filtrations that depend on the reference sequence (e.g. base context).
        :param filtrations: The filtrations.
        :param info_ancestral: The info field for the ancestral allele.
        :param max_sites: The maximum number of sites to process.
        :param seed: The seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files downloaded from urls.
        :param aliases: Dictionary of aliases for the contigs in the input, e.g. ``{'chr1': ['1']}``.
        :param vcf: Deprecated alias for ``source``, kept for backward compatibility. Provide either
            ``source`` or ``vcf``, not both.
        :raises ValueError: If ``max_sites`` is not positive.
        """
        if max_sites <= 0:
            raise ValueError(f'max_sites must be positive, got {max_sites}.')

        super().__init__(
            source=source,
            vcf=vcf,
            gff=gff,
            fasta=fasta,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed,
            cache=cache,
            aliases=aliases
        )

        #: The filtrations.
        self.filtrations: List[Filtration] = filtrations

        #: The output file.
        self.output: str = output

        #: The number of sites that did not pass the filters.
        self.n_filtered: int = 0

        #: The variant writer (format chosen by the output extension).
        self._writer: VariantWriter | None = None

    def is_filtered(self, variant: Site) -> bool:
        """
        Whether the given variant is kept.

        :param variant: The variant to check.
        :return: ``True`` if the variant is kept, ``False`` otherwise.
        """
        # filter the variant
        for filtration in self.filtrations:
            if not filtration.filter_site(variant):
                self.n_filtered += 1
                return False

        return True

    def _setup(self):
        """
        Set up the filtrations.
        """
        # setup filtrations
        for f in self.filtrations:
            f._setup(self)

        # create the writer for the format implied by the output extension
        self._writer = VariantWriter.open(self.output, self._reader, info_ancestral=self.info_ancestral)

    def _teardown(self):
        """
        Tear down the filtrations.
        """
        for f in self.filtrations:
            f._teardown()
            f._rewind()

        # close the writer and reader (guarded so an error mid-setup still releases what was opened).
        # _reader is a cached_property, so only close it when it was actually opened, checking the cache
        # directly rather than via hasattr, which would trigger a spurious open.
        if self._writer is not None:
            self._writer.close()
        if '_reader' in self.__dict__:
            self._reader.close()

    def filter(self):
        """
        Filter the input.
        """
        self._logger.info('Start filtering')

        # discard the reader a previous pass left behind in the cache, and reset the count it left filled,
        # so that a second call starts at the first record rather than writing a header and raising on it.
        # Release it first, as the cache is the only reference to it and a pass that raised before its
        # teardown left it open. The FASTA and GFF caches are kept, as they do not depend on the pass
        VCFHandler._rewind(self)

        self.n_filtered = 0

        # tear down (closing the writer/reader) even if setup or iteration raises, so a failure after the
        # reader is opened but before/within writing does not leak the open reader and the output is flushed
        try:
            # setup filtrations
            self._setup()

            # get progress bar
            with self.get_pbar(desc=f"{self.__class__.__name__}>Processing sites") as pbar:

                # iterate over the sites
                for i, variant in enumerate(self._reader):

                    if self.is_filtered(variant):
                        # write the variant
                        self._writer.write(variant)

                    pbar.update()

                    # explicitly stopping after ``n`` sites fixes a bug with cyvcf2:
                    # 'error parsing variant with `htslib::bcf_read` error-code: 0 and ret: -2'
                    if i + 1 == self.n_sites or i + 1 >= self.max_sites:
                        break
        finally:
            # teardown filtrations
            self._teardown()

        self._logger.info(f'Filtered out {self.n_filtered} of {self.n_sites} sites in total.')
