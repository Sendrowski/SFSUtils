"""
A parser that extracts the site frequency spectrum (SFS) from a VCF file, a VCF-Zarr store, or a tskit
tree sequence (ARG). Stratifying the SFS is supported by providing a list of :class:`Stratification`
instances.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-26"

import functools
import itertools
import logging
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from typing import List, Callable, Literal, Optional, Dict, Tuple

import numpy as np
from Bio.SeqRecord import SeqRecord
from scipy.stats import hypergeom
from tqdm import tqdm

from .annotation import Annotation, SynonymyAnnotation, DegeneracyAnnotation, AncestralAlleleAnnotation
from .filtration import Filtration, PolyAllelicFiltration, SNPFiltration
from .io_handlers import bases, get_called_bases, FASTAHandler, NoTypeException, \
    DummyVariant, Site, MultiHandler, VCFHandler, VariantReader, is_monomorphic_snp
from .settings import Settings
from .spectrum import Spectra, TwoSFS, TwoSpectra, JointSFS, JointSpectra

# logger
logger = logging.getLogger('sfsutils')


def _count_valid_type(func: Callable) -> Callable:
    """
    Decorator for counting the number of sites that had a valid type.
    """

    @functools.wraps(func)
    def wrapper(self, variant: Site):
        """
        Wrapper function.

        :param self: Class instance
        :param variant: The site
        :return: The result of the decorated function
        """
        res = func(self, variant)
        self.n_valid += 1
        return res

    return wrapper


class Stratification(ABC):
    """
    Abstract class for Stratifying the SFS by determining a site's type based on its properties.
    """

    def __init__(self):
        """
        Create instance.
        """
        self._logger = logger.getChild(self.__class__.__name__)

        #: Parser instance
        self.parser: Optional['Parser'] = None

        #: The number of sites that didn't have a type.
        self.n_valid: int = 0

    def _setup(self, parser: 'Parser'):
        """
        Provide the stratification with some context by specifying the parser.
        This should be done before calling :meth:`get_type`.

        :param parser: The parser
        """
        self.parser = parser

    def _rewind(self):
        """
        Rewind the stratification.
        """
        self.n_valid = 0

    def _teardown(self):
        """
        Perform any necessary post-processing.
        """
        self._logger.info(f"Number of sites with valid type: {self.n_valid}")

    @abstractmethod
    def get_type(self, variant: Site) -> Optional[str]:
        """
        Get type of given Variant. Only the types
        given by :meth:`get_types()` are valid, or ``None`` if
        no type could be determined.

        :param variant: The site
        :return: Type of the variant
        """
        pass

    @abstractmethod
    def get_types(self) -> List[str]:
        """
        Get all possible types.

        :return: List of types
        """
        pass


class SNPStratification(Stratification, ABC):
    """
    Abstract class for stratifications that can only handle SNPs. We need to issue a warning in this case.
    """

    def _setup(self, parser: 'Parser'):
        """
        Set up the stratification.

        :param parser: The parser
        """
        super()._setup(parser)

        # issue warning if we have an SNP stratification
        self._logger.warning(f"{self.__class__.__name__} can only handle SNPs and not mono-allelic sites. "
                             "This means you have to update the number of mono-allelic sites manually.")


class BaseContextStratification(Stratification, FASTAHandler):
    """
    Stratify the SFS by the base context of the mutation. The number of flanking bases
    can be configured. Note that we attempt to take the ancestral allele as the
    middle base. If ``skip_non_polarized`` is set to ``False``, we use the reference
    allele as the middle base.
    """

    def __init__(
            self,
            fasta: str,
            n_flanking: int = 1,
            aliases: Dict[str, List[str]] = {},
            cache: bool = True
    ):
        """
        Create instance. Note that we require a fasta file to be specified
        for base context to be able to be inferred

        :param fasta: The fasta file path, possibly gzipped or a URL
        :param n_flanking: The number of flanking bases
        :param aliases: Dictionary of aliases for the contigs in the input, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the input with the contig names in the FASTA file and GFF file.
        :param cache: Whether to cache files that are downloaded from URLs
        """
        Stratification.__init__(self)

        FASTAHandler.__init__(self, fasta, cache=cache, aliases=aliases)

        #: The number of flanking bases
        self.n_flanking: int = n_flanking

        #: The current contig
        self.contig: Optional[SeqRecord] = None

    def _rewind(self):
        """
        Rewind the stratification.
        """
        Stratification._rewind(self)
        FASTAHandler._rewind(self)

        self.contig = None

    @_count_valid_type
    def get_type(self, variant: Site) -> str:
        """
        Get the base context for a given mutation

        :param variant: The site
        :return: Base context of the mutation
        """
        pos = variant.POS - 1

        # get the ancestral allele
        aa = self.parser._get_ancestral(variant)

        # get aliases
        aliases = self.get_aliases(variant.CHROM)

        # check if contig is up-to-date
        if self.contig is None or self.contig.id not in aliases:
            self._logger.debug(f"Fetching contig '{variant.CHROM}'.")

            # fetch contig
            self.contig = self.get_contig(aliases)

        # check if position is valid
        if pos < 0 or pos >= len(self.contig):
            raise NoTypeException("Invalid position: Position must be within the bounds of the sequence.")

        # get upstream and downstream bases, upper-casing soft-masked (lowercase) reference bases so
        # they match the upper-case contexts in get_types()
        upstream_start = max(0, pos - self.n_flanking)
        upstream_bases = str(self.contig.seq[upstream_start:pos]).upper()

        downstream_end = min(len(self.contig), pos + self.n_flanking + 1)
        downstream_bases = str(self.contig.seq[pos + 1:downstream_end]).upper()

        context = f"{upstream_bases}{aa}{downstream_bases}"

        # near a contig edge the flanking window is clamped, giving a context shorter than the full
        # k-mer; such a truncated context is not among get_types() and would collide with others (a
        # start-edge and an end-edge context of the same letters share a key), so skip the site
        if len(context) != 2 * self.n_flanking + 1:
            raise NoTypeException(f"Base context '{context}' is truncated at a contig edge at "
                                  f"{variant.CHROM}:{variant.POS}")

        # a flanking base outside ACGT (e.g. an N at an assembly gap) has no valid context; skip the site
        # rather than letting it accrue into a spurious stratum
        if any(b not in bases for b in context):
            raise NoTypeException(f"Base context '{context}' contains a non-ACGT base at "
                                  f"{variant.CHROM}:{variant.POS}")

        return context

    def get_types(self) -> List[str]:
        """
        Create all possible base contexts.

        :return: List of contexts
        """
        return [''.join(c) for c in itertools.product(bases, repeat=2 * self.n_flanking + 1)]


class BaseTransitionStratification(SNPStratification):
    """
    Stratify the SFS by the base transition of the mutation, i.e., ``A>T``.

    .. warning::
        This stratification only works for SNPs. You thus need to update the number of mono-allelic sites manually.
    """

    @_count_valid_type
    def get_type(self, variant: Site) -> str:
        """
        Get the base transition for the given variant.

        :param variant: The site
        :return: Base transition
        :raises ~sfsutils.io_handlers.NoTypeException: if not type could be determined
        """
        if variant.is_snp:
            ancestral = self.parser._get_ancestral(variant)

            derived = variant.REF if variant.REF != ancestral else variant.ALT[0]

            if ancestral in bases and derived in bases and ancestral != derived:
                return f"{ancestral}>{derived}"

            raise NoTypeException("Not a valid base transition.")

        raise NoTypeException("Site is not a SNP.")

    def get_types(self) -> List[str]:
        """
        Get all possible base transitions.

        :return: List of contexts
        """
        return ['>'.join([a, b]) for a in bases for b in bases if a != b]


class TransitionTransversionStratification(BaseTransitionStratification):
    """
    Stratify the SFS by whether we have a transition or transversion.

    .. warning::
        This stratification only works for SNPs. You thus need to update the number of mono-allelic sites manually.
    """

    @_count_valid_type
    def get_type(self, variant: Site) -> str:
        """
        Get the mutation type (transition or transversion) for a given mutation.

        :param variant: The site
        :return: Mutation type
        """
        if variant.is_snp:

            if variant.ALT[0] not in bases:
                raise NoTypeException("Invalid alternate allele: Alternate allele must be a valid base.")

            if (variant.REF, variant.ALT[0]) in [('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')]:
                return "transition"
            else:
                return "transversion"

        raise NoTypeException("Site is not a SNP.")

    def get_types(self) -> List[str]:
        """
        All possible mutation types (transition and transversion).

        :return: List of mutation types
        """
        return ["transition", "transversion"]


class AncestralBaseStratification(Stratification):
    """
    Stratify the SFS by the base context of the mutation: the reference base.
    If ``skip_non_polarized`` is set to ``False``, we use the reference allele as
    ancestral base. By default, we use the ``AA`` tag to determine the ancestral allele.

    Any subclass of :class:`~sfsutils.annotation.AncestralAlleleAnnotation` can be used to annotate the ancestral allele.
    """

    @_count_valid_type
    def get_type(self, variant: Site) -> str:
        """
        Get the type which is the reference allele.

        :param variant: The site
        :return: reference allele
        """
        return self.parser._get_ancestral(variant)

    def get_types(self) -> List[str]:
        """
        The possible base types.

        :return: List of contexts
        """
        return bases


class DegeneracyStratification(Stratification):
    """
    Stratify SFS by degeneracy. We only consider sides which 4-fold degenerate (neutral) or
    0-fold degenerate (selected) which facilitates counting.

    :class:`~sfsutils.annotation.DegeneracyAnnotation` can be used to annotate the degeneracy of a site.
    """

    def __init__(
            self,
            custom_callback: Callable[['cyvcf2.Variant'], str] = None,
    ):
        """
        Initialize the stratification.

        :param custom_callback: Custom callback to determine the type of mutation
        """
        super().__init__()

        #: Custom callback to determine the degeneracy of mutation
        self.get_degeneracy = custom_callback if custom_callback is not None else self._get_degeneracy_default

    @staticmethod
    def _get_degeneracy_default(
            variant: Site
    ) -> Optional[Literal['neutral', 'selected']]:
        """
        Get degeneracy based on 'Degeneracy' tag.

        :param variant: The site
        :return: Type of the mutation
        """
        degeneracy = variant.INFO.get('Degeneracy')

        if degeneracy is None:
            raise NoTypeException("No degeneracy tag found.")
        else:
            if degeneracy == 4:
                return 'neutral'

            if degeneracy == 0:
                return 'selected'

            raise NoTypeException(f"Degeneracy tag has invalid value: '{degeneracy}' at {variant.CHROM}:{variant.POS}")

    @_count_valid_type
    def get_type(self, variant: Site) -> Literal['neutral', 'selected']:
        """
        Get the degeneracy.

        :param variant: The site
        :return: Type of the mutation
        :raises ~sfsutils.io_handlers.NoTypeException: If the mutation is not synonymous or non-synonymous
        """
        return self.get_degeneracy(variant)

    def get_types(self) -> List[str]:
        """
        Get all possible degeneracy type (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']


class SynonymyStratification(SNPStratification):
    """
    Stratify SFS by synonymy (neutral or selected).

    :class:`~sfsutils.annotation.SynonymyAnnotation` can be used to annotate the synonymy of a site.

    .. warning::
        This stratification only works for SNPs. You thus need to update the number of mono-allelic sites manually.
    """

    def get_types(self) -> List[str]:
        """
        Get all possible synonymy types (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']

    @_count_valid_type
    def get_type(self, variant: Site) -> Literal['neutral', 'selected']:
        """
        Get the synonymy using the custom synonymy annotation.

        :param variant: The site
        :return: Type of the mutation, either ``neutral`` or ``selected``
        """
        synonymy = variant.INFO.get('Synonymy')

        if synonymy is None:
            raise NoTypeException("No synonymy tag found.")
        else:
            if synonymy == 1:
                return 'neutral'

            if synonymy == 0:
                return 'selected'

            raise NoTypeException(f"Synonymy tag has invalid value: '{synonymy}' at {variant.CHROM}:{variant.POS}")


class VEPStratification(SynonymyStratification):
    """
    Stratify SFS by synonymy (neutral or selected) based on annotation provided by VEP.

    .. warning::
        This stratification only works for SNPs. You thus need to update the number of mono-allelic sites manually.
    """

    #: The tag used by VEP to annotate the synonymy
    info_tag: str = 'CSQ'

    def get_types(self) -> List[str]:
        """
        Get all possible synonymy types (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']

    @_count_valid_type
    def get_type(self, variant: Site) -> Literal['neutral', 'selected']:
        """
        Get the synonymy of a site.

        :param variant: The site
        :return: Type of the mutation, either ``neutral`` or ``selected``
        """
        synonymy = variant.INFO.get(self.info_tag, '')

        if 'synonymous_variant' in synonymy:
            return 'neutral'

        if 'missense_variant' in synonymy:
            return 'selected'

        raise NoTypeException(f"Synonymy tag has invalid value: '{synonymy}' at {variant.CHROM}:{variant.POS}")


class SnpEffStratification(VEPStratification):
    """
    Stratify SFS by synonymy (neutral or selected) based on annotation provided by SnpEff.

    .. warning::
        This stratification only works for SNPs. You thus need to update the number of mono-allelic sites manually.
    """

    #: The tag used by SnpEff to annotate the synonymy
    info_tag: str = 'ANN'


class GenomePositionDependentStratification(Stratification, ABC):
    """
    Base class for stratifications that derive the type from a site's genomic position.
    """


class ContigStratification(GenomePositionDependentStratification):
    """
    Stratify SFS by contig.
    """

    def __init__(self, contigs: List[str] = None):
        """
        Initialize the stratification.

        :param contigs: List of contigs to stratify by. Defaults to all contigs in the input.
        """
        super().__init__()

        #: List of contigs
        self.contigs: List[str] = contigs

    @_count_valid_type
    def get_type(self, variant: Site) -> str:
        """
        Get the contig.

        :param variant: The site
        :return: The contig name
        """
        if self.contigs is not None and variant.CHROM not in self.contigs:
            raise NoTypeException(f"Contig '{variant.CHROM}' not in list of contigs.")

        return variant.CHROM

    def get_types(self) -> List[str]:
        """
        Get all possible contig type.

        :return: List of contexts
        """
        return self.contigs or list(self.parser._reader.seqnames)


class ChunkedStratification(GenomePositionDependentStratification):
    """
    Stratify SFS by creating ``n`` contiguous chunks of roughly equal size.

    .. note::
        Since the total number of sites is not known in advance, we cannot create contiguous
        chunks of exactly equal size.

    .. warning::
        Chunk boundaries are sized from the raw input record count (``parser.n_sites``), but a site
        is assigned to a chunk only once it has survived filtration and down-projection. When any
        filtration or projection drops sites, all included sites fall within the first
        ``n_included / n_sites`` fraction of the record range, so they concentrate in the leading
        chunks and the trailing chunks come out under-filled or empty. The included count is not known
        at setup without an extra pass over the data, so the chunks cannot be pre-balanced; a warning
        is logged at setup when the parser carries filtrations. Sites seen beyond the last boundary
        (for instance during a second, target-site sampling pass) are folded into the final chunk.
    """

    def __init__(self, n_chunks: int):
        """
        Initialize the stratification.

        :param n_chunks: Number of sites per window
        """
        super().__init__()

        #: Number of chunks
        self.n_chunks: int = int(n_chunks)

        #: List of chunk sizes
        self.chunk_sizes: Optional[List[int]] = None

        #: Number of sites seen so far
        self.counter: int = 0

    def _setup(self, parser: 'Parser'):
        """
        Set up the stratification.

        :param parser: The parser
        """
        super()._setup(parser)

        # chunk boundaries are sized from the raw record count, but sites are assigned only after
        # surviving filtration and projection, so with active filtrations the included sites bunch into
        # the leading chunks and the trailing chunks come out under-filled or empty. The included count
        # is not available here without an extra pass, so warn rather than silently produce empty chunks.
        if parser.filtrations:
            self._logger.warning(
                f"ChunkedStratification sizes its {self.n_chunks} chunks from the raw input record count, but "
                f"{len(parser.filtrations)} filtration(s) are active and will drop sites before they are chunked. "
                f"Included sites will concentrate in the leading chunks and the trailing chunks may be empty. "
                f"To obtain balanced chunks, chunk a pre-filtered input or one whose records all pass filtration."
            )

        # compute base chunk size and remainder
        base_chunk_size, remainder = divmod(parser.n_sites, self.n_chunks)

        # create list of chunk sizes
        self.chunk_sizes = [base_chunk_size + (i < remainder) for i in range(self.n_chunks)]

    def _rewind(self):
        """
        Rewind the stratification, also resetting the per-pass site counter so a second pass
        (for example the :class:`TargetSiteCounter` sampling pass) restarts from the first chunk.
        """
        super()._rewind()
        self.counter = 0

    def get_types(self) -> List[str]:
        """
        Get all possible window types.

        :return: List of contexts
        """
        return [f'chunk{i}' for i in range(self.n_chunks)]

    @_count_valid_type
    def get_type(self, variant: Site) -> str:
        """
        Get the type.

        :param variant: The site
        :return: The type
        """
        # find the index of the chunk to which the current site belongs; a second pass may process
        # more sites than the first (e.g. target-site sampling), so fall back to the last chunk
        chunk_index = next(
            (i for i, size in enumerate(self.chunk_sizes) if self.counter < sum(self.chunk_sizes[:i + 1])),
            self.n_chunks - 1,
        )

        # get the type
        t = f'chunk{chunk_index}'

        # update the counter
        self.counter += 1

        return t


class RandomStratification(Stratification):
    """
    Stratify the SFS randomly into a fixed number of bins.
    Can be used to analyze expected sampling variance between different stratifications.
    """

    def __init__(self, n_bins: int, seed: Optional[int] = 0):
        """
        Initialize random stratification.

        :param n_bins: Number of bins to randomly assign sites to.
        """
        super().__init__()

        if n_bins < 1:
            raise ValueError("n_bins must be at least 1.")

        #: Number of bins
        self.num_bins: int = n_bins

        #: Random seed for reproducibility
        self.seed: Optional[int] = seed

        #: Random generator instance
        self.rng = random.Random(seed)

    @_count_valid_type
    def get_type(self, variant: Site) -> str:
        """
        Assign the variant to a random bin.

        :param variant: The site
        :return: Randomly chosen bin label
        """
        return f"bin{self.rng.randint(0, self.num_bins - 1)}"

    def get_types(self) -> List[str]:
        """
        Get all possible bin labels.

        :return: List of bin labels
        """
        return [f"bin{i}" for i in range(self.num_bins)]


class TargetSiteCounter:
    """
    Class for counting the number of target sites when parsing an input that does not contain monomorphic sites.
    This class is used in conjunction with :class:`~sfsutils.parser.Parser` and samples sites from the given fasta
    file that are found in between variants on the same contig that were parsed from the input.
    Ideally, we obtain the SFS by parsing inputs that contain both mono- and polymorphic sites. This is because
    we need to know about the number of mutational opportunities for synonymous and non-synonymous sites which
    contain plenty of information on the strength of selection. It is recommended to use a :class:`~sfsutils.filtration.SNPFiltration` when
    using this class to avoid biasing the result by monomorphic sites present in the input.

    .. warning::
        This class is not compatible with stratifications based on info tags that are pre-defined in the input, as
        opposed to those added dynamically using the ``annotations`` argument of the parser. We also need to
        stratify mono-allelic sites which, in this case, won't be present in the input so that they have no
        info tags when sampling from the FASTA file, and are thus ignored by the stratifications. However, using the
        ``annotations`` argument of the parser, the info tags the stratifications are based on are added on-the-fly,
        also for monomorphic sites sampled from the FASTA file.

    .. note::
        With the unstratified two-SFS (``two_sfs=True``) this counter extrapolates the monomorphic-involving pairs
        from the target-site count. That anchors the marginal only approximately, so the resulting
        :meth:`~sfsutils.spectrum.TwoSFS.cov` / ``corr`` are approximate (a real all-sites input is preferred);
        :meth:`~sfsutils.spectrum.TwoSFS.fpmi` needs no monomorphic sites. The stratified two-SFS with a counter is
        not supported.
    """

    def __init__(
            self,
            n_target_sites: int,
            n_samples: int = int(1e5),
    ):
        """
        Initialize counter.

        :param n_target_sites: The total number of sites (mono- and polymorphic) that would be present in the input
            if it contained monomorphic sites. This number should be considerably larger than the number of polymorphic
            sites in the input. this value is not extremely important for downstream inference, the ratio of synonymous
            to non-synonymous sites being more informative, but the order of magnitude should be correct, in any case.
        :param n_samples: The number of sites to sample from the fasta file. Many sampled sites will not be valid as
            they are non-coding. To obtain good estimates, a few thousand sites should be sampled per type of site
            (depending on the stratifications used).
        """
        #: The logger
        self._logger = logger.getChild(self.__class__.__name__)

        #: The total number of sites considered when parsing the input
        self.n_target_sites: int | None = int(n_target_sites)

        #: Number of samples
        self.n_samples: int = int(n_samples)

        #: The spectra before inferring the number of target sites
        self._sfs_polymorphic: Spectra | None = None

    def _setup(self, parser: 'Parser'):
        """
        Set up the counter.

        :param parser: The parser
        """
        self.parser = parser

        # with the two-SFS a TargetSiteCounter extrapolates the monomorphic-involving pairs from the target-site
        # count (see _extrapolate_two_sfs), which makes the extrapolated cov()/corr() approximately usable on
        # SNP-only input (real all-sites input is preferred). fpmi() ignores the monomorphic bins and works regardless.
        # That extrapolation uses n_target_sites and the region length, not the FASTA, so only the single-SFS mode
        # (which samples monomorphic sites from the reference) actually requires it.
        if not self.parser.two_sfs:
            self.parser._require_fasta(self.__class__.__name__)

        # check if we have a SNPFiltration
        if not any([isinstance(f, SNPFiltration) for f in self.parser.filtrations]):
            self._logger.warning("It is recommended to use SNPFiltration together with "
                                 "TargetSiteCounter to avoid biasing the result by monomorphic sites.")

        # check if have degeneracy stratification but no degeneracy annotation
        if any([isinstance(s, DegeneracyStratification) for s in self.parser.stratifications]) \
                and not any([isinstance(a, DegeneracyAnnotation) for a in self.parser.annotations]):
            self._logger.warning("When using TargetSiteCounter with DegeneracyStratification, "
                                 "make sure to provide DegeneracyAnnotation to make sure the "
                                 "sites sampled from the FASTA file have a degeneracy tag.")

    def _teardown(self):
        """
        Perform any necessary post-processing.
        """
        # tear down parser
        self.parser._teardown()

    def _suspend_snp_filtration(self):
        """
        Suspend SNP filtration to make sure we sample can actually sample monomorphic sites.
        """
        # store original filtrations
        self._filtrations = self.parser.filtrations

        # remove SNPFiltration
        self.parser.filtrations = [f for f in self.parser.filtrations if not isinstance(f, SNPFiltration)]

    def _resume_snp_filtration(self):
        """
        Resume SNP filtration.
        """
        # restore original filtrations
        self.parser.filtrations = self._filtrations

    def count(self):
        """
        Count the number of target sites.

        :return: The number of target sites
        """
        # rewind parser components
        self.parser._rewind()

        # suspend SNP filtration
        self._suspend_snp_filtration()

        # rewind fasta iterator
        FASTAHandler._rewind(self.parser)

        # initialize random number generator
        rng = np.random.default_rng(self.parser.seed)

        # initialize progress bar
        pbar = tqdm(
            total=self.n_samples,
            desc=f'{self.__class__.__name__}>Sampling target sites',
            disable=Settings.disable_pbar
        )

        # get array of ranges per contig of parsed variants
        ranges = np.array(list(self.parser._contig_bounds.values()))

        # get range sizes
        range_sizes = ranges[:, 1] - ranges[:, 0]

        # every parsed contig spans a single position (e.g. one site per contig): there is no interval to
        # sample monomorphic sites from, so sample nothing rather than dividing by a zero total into NaNs
        total_range = np.sum(range_sizes)
        if total_range == 0:
            self._logger.info("No interval to sample target sites from (parsed sites span no interval); "
                              "skipping monomorphic sampling.")
            samples = np.zeros(len(range_sizes), dtype=int)
        else:
            # determine sampling probabilities and sample the number of sites per contig
            samples = rng.multinomial(self.n_samples, range_sizes / total_range)

        # keep track of SFS before update (a raw dict of joint arrays for joint parsing, else a Spectra); the
        # two-SFS counting pass only tallies per-stratum monomorphic sites and needs no pre-sampling snapshot
        if self.parser.two_sfs:
            self._sfs_polymorphic = None
        elif self.parser.pops is not None:
            self._sfs_polymorphic = {t: np.array(arr, dtype=float) for t, arr in self.parser.sfs.items()}
        else:
            self._sfs_polymorphic = Spectra(self.parser.sfs)

        # initialize counter
        i = 0

        # iterate over contigs
        for contig, bounds, n in zip(self.parser._contig_bounds.keys(), ranges, samples):

            # get aliases
            aliases = self.parser.get_aliases(contig)

            # make sure we have a valid range
            if bounds[1] > bounds[0] and n > 0:

                self._logger.debug(f"Sampling {n} sites from contig '{contig}'.")

                # fetch contig
                record = self.parser.get_contig(aliases, notify=False)

                # get positions
                # we sort in ascending order as the parser expects the positions to be sorted
                positions = np.sort(rng.integers(*bounds, size=n))

                # sample sites
                for pos in positions:

                    # create dummy variant (a monomorphic reference site); n_samples so its gt_bases
                    # aligns with the parser's sample masks, as the Site interface promises
                    variant = DummyVariant(
                        ref=record.seq[pos - 1].upper(),  # fasta is 0-based; upper-case soft-masked bases
                        pos=pos,  # VCF is 1-based
                        chrom=contig,
                        n_samples=len(self.parser._reader.samples),
                    )

                    # check if site was included in the SFS
                    if self.parser._process_site(variant):
                        i += 1

                    # update progress bar
                    pbar.update()

        # close progress bar
        pbar.close()

        # resume SNP filtration
        self._resume_snp_filtration()

        # tear down
        self._teardown()

        # notify on number of sites included in the SFS
        self._logger.info(f"{i} out of {self.n_samples} sampled sites were valid.")

    def _update_target_sites(self, spectra: Spectra) -> Spectra:
        """
        Update the target sites of the spectra.

        :param spectra: The spectra, including the sampled monomorphic sites.
        :return: The updated spectra.
        """
        # copy spectra
        spectra = spectra.copy()

        # cast to float to avoid implicit type conversion later on
        spectra.data = spectra.data.astype(float)

        # the sampling pass can introduce stratification types absent from the pre-sampling snapshot (a
        # stratum with no polymorphic site in the input but present among the sampled monomorphic sites).
        # Align the snapshot onto the post-sampling types with zeros, so the label-aligned subtractions
        # below do not write NaN into those types, mirroring the joint path's ``_before`` default.
        before = self._sfs_polymorphic.data.reindex(columns=spectra.data.columns, fill_value=0)
        before_n_polymorphic = self._sfs_polymorphic.n_polymorphic.reindex(spectra.data.columns, fill_value=0)

        # subtract by monomorphic counts of original spectra
        # we only want to consider the monomorphic sites sampled from the FASTA file
        spectra.data.iloc[[0, -1], :] -= before.iloc[[0, -1], :]

        # get number of monomorphic and polymorphic sites sampled from the FASTA and VCF file
        n_monomorphic = spectra.data.iloc[0, :].sum()
        n_polymorphic = spectra.data.iloc[1:, :].sum().sum()

        # check if we have enough target sites
        if self.n_target_sites < n_polymorphic:
            self._logger.warning(f"Number of polymorphic sites ({n_polymorphic}) exceeds the total "
                                 f"number of target sites ({self.n_target_sites}) which does not make sense. "
                                 f"The number of target sites unchanged is left unchanged.")
        elif n_monomorphic == 0:
            self._logger.warning(f"Number of monomorphic sites is zero which should only happen "
                                 f"if there are very few sites considered. Failed to adjust "
                                 f"the number of monomorphic sites.")
        else:

            # compute multiplicative factor to scale the total number of sites
            # to the number of target sites plus the number of polymorphic sites
            x = (self.n_target_sites + self._sfs_polymorphic.n_polymorphic.sum() - n_polymorphic) / n_monomorphic

            # extrapolate monomorphic counts using scaling factor
            spectra.data.iloc[0, :] *= x

            # subtract polymorphic counts from original spectra,
            # so that the total number of sites is equal to the number of target sites
            # we do this to correct for the fact that, for a type, we have relatively
            # fewer monomorphic sites if we have more polymorphic sites
            # TODO include monomorphic sites here from VCF?
            spectra.data.iloc[0, :] -= before_n_polymorphic

        return spectra

    def _update_target_sites_joint(self, sfs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extrapolate the monomorphic corner of a joint SFS. All monomorphic (all-ancestral) sites map to the single
        origin ``(0, ..., 0)`` of the joint spectrum, so the number of target sites fixes the mass placed there.
        The monomorphic budget ``n_target_sites - (observed sites)`` is distributed across stratification types in
        proportion to the valid monomorphic sites sampled from the FASTA file for each type.

        :param sfs: The per-type joint SFS after sampling (its origin holds the sampled monomorphic mass).
        :return: The per-type joint SFS with the origin set to the extrapolated monomorphic count.
        """
        origin = (0,) * len(self.parser._joint_shape)

        def _before(t: str) -> np.ndarray:
            # the type's polymorphic joint SFS from before sampling (zeros if the type only appears from sampling)
            return self._sfs_polymorphic.get(t, np.zeros(self.parser._joint_shape))

        # observed (polymorphic) mass, from before the monomorphic sampling
        n_observed = sum(float(arr.sum()) for arr in self._sfs_polymorphic.values())
        n_monomorphic = self.n_target_sites - n_observed

        if n_monomorphic <= 0:
            self._logger.warning(f"The number of target sites ({self.n_target_sites}) does not exceed the number "
                                 f"of observed sites ({n_observed:.0f}); the joint SFS is left unchanged.")
            # return the pre-sampling polymorphic spectrum; ``sfs`` has the sampled monomorphic mass at the origin
            return {t: np.array(_before(t), dtype=float) for t in sfs}

        # monomorphic mass sampled from the FASTA file per type (added at the origin during sampling)
        sampled = {t: float(sfs[t][origin] - _before(t)[origin]) for t in sfs}
        total_sampled = sum(sampled.values())

        updated = {}
        for t in sfs:
            arr = np.array(_before(t), dtype=float)
            fraction = sampled[t] / total_sampled if total_sampled > 0 else 1.0 / len(sfs)
            arr[origin] += n_monomorphic * fraction
            updated[t] = arr

        return updated

    def _extrapolate_two_sfs(
            self,
            two_sfs: np.ndarray,
            marginal: np.ndarray,
            region_length: float,
            distance: int,
    ) -> np.ndarray:
        """
        Add the monomorphic-involving pairs to a two-SFS by extrapolating from the target-site count.

        The observed matrix holds mainly the polymorphic-polymorphic pairs. Under a uniform site density a window of
        width ``distance`` holds on average ``rho_m * distance`` monomorphic sites, where ``rho_m = n_monomorphic /
        region_length`` and ``n_monomorphic = n_target_sites - n_polymorphic``. Each polymorphic site of derived
        count ``j`` therefore pairs with ``rho_m * distance`` monomorphic sites (the ``(0, j)`` and ``(j, 0)``
        entries) and each monomorphic site pairs with ``rho_m * distance`` others (the ``(0, 0)`` entry). The window
        offset drops out: a band of width ``distance`` holds the same expected number of sites regardless of where it
        starts. This anchors the marginal for :meth:`~sfsutils.spectrum.TwoSFS.cov` / ``corr`` only approximately;
        :meth:`~sfsutils.spectrum.TwoSFS.fpmi` needs no monomorphic sites and is preferred for SNP-only input.

        :param two_sfs: The symmetrized polymorphic-polymorphic two-SFS.
        :param marginal: The one-dimensional marginal of the polymorphic sites.
        :param region_length: The length (bp) over which the sites are distributed.
        :param distance: The two-SFS distance window width (bp).
        :return: The two-SFS with the monomorphic-involving pairs added.
        """
        two_sfs = np.array(two_sfs, dtype=float)
        # bins 0 (all-ancestral) and -1 (all-derived) are monomorphic; only 1..n-1 are polymorphic
        n_polymorphic = float(marginal[1:-1].sum())
        n_monomorphic = self.n_target_sites - n_polymorphic

        if n_monomorphic <= 0:
            self._logger.warning(f"The number of target sites ({self.n_target_sites}) does not exceed the number of "
                                 f"polymorphic sites ({n_polymorphic:.0f}); the two-SFS is left unchanged.")
            return two_sfs

        if region_length <= 0:
            self._logger.warning("The region length is zero; cannot extrapolate the monomorphic pairs of the two-SFS.")
            return two_sfs

        rho_m = n_monomorphic / region_length

        # monomorphic-polymorphic pairs: a polymorphic site of derived count j has rho_m * distance monomorphic
        # partners within the window; and monomorphic-monomorphic pairs for the (0, 0) entry.
        # the polymorphic-polymorphic block is stored half-averaged (parse applies symmetrize = (A + A.T) / 2),
        # so each unordered pair sits as count / 2 in both symmetric slots. To match that convention (downstream
        # TwoSFS._branch_length_covariance folds the matrix back with data + data.T), we add half the extrapolated
        # count to each symmetric slot rather than the full count, keeping the monomorphic entries on the same
        # scale as the poly-poly ones after the fold.
        contribution = marginal * rho_m * distance
        contribution[0] = 0.0
        contribution[-1] = 0.0
        two_sfs[0, :] += contribution / 2
        two_sfs[:, 0] += contribution / 2
        two_sfs[0, 0] += 0.5 * n_monomorphic * rho_m * distance

        return two_sfs


class Parser(MultiHandler):
    """
    Parse site-frequency spectra from VCF files, VCF-Zarr stores, or tskit tree sequences.

    By default, the parser looks at the ``AA`` tag in the VCF file's info field to retrieve
    the correct polarization. Polymorphic sites for which this tag is not well-defined are by default
    ignored (see :attr:`skip_non_polarized`).

    This class also offers on-the-fly annotation of the sites such as site degeneracy and
    ancestral allele state. This is done by providing a list of annotations to the parser which are
    applied in the order they are provided.

    The parser also allows to filter sites based on site properties which is done by
    passing a list of filtrations. By default, we filter out poly-allelic sites as sites are assumed to be
    at most bi-allelic.

    In addition, the parser allows to stratify the SFS by providing a list of stratifications. This is useful
    to obtain the SFS separately for different types of sites.

    To correctly determine the number of target sites when parsing an input that does not contain monomorphic sites,
    we can use a :class:`~sfsutils.parser.TargetSiteCounter`. This class is used in conjunction with the parser and
    samples sites from the given FASTA file that are found in between variants on the same contig that were parsed
    from the input.

    Note that we assume the sites in the input to be sorted by position in ascending order (per contig).

    Example usage:

    ::

        import sfsutils as su

        # Parse selected and neutral SFS from human chromosome 1.
        p = su.Parser(
            source="https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/"
                "hgdp_wgs.20190516.full.chr21.vcf.gz",
            fasta="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
                  "dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz",
            gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
            aliases=dict(chr21=['21']),  # mapping for contig names
            n=10,  # SFS sample size
            # we use a target site counter to infer the number of target sites.
            target_site_counter=su.TargetSiteCounter(
                n_samples=1000000,
                # determine number of target sites by looking at total length of coding sequences
                n_target_sites=su.Annotation.count_target_sites(
                    "http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                    "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz"
                )['21']
            ),
            # add degeneracy annotation for sites
            annotations=[
                su.DegeneracyAnnotation()
            ],
            filtrations=[
                # exclude non-SNPs as we infer monomorphic sites with target site counter
                su.SNPFiltration(),
                # filter out sites not in coding sequences
                su.CodingSequenceFiltration()
            ],
            # stratify by 4-fold/0-fold degeneracy
            stratifications=[su.DegeneracyStratification()],
            info_ancestral='AA_ensembl'
        )

        sfs = p.parse()

        sfs.plot()

    """

    def __init__(
            self,
            source: "str | os.PathLike | 'tskit.TreeSequence' | VariantReader | None" = None,
            n: int | Dict[str, int] | List[int] = None,
            pops: Dict[str, List[str]] | None = None,
            gff: str | None = None,
            fasta: str | None = None,
            info_ancestral: str = 'AA',
            info_ancestral_prob: str = 'AA_prob',
            skip_non_polarized: bool = True,
            stratifications: List[Stratification] = [],
            annotations: List[Annotation] = [],
            filtrations: List[Filtration] = None,
            include_samples: List[str] = None,
            exclude_samples: List[str] = None,
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {},
            target_site_counter: TargetSiteCounter = None,
            subsample_mode: Literal['random', 'probabilistic'] = 'probabilistic',
            polarize_probabilistically: bool = False,
            two_sfs: bool = False,
            d: int = 1000,
            two_sfs_offset: int = 0,
            vcf: "str | os.PathLike | 'tskit.TreeSequence' | VariantReader | None" = None
    ):
        """
        Initialize the parser.

        :param source: The variant source: a path to a VCF file (gzipped or a URL), a path to a VCF-Zarr store
            (a ``.vcz`` or ``.zarr`` directory), a tskit tree sequence (a ``.trees`` file or an in-memory
            ``tskit.TreeSequence``), or a pre-built :class:`~sfsutils.io_handlers.VariantReader`. A non-path,
            non-tree-sequence source must be a ``VariantReader`` (it exposes the sample names, contig names and
            site count the parser needs, and supports a fresh iteration pass); a bare iterable of sites or a
            raw ``cyvcf2.VCF`` object is not accepted and raises a ``TypeError``. VCF-Zarr requires the optional
            ``zarr`` package and tree sequences the optional ``tskit`` package.
        :param gff: The path to the GFF file, possibly gzipped or a URL. This file is optional and depends on
            the stratifications, annotations and filtrations that are used.
        :param fasta: The path to the FASTA file, possibly gzipped or a URL. This file is optional and depends on
            the annotations and filtrations that are used.
        :param n: The size of the resulting SFS. We down-sample to this number by drawing without replacement from
            the set of all available genotypes per site. Sites with fewer than ``n`` genotypes are skipped. For a
            joint (multi-population) SFS (see ``pops``), this is the per-population sample size, given either as a
            single ``int`` applied to every population, a list aligned with ``pops`` (in insertion order), or a
            dictionary keyed by population name.
        :param pops: Mapping of population name to the list of sample names making up that population. When given,
            :meth:`parse` returns a :class:`~sfsutils.spectrum.JointSpectra` holding the joint SFS across these
            populations (one :class:`~sfsutils.spectrum.JointSFS` per stratification type) instead of the
            one-dimensional :class:`~sfsutils.spectrum.Spectra`. The joint SFS is obtained by down-projecting each
            population independently and accumulating the outer product of the per-population projections, which is
            the exact hypergeometric down-projection under independent sampling within populations. When ``None``
            (default), a single-population, one-dimensional SFS is parsed as before. Note that ``pops`` supersedes
            ``include_samples`` / ``exclude_samples``. A ``target_site_counter`` is supported: monomorphic sites map
            to the all-ancestral origin of the joint SFS, which is scaled to the target-site count.
        :param info_ancestral: The tag in the INFO field that contains ancestral allele information. Consider using
            an ancestral allele annotation if this information is not available yet.
        :param skip_non_polarized: Whether to skip poly-morphic sites that are not polarized, i.e., without a valid
            info tag providing the ancestral allele. If ``False``, we use the reference allele as ancestral allele
            (only recommended if working with folded spectra).
        :param stratifications: List of stratifications to use.
        :param annotations: List of annotations to use.
        :param filtrations: List of filtrations to use. By default, we use
            :class:`~sfsutils.filtration.PolyAllelicFiltration`.
        :param include_samples: List of sample names to consider when determining the SFS. If ``None``, all samples
            are used. Note that this restriction does not apply to the annotations and filtrations.
        :param exclude_samples: List of sample names to exclude when determining the SFS. If ``None``, no samples
            are excluded. Note that this restriction does not apply to the annotations and filtrations.
        :param max_sites: Maximum number of sites to parse from the input.
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files downloaded from URLs.
        :param aliases: Dictionary of aliases for the contigs in the input, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the input with the contig names in the FASTA file and GFF file.
        :param target_site_counter: The target site counter, used to recover the number of monomorphic sites when the
            input contains only polymorphic sites. If ``None``, no target sites are added. It applies to the
            one-dimensional SFS (sampling monomorphic sites from the FASTA), the joint SFS (scaling the all-ancestral
            corner to the target-site count), and the unstratified two-SFS (extrapolating the monomorphic-involving
            pairs from the target-site count, which makes the resulting cov/corr only approximate).
        :param subsample_mode: The subsampling mode. For ``random``, we draw once without replacement from the set of
            all available genotypes per site. For ``probabilistic``, we add up the hypergeometric distribution for all
            sites. This will produce a smoother SFS, especially when a small number of sites is considered.
        :param polarize_probabilistically: Whether to probabilistically polarize sites. In addition to the ``AA`` tag
            (see ``info_ancestral``), we use the ``AA_prob`` tag (see ``info_ancestral_prob``) to polarize sites
            probabilistically. For example, if the ancestral allele is ``A`` with a probability of 0.8 and
            the derived allele is ``G``, we assign 0.8 probability mass to the ancestral allele and 0.2 to the
            derived allele. This should enhance accuracy, especially for small datasets. Whenever the ancestral
            probability tag is not present, we assume a probability of 1 for the ancestral allele.
        :param two_sfs: Whether to parse the two-dimensional (two-site) SFS instead of the ordinary SFS. When
            ``True``, :meth:`parse` returns a square :class:`~sfsutils.spectrum.TwoSFS` whose entry ``(i, j)`` counts
            pairs of sites, on the same contig and within the distance window of one another, where one site has ``i``
            and the other ``j`` derived alleles (down-projected to ``n``). As for the one-dimensional SFS, monomorphic
            sites are retained (contributing to the zero-frequency row and column); add a
            :class:`~sfsutils.filtration.SNPFiltration` to restrict the spectrum to segregating sites. Each site's
            contribution is the outer product of the two per-site down-projection vectors, so it is exact when
            ``subsample_mode='random'`` at the full sample size and smoother under ``'probabilistic'``. The matrix is
            symmetrized. Not compatible with ``pops``. Stratifications are supported (counting only within-stratum
            pairs, into a :class:`~sfsutils.spectrum.TwoSpectra`). The unstratified two-SFS accepts a
            ``target_site_counter``, which extrapolates the monomorphic-involving pairs from the target-site count;
            the resulting :meth:`~sfsutils.spectrum.TwoSFS.cov` / :meth:`~sfsutils.spectrum.TwoSFS.corr` are then only
            approximate, so a real all-sites input (monomorphic sites retained) is preferred for an accurate
            covariance, while :meth:`~sfsutils.spectrum.TwoSFS.fpmi` needs no monomorphic sites.
        :param d: The width (in base pairs) of the distance window over which the two sites of a pair
            are separated when ``two_sfs=True``. Together with ``two_sfs_offset`` it defines the window
            ``(two_sfs_offset, two_sfs_offset + d]``; with the default ``two_sfs_offset=0`` this is
            simply the maximum separation. Restricting the window restricts the spectrum to (approximately) linked
            pairs.
        :param two_sfs_offset: The minimum genomic distance (in base pairs, exclusive) between the two sites of a
            pair when ``two_sfs=True``; pairs are formed for separations in ``(two_sfs_offset, two_sfs_offset +
            d]``. Defaults to ``0`` (pairs at any separation up to ``d``).
        :param vcf: Deprecated alias for ``source``, kept for backward compatibility. Provide either
            ``source`` or ``vcf``, not both.
        """
        MultiHandler.__init__(
            self,
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

        # warn if SynonymyAnnotation is used
        if any(isinstance(a, SynonymyAnnotation) for a in annotations):
            logger.warning("SynonymyAnnotation is not recommended to be used with the parser as "
                           "it is not possible to determine the synonymy of monomorphic sites. "
                           "Consider using DegeneracyAnnotation instead.")

        #: The target site counter
        self.target_site_counter: TargetSiteCounter | None = target_site_counter

        #: Mapping of population name to sample names for the joint SFS, or ``None`` for a single-population SFS
        self.pops: Dict[str, List[str]] | None = pops

        if self.pops is None:
            #: The per-population sample size (single population)
            self.n: int = int(n)

            #: Ordered population names (single ``all`` population)
            self._pop_names: List[str] = []

            #: Per-population sample sizes
            self._n_per_pop: List[int] = [self.n]
        else:
            if len(self.pops) < 1:
                raise ValueError("At least one population must be provided in 'pops'.")

            self._pop_names = list(self.pops.keys())

            # normalize n to a per-population list aligned with the population order
            if isinstance(n, dict):
                missing = set(self._pop_names) - set(n.keys())
                if missing:
                    raise ValueError(f"Missing sample size in 'n' for populations: {sorted(missing)}.")
                self._n_per_pop = [int(n[name]) for name in self._pop_names]
            elif isinstance(n, (list, tuple, np.ndarray)):
                if len(n) != len(self._pop_names):
                    raise ValueError(f"Length of 'n' ({len(n)}) must match the number of populations "
                                     f"({len(self._pop_names)}).")
                self._n_per_pop = [int(x) for x in n]
            else:
                self._n_per_pop = [int(n)] * len(self._pop_names)

            # the single-population sample size is undefined in joint mode
            self.n: int | None = None

            if include_samples is not None or exclude_samples is not None:
                logger.warning("'include_samples' and 'exclude_samples' are ignored when 'pops' is given.")

        #: The shape of the joint SFS array (one axis per population)
        self._joint_shape: Tuple[int, ...] = tuple(x + 1 for x in self._n_per_pop)

        #: Per-population sample masks (joint mode), set up in :meth:`_prepare_samples_mask`
        self._pop_masks: List[np.ndarray] | None = None

        #: The list of samples to include
        self.include_samples: List[str] | None = include_samples

        #: The list of samples to exclude
        self.exclude_samples: List[str] | None = exclude_samples

        #: The mask of samples to use
        self._samples_mask: np.ndarray | None = None

        #: Whether to skip sites that are not polarized, i.e., without a valid info tag providing the ancestral allele
        self.skip_non_polarized: bool = skip_non_polarized

        #: List of stratifications to use
        self.stratifications: List[Stratification] = stratifications

        #: List of annotations to use
        self.annotations: List[Annotation] = annotations

        #: List of filtrations to use
        self.filtrations: List[Filtration] = [PolyAllelicFiltration()] if filtrations is None else filtrations

        #: The number of sites that were skipped for various reasons
        self.n_skipped: int = 0

        # The number of sites with a valid ancestral allele probability
        self.n_aa_prob: int = 0

        #: The number of sites that were skipped because they had no valid ancestral allele
        self.n_no_ancestral: int = 0

        #: Dictionary of SFS indexed by (stratification) type. Each value is a 1-D array of length ``n + 1`` for a
        #: single-population SFS, or a joint SFS array of shape :attr:`_joint_shape` when ``pops`` is given.
        self.sfs: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(self._joint_shape))

        #: 1-based positions of lowest and highest site position per contig (only when target_site_counter is used)
        # noinspection PyTypeChecker
        self._contig_bounds: Dict[str, Tuple[int, int]] = defaultdict(lambda: (np.inf, -np.inf))

        if subsample_mode not in ['random', 'probabilistic']:
            raise ValueError(f"Subsampling mode '{subsample_mode}' is not valid. "
                             f"Valid modes are 'random' and 'probabilistic'.")

        #: The subsampling mode
        self.subsample_mode: Literal['random', 'probabilistic'] = subsample_mode

        #: The tag in the INFO field that contains the ancestral allele probability
        self.info_ancestral_prob: str = info_ancestral_prob

        #: Whether to probabilistically polarize sites
        self.polarize_probabilistically: bool = polarize_probabilistically

        #: Whether to parse the two-dimensional (two-site) SFS
        self.two_sfs: bool = two_sfs

        #: Maximum genomic distance (bp) between the two sites of a pair for the two-SFS
        self.d: int = int(d)

        #: Minimum genomic distance (bp, exclusive) between the two sites of a pair for the two-SFS
        self.two_sfs_offset: int = int(two_sfs_offset)

        if self.two_sfs:
            if self.pops is not None:
                raise NotImplementedError("The two-SFS (two_sfs=True) is not supported together with 'pops'.")

            if self.d < 1:
                raise ValueError("d must be at least 1.")

        #: The accumulating two-SFS matrices of shape ``(n + 1, n + 1)`` keyed by stratification type. Stratified
        #: parsing counts only within-stratum pairs, so each type accumulates independently (only when ``two_sfs``).
        self._two_sfs_matrices: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros((self.n + 1, self.n + 1)))

        #: The accumulating one-dimensional marginal (summed down-projection) per type, i.e. the site-frequency
        #: spectrum including the monomorphic bins; used to warn when the monomorphic sites appear to be missing.
        self._two_sfs_marginal: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(self.n + 1))

        #: Sliding buffer of ``(position, down-projection vector, type)`` for recent sites on the current contig
        self._two_sfs_buffer: deque = deque()

        #: The contig currently held in the two-SFS buffer
        self._two_sfs_contig: str | None = None

    def _get_ancestral(self, variant: Site) -> str:
        """
        Determine the ancestral allele.

        :param variant: The site
        :return: Ancestral allele
        :raises ~sfsutils.io_handlers.NoTypeException: If the site is not polarized and ``skip_non_polarized`` is ``True`` or if
            the ancestral allele or reference allele (in case of monomorphic sites) is not a valid base.
        """
        if variant.is_snp:
            # obtain ancestral allele
            aa = variant.INFO.get(self.info_ancestral)

            # return the ancestral allele if it is a valid base
            if aa in bases:
                return aa

            # if we skip non-polarized sites, we raise an error here
            if self.skip_non_polarized:
                raise NoTypeException("No valid AA tag found")

        # if we don't skip non-polarized sites, or if the site is not an SNP
        # we return the reference allele if valid
        if variant.REF in bases:
            return variant.REF

        # if the reference allele is not a valid base, we raise an error
        raise NoTypeException("Reference allele is not a valid base")

    def _get_ancestral_prob(self, variant: Site) -> float:
        """
        Determine the ancestral allele probabilistically.

        :param variant: The site
        :return: The probability of the ancestral allele being the true ancestral allele
        """
        if variant.is_snp and self.polarize_probabilistically:
            raw = variant.INFO.get(self.info_ancestral_prob)

            # INFO comes through typed from cyvcf2 but as a plain string from the VCF-Zarr backend, so
            # cast explicitly; a missing value (None) or an empty / '.' sentinel means the site is not
            # probabilistically polarized and is treated as certain (probability 1)
            if raw is not None and raw not in ('', '.'):
                self.n_aa_prob += 1
                return float(raw)

        return 1.0

    def _parse_site(self, variant: Site) -> bool:
        """
        Parse a single site, adding its down-projected mass to the (possibly joint) SFS of its type.

        :param variant: The variant.
        :return: Whether the site was included in the SFS.
        """
        # the two-SFS pairs each site with nearby sites rather than accumulating a per-type spectrum
        if self.two_sfs:
            return self._parse_site_two_sfs(variant)

        # compute the down-projected mass for this site (1-D or joint depending on whether populations are given)
        m = self._project_joint(variant) if self.pops is not None else self._project(variant)

        if m is None:
            return False

        # try to obtain type
        try:
            # create joint type
            t = '.'.join([s.get_type(variant) for s in self.stratifications]) or 'all'

            # add mass
            self.sfs[t] += m

        except NoTypeException as e:
            self._logger.debug(e)
            return False

        return True

    def _project(self, variant: Site) -> Optional[np.ndarray]:
        """
        Down-project a single site to the one-dimensional SFS, returning the mass over derived-allele counts.

        :param variant: The variant.
        :return: The mass array of length ``n + 1``, or ``None`` if the site is skipped.
        """
        # check `is_snp` property for performance reasons but site may still be monomorphic
        if variant.is_snp:

            # obtain called bases
            genotypes = get_called_bases(variant.gt_bases[self._samples_mask])

            # number of samples
            n_samples = len(genotypes)

            # skip if not enough samples
            if n_samples < self.n:
                self._logger.debug(f'Skipping site due to too few samples at {variant.CHROM}:{variant.POS}.')
                return None

            try:
                # determine ancestral allele
                aa = self._get_ancestral(variant)
            except NoTypeException:
                self.n_no_ancestral += 1
                return None

            # determine ancestral allele probability
            aa_prob = self._get_ancestral_prob(variant)

            # count called bases
            counter = Counter(genotypes)

            # determine ancestral allele count
            n_aa = counter[aa]

            # count the ancestral allele among the site's alleles: if the AA tag names a base that is
            # absent from the observed genotypes the site is effectively multi-allelic (ancestral plus two
            # derived) and must be skipped, not silently polarised into the all-derived bin
            n_alleles = len(counter) + (0 if aa in counter else 1)
            if n_alleles > 2:
                self._logger.debug(f'Site has more than two alleles at {variant.CHROM}:{variant.POS} ({dict(counter)}, AA={aa})')
                return None

            # determine down-projected allele count.
            if self.subsample_mode == 'random':
                m = np.zeros(self.n + 1)
                k = hypergeom.rvs(M=n_samples, n=n_samples - n_aa, N=self.n, random_state=self.rng)

                # polarize probabilistically only for bi-allelic sites, as the probabilistic branch and
                # both joint branches do; otherwise the two subsample modes disagree on the same input
                if len(counter) == 2:
                    m[k] += aa_prob
                    m[self.n - k] += 1 - aa_prob
                else:
                    m[k] += 1.0
            else:
                # subsample probabilistically drawing from the hypergeometric distribution (without replacement)
                m = hypergeom.pmf(k=range(self.n + 1), M=n_samples, n=n_samples - n_aa, N=self.n)

                # polarize probabilistically if site is bi-allelic
                if len(counter) == 2:
                    m = aa_prob * m + (1 - aa_prob) * m[::-1]

        # if we have a mono-allelic SNPs
        elif is_monomorphic_snp(variant):
            # apply the same coverage requirement as segregating sites, so a low-coverage monomorphic
            # site is not asymmetrically retained and does not inflate the monomorphic:polymorphic ratio
            # (TargetSiteCounter's DummyVariant sites are fully covered by construction and always pass)
            if len(get_called_bases(variant.gt_bases[self._samples_mask])) < self.n:
                self._logger.debug(f'Skipping monomorphic site due to too few samples at {variant.CHROM}:{variant.POS}.')
                return None

            # the reference allele is assumed ancestral, so the derived-allele count is 0 (the
            # polarization of monomorphic sites does not matter)
            m = np.zeros(self.n + 1)
            m[0] = 1
        else:
            # skip other types of sites
            self._logger.debug(f'Site is not a valid single nucleotide site at {variant.CHROM}:{variant.POS}.')
            return None

        return m

    def _project_joint(self, variant: Site) -> Optional[np.ndarray]:
        """
        Down-project a single site to the joint (multi-population) SFS.

        Each population is down-projected independently to a distribution over its derived-allele count, and the
        site's contribution is the outer product of these per-population distributions. This is the exact
        hypergeometric down-projection under sampling without replacement independently within each population.

        :param variant: The variant.
        :return: The joint mass array of shape :attr:`_joint_shape`, or ``None`` if the site is skipped.
        """
        if variant.is_snp:

            # obtain called bases per population
            pop_bases = [get_called_bases(variant.gt_bases[mask]) for mask in self._pop_masks]

            # skip if any population has too few called genotypes
            if any(len(b) < n for b, n in zip(pop_bases, self._n_per_pop)):
                self._logger.debug(f'Skipping site due to too few samples at {variant.CHROM}:{variant.POS}.')
                return None

            try:
                # determine ancestral allele (site-level, shared across populations)
                aa = self._get_ancestral(variant)
            except NoTypeException:
                self.n_no_ancestral += 1
                return None

            # determine ancestral allele probability
            aa_prob = self._get_ancestral_prob(variant)

            # biallelic check across all populations combined; count the ancestral allele too, so a site
            # whose AA is a third base absent from the genotypes is skipped rather than polarised into the
            # all-derived joint corner
            counter = Counter(np.concatenate(pop_bases))

            n_alleles = len(counter) + (0 if aa in counter else 1)
            if n_alleles > 2:
                self._logger.debug(f'Site has more than two alleles at {variant.CHROM}:{variant.POS} ({dict(counter)}, AA={aa})')
                return None

            # down-project each population independently to a distribution over its derived-allele count
            projections = []
            for b, n in zip(pop_bases, self._n_per_pop):
                n_samples = len(b)
                n_der = n_samples - int(np.sum(b == aa))

                if self.subsample_mode == 'random':
                    vec = np.zeros(n + 1)
                    vec[hypergeom.rvs(M=n_samples, n=n_der, N=n, random_state=self.rng)] = 1.0
                else:
                    vec = hypergeom.pmf(k=range(n + 1), M=n_samples, n=n_der, N=n)

                projections.append(vec)

            # the joint mass is the outer product of the per-population projections
            m = functools.reduce(np.multiply.outer, projections)

            # probabilistically polarize bi-allelic sites: a mispolarized site flips every population's
            # derived count simultaneously, i.e. reflects the joint array on all axes
            if len(counter) == 2 and aa_prob != 1.0:
                reflected = m[tuple(slice(None, None, -1) for _ in projections)]
                m = aa_prob * m + (1 - aa_prob) * reflected

        # if we have a mono-allelic SNP
        elif is_monomorphic_snp(variant):
            # apply the same per-population coverage requirement as segregating sites
            if any(len(get_called_bases(variant.gt_bases[mask])) < n for mask, n in zip(self._pop_masks, self._n_per_pop)):
                self._logger.debug(f'Skipping monomorphic site due to too few samples at {variant.CHROM}:{variant.POS}.')
                return None

            # all-ancestral: place mass at the origin (zero derived alleles in every population)
            m = np.zeros(self._joint_shape)
            m[(0,) * len(self._joint_shape)] = 1.0
        else:
            # skip other types of sites
            self._logger.debug(f'Site is not a valid single nucleotide site at {variant.CHROM}:{variant.POS}.')
            return None

        return m

    def _parse_site_two_sfs(self, variant: Site) -> bool:
        """
        Add a single site to the two-SFS by pairing it with the recently seen sites within the distance window.

        The site's down-projection vector is paired (via an outer product) with that of every buffered site on the
        same contig whose separation falls in ``(two_sfs_offset, two_sfs_offset + d]``. A sliding
        buffer keeps only the sites still within reach, so the pass is linear in the number of pairs. When
        stratifications are used, only pairs of sites of the same type are counted, into that type's matrix.

        :param variant: The variant.
        :return: Whether the site was included (i.e. had a valid down-projection and, if stratified, a type).
        """
        m = self._project(variant)

        if m is None:
            return False

        # determine the site's stratification type; a site without a type is skipped (as in the ordinary SFS)
        try:
            t = '.'.join([s.get_type(variant) for s in self.stratifications]) or 'all'
        except NoTypeException as e:
            self._logger.debug(e)
            return False

        # register the stratum so it appears in the output even if it never forms a within-window pair
        _ = self._two_sfs_matrices[t]

        # accumulate the one-dimensional marginal (the site-frequency spectrum, incl. the monomorphic bins)
        self._two_sfs_marginal[t] += m

        # reset the buffer when we move to a new contig (pairs never cross contigs)
        if variant.CHROM != self._two_sfs_contig:
            self._two_sfs_buffer.clear()
            self._two_sfs_contig = variant.CHROM

        # the furthest separation at which a pair is still formed
        max_distance = self.two_sfs_offset + self.d

        # drop buffered sites that are now too far behind to pair with any future site
        while self._two_sfs_buffer and variant.POS - self._two_sfs_buffer[0][0] > max_distance:
            self._two_sfs_buffer.popleft()

        # pair with each buffered site of the same type whose separation falls in the window
        for pos, m_prev, t_prev in self._two_sfs_buffer:
            distance = variant.POS - pos

            if self.two_sfs_offset < distance <= max_distance and t_prev == t:
                # accumulate the (forward) within-stratum pair; the matrix is symmetrized once at the end
                self._two_sfs_matrices[t] += np.multiply.outer(m_prev, m)

        self._two_sfs_buffer.append((variant.POS, m, t))

        return True

    def _warn_if_monomorphic_missing(self):
        """
        Warn when the parsed two-SFS appears to carry (almost) no monomorphic sites. The two-SFS covariance and
        correlation (:meth:`~sfsutils.spectrum.TwoSFS.cov` / :meth:`~sfsutils.spectrum.TwoSFS.corr`) require the
        monomorphic sites to anchor the marginal to the site-frequency spectrum, so an all-sites input is needed.
        """
        marginal = sum(self._two_sfs_marginal.values(), np.zeros(self.n + 1))
        n_sites = float(marginal.sum())
        n_monomorphic = float(marginal[0] + marginal[-1])

        if n_sites > 0 and n_monomorphic / n_sites < 0.95:
            self._logger.warning(
                "The number of monomorphic sites is unusually low. Including monomorphic sites is necessary for a "
                "meaningful two-SFS covariance/correlation (they anchor the marginal to the site-frequency "
                "spectrum). If your dataset does not contain monomorphic sites, provide an all-sites input."
            )

    def _region_length(self) -> float:
        """
        The length (bp) of the region over which sites are distributed, used when a :class:`TargetSiteCounter`
        extrapolates the monomorphic pairs of the two-SFS. When the source reports its own length (a tree sequence
        does) that is used, since the observed polymorphic sites never reach the ends and their span underestimates
        the region; otherwise the summed per-contig span of the parsed variants is used.

        :return: The region length.
        """
        sequence_length = getattr(self._reader, 'sequence_length', None)
        if sequence_length is not None:
            return float(sequence_length)

        return float(sum(
            high - low for low, high in self._contig_bounds.values()
            if np.isfinite(low) and np.isfinite(high) and high > low
        ))

    def _process_site(self, variant: Site) -> bool:
        """
        Handle a single variant.

        :param variant: The variant
        :return: Whether the site was included in the SFS.
        """
        # filter the variant
        for filtration in self.filtrations:
            if not filtration.filter_site(variant):
                return False

        # apply annotations
        for annotation in self.annotations:
            annotation.annotate_site(variant)

        # parse site
        return self._parse_site(variant)

    def _rewind(self):
        """
        Rewind the filtrations, annotations and stratifications, and fasta handler.
        """
        FASTAHandler._rewind(self)

        for f in self.filtrations:
            f._rewind()

        for a in self.annotations:
            a._rewind()

        for s in self.stratifications:
            s._rewind()

    def _setup(self):
        """
        Set up the parser.
        """
        # set up target site counter
        if self.target_site_counter is not None:
            self.target_site_counter._setup(self)

        # make parser available to stratifications
        for s in self.stratifications:
            s._setup(self)

        # create a string representation of the stratifications
        representation = '.'.join(['[' + ', '.join(s.get_types()) + ']' for s in self.stratifications]) or "[all]"

        # log the stratifications
        self._logger.info(f'Using stratification: {representation}.')

        # prepare samples mask
        self._prepare_samples_mask()

        # setup annotations
        for annotation in self.annotations:
            annotation._setup(self)

        # setup filtrations
        for f in self.filtrations:
            f._setup(self)

    def _prepare_samples_mask(self):
        """
        Prepare the samples mask, or per-population masks in joint mode.
        """
        samples = np.array(self._reader.samples)

        # in joint mode, build one boolean mask per population
        if self.pops is not None:
            self._pop_masks = []

            for name in self._pop_names:
                mask = np.isin(samples, self.pops[name])

                if not mask.any():
                    raise ValueError(f"None of the samples for population '{name}' were found in the input.")

                self._pop_masks.append(mask)

            return

        # determine samples to include
        if self.include_samples is None:
            mask = np.ones(len(samples)).astype(bool)
        else:
            mask = np.isin(samples, self.include_samples)

        # determine samples to exclude
        if self.exclude_samples is not None:
            mask &= ~np.isin(samples, self.exclude_samples)

        self._samples_mask = mask

    def _teardown(self):
        """
        Tear down parser components.
        """
        # tear down all objects
        for f in self.filtrations:
            f._teardown()

        for s in self.stratifications:
            s._teardown()

        for a in self.annotations:
            a._teardown()

    def parse(self) -> Spectra | JointSpectra | TwoSpectra:
        """
        Parse the site-frequency spectrum from the configured source (VCF, VCF-Zarr, or tree sequence).

        The return type is fixed by the parsing mode, and this mapping is a stable part of the API. Every mode
        returns a collection keyed by stratification type (a single ``'all'`` key when no stratifications were
        given), so the return type is consistent across modes:

        - one-dimensional SFS (the default): a :class:`~sfsutils.spectrum.Spectra` of one-dimensional spectra;
        - multi-population (``pops`` given): a :class:`~sfsutils.spectrum.JointSpectra` of joint spectra, keyed
          by (sorted) stratification type;
        - two-SFS (``two_sfs`` set, with or without stratifications): a
          :class:`~sfsutils.spectrum.TwoSpectra` of per-type two-SFS matrices. Without stratifications this holds
          the single ``'all'`` entry, so the two-SFS is reached as ``parse()['all']``.

        :return: A :class:`~sfsutils.spectrum.Spectra`, :class:`~sfsutils.spectrum.JointSpectra`, or
            :class:`~sfsutils.spectrum.TwoSpectra` as described above.
        """
        # set up parser
        self._setup()

        pbar = self.get_pbar(
            total=self.n_sites,
            desc=f"{self.__class__.__name__}>Processing sites"
        )

        # iterate over variants
        for i, variant in enumerate(self._reader):

            # handle site
            if self._process_site(variant):

                if self.target_site_counter is not None:
                    # update bounds
                    low, high = self._contig_bounds[variant.CHROM]
                    self._contig_bounds[variant.CHROM] = (min(low, variant.POS), max(high, variant.POS))
            else:
                self.n_skipped += 1

            # update progress bar
            pbar.update()

            # explicitly stopping after ``n`` sites fixes a bug with cyvcf2:
            # 'error parsing variant with `htslib::bcf_read` error-code: 0 and ret: -2'
            if i + 1 == self.n_sites or i + 1 == self.max_sites:
                break

        # close progress bar
        pbar.close()

        # tear down components
        self._teardown()

        # inform about number of sites without ancestral tag
        if self.n_no_ancestral > 0:
            self._logger.info(f'Skipped {self.n_no_ancestral} sites without ancestral allele information.')

        if len(self.sfs) == 0 and not self.two_sfs:
            self._logger.warning(f"No sites were included in the spectra. If this is not expected, "
                                 "please check that all components work as expected. You can also "
                                 "set the log level to DEBUG.")

            # warn that sites might not be polarized
            if self.skip_non_polarized and not any(isinstance(a, AncestralAlleleAnnotation) for a in self.annotations):
                self._logger.warning("Your variants might not be polarized and are thus not included in the spectra. "
                                     "If this is the case, consider annotating the ancestral states or setting "
                                     "'Parser.skip_non_polarized' to False.")
        else:
            n_included = self.n_sites - self.n_skipped

            self._logger.info(f'Included {n_included} out of {self.n_sites} sites in total from the input.')

            if self.polarize_probabilistically:
                self._logger.info(f'Considered {self.n_aa_prob} sites with valid ancestral allele probability.')

        # close VCF reader
        VCFHandler._rewind(self)

        # in two-SFS mode, return the symmetrized pair-count matrix (or a collection thereof when stratified)
        if self.two_sfs:
            total = sum(matrix.sum() for matrix in self._two_sfs_matrices.values())
            self._logger.info(f'Counted {total:.0f} site pairs within {self.d} bp.')

            if self.target_site_counter is None:
                self._warn_if_monomorphic_missing()

            # without stratifications, return a single-entry collection keyed 'all' (consistent with the other
            # parsing modes); a TargetSiteCounter extrapolates the monomorphic-involving pairs from the
            # target-site count (approximate; fpmi() ignores them)
            if len(self.stratifications) == 0:
                matrix = TwoSFS(self._two_sfs_matrices['all']).symmetrize().data
                if self.target_site_counter is not None:
                    matrix = self.target_site_counter._extrapolate_two_sfs(
                        matrix, self._two_sfs_marginal['all'], self._region_length(), self.d
                    )
                return TwoSpectra({'all': TwoSFS(matrix)})

            if self.target_site_counter is not None:
                raise NotImplementedError(
                    "A TargetSiteCounter with the stratified two-SFS is not supported. Parse an all-sites input, or "
                    "use the unstratified two-SFS with a TargetSiteCounter."
                )

            # one symmetrized two-SFS per (within-stratum) type
            return TwoSpectra(
                {t: TwoSFS(self._two_sfs_matrices[t]).symmetrize() for t in sorted(self._two_sfs_matrices)}
            )

        # in joint mode, return a collection of joint spectra keyed by (sorted) type
        if self.pops is not None:
            # extrapolate the monomorphic corner from the target-site count
            if self.target_site_counter is not None and self.n_skipped < self.n_sites:
                self.target_site_counter.count()
                self.sfs = self.target_site_counter._update_target_sites_joint(dict(self.sfs))

            return JointSpectra(
                {t: JointSFS(self.sfs[t], self._pop_names) for t in sorted(self.sfs)}
            )

        # count target sites
        if self.target_site_counter is not None and self.n_skipped < self.n_sites:
            # count target sites
            self.target_site_counter.count()

            # update target sites
            self.sfs = self.target_site_counter._update_target_sites(Spectra(dict(self.sfs))).to_dict()

        return Spectra(dict(self.sfs)).sort_types()
