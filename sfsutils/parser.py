"""
A parser that extracts the site frequency spectrum (SFS) from a VCF file, a VCF-Zarr store, or a tskit
tree sequence (ARG). Stratifying the SFS is supported by providing a list of :class:`Stratification`
instances.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-26"

import bisect
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
    DummyVariant, Site, MultiHandler, VCFHandler, VariantReader, is_monomorphic_snp, SiteAlleles
from .settings import Settings
from .spectrum import Spectra, TwoSFS, TwoSpectra, JointSFS, JointSpectra

# logger
logger = logging.getLogger('sfsutils')

#: Number of sites between full rebuilds of the two-SFS running window sums from the sites in the window. The
#: sums are maintained by adding a site's vector when it enters the window and subtracting it when it leaves,
#: which accumulates rounding error over a long contig; rebuilding them at this interval bounds that drift while
#: costing one pass over the window per few thousand sites.
_TWO_SFS_RESYNC_INTERVAL = 4096


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

            # fetch contig; a contig missing from the FASTA makes the site untypeable rather than being fatal,
            # as every other consumer of get_contig already treats it
            try:
                self.contig = self.get_contig(aliases)
            except LookupError as e:
                raise NoTypeException(str(e))

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
        if self.contigs is not None:
            # match through the aliases, as ContigFiltration does, so an aliased contig is not dropped
            # match through the aliases where a parser is attached, so an aliased contig is not dropped
            aliases = self.parser.get_aliases(variant.CHROM) if self.parser is not None else [variant.CHROM]
            matched = next((c for c in self.contigs if c in aliases), None)

            if matched is None:
                raise NoTypeException(f"Contig '{variant.CHROM}' not in list of contigs.")

            return self._sanitize(matched)

        return self._sanitize(variant.CHROM)

    @staticmethod
    def _sanitize(contig: str) -> str:
        """
        Replace the dots of an accession-style contig name, which Spectra uses to separate stratification
        levels and would otherwise split such a name into several levels.

        :param contig: The contig name.
        :return: The type name.
        """
        return contig.replace('.', '_')

    def get_types(self) -> List[str]:
        """
        Get all possible contig type.

        :return: List of contexts
        """
        return [self._sanitize(c) for c in (self.contigs or list(self.parser._reader.seqnames))]


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
        is logged at setup when the parser carries filtrations.

    The first pass assigns sites by counting them and records the genomic position at which each chunk
    begins. Any further pass over the same input (the :class:`TargetSiteCounter` sampling pass, which visits a
    different number of sites) is assigned by position instead, so a sampled site lands in the same chunk as the
    variants surrounding it.
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

        #: Cumulative chunk sizes, i.e. the site counter at which each chunk ends
        self._chunk_ends: List[int] = []

        #: Number of sites seen so far
        self.counter: int = 0

        #: Index of each contig in the order in which the first pass encountered it
        self._contig_order: Dict[str, int] = {}

        #: Sort key (contig index, position) of the first site of each chunk, as recorded by the first pass
        self._chunk_starts: List[Tuple[int, int]] = []

        #: Whether sites are assigned by position rather than by counting them
        self._positional: bool = False

    def _setup(self, parser: 'Parser'):
        """
        Set up the stratification.

        :param parser: The parser
        """
        super()._setup(parser)

        # a parse derives the chunk boundaries from the input it is about to read, so any boundaries left
        # over from an earlier parse of a different input or site cap are discarded
        self._contig_order = {}
        self._chunk_starts = []
        self._positional = False

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

        # the site counter is located among the chunk boundaries once per site, so the boundaries are
        # accumulated here instead of being summed anew inside that lookup
        self._chunk_ends = list(itertools.accumulate(self.chunk_sizes))

    def _rewind(self):
        """
        Rewind the stratification, also resetting the per-pass site counter so a second pass
        (for example the :class:`TargetSiteCounter` sampling pass) restarts from the first chunk.
        """
        super()._rewind()
        self.counter = 0

    def _teardown(self):
        """
        Tear down the stratification, switching subsequent passes to positional assignment.
        """
        super()._teardown()

        # the boundaries recorded by the pass that just finished span the input, so any further pass can be
        # assigned by position, which keeps its sites in the chunk their genomic neighbourhood belongs to
        self._positional = len(self._chunk_starts) > 0

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
        # sites are ordered by contig of first appearance and then by position, as the input is read
        key = (self._contig_order.setdefault(variant.CHROM, len(self._contig_order)), variant.POS)

        if self._positional:
            # sites preceding the first recorded boundary, and contigs the first pass never saw, fall into
            # the first and last chunk respectively
            return f'chunk{max(bisect.bisect_right(self._chunk_starts, key) - 1, 0)}'

        # the boundaries are held in step with the chunk sizes here as well, so that they are also correct
        # when the sizes are assigned directly rather than by _setup
        if len(self._chunk_ends) != len(self.chunk_sizes):
            self._chunk_ends = list(itertools.accumulate(self.chunk_sizes))

        # find the index of the chunk to which the current site belongs; a pass may process more sites than
        # the chunk sizes account for (e.g. when a previous pass saw fewer), so fall back to the last chunk
        chunk_index = min(bisect.bisect_right(self._chunk_ends, self.counter), self.n_chunks - 1)

        # record where each chunk begins; chunks of size zero share the start of the chunk that follows them,
        # and the bisect above then resolves such a tie to the last of them, which is the non-empty one
        while len(self._chunk_starts) <= chunk_index:
            self._chunk_starts.append(key)

        # update the counter
        self.counter += 1

        return f'chunk{chunk_index}'


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

    def _rewind(self):
        """
        Rewind the stratification, re-seeding the random generator so that every pass draws the same bins.
        Without this a second parse, or the sampling pass of a :class:`TargetSiteCounter`, would continue the
        stream of the previous pass and assign the same sites to different bins.
        """
        super()._rewind()

        self.rng = random.Random(self.seed)

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

        #: The parser's filtrations while the SNP filtration is suspended
        self._filtrations: List[Filtration] | None = None

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
        if self._filtrations is not None:
            self.parser.filtrations = self._filtrations
            self._filtrations = None

    def count(self):
        """
        Count the number of target sites.

        :return: The number of target sites
        """
        # rewind parser components
        self.parser._rewind()

        # suspend SNP filtration
        self._suspend_snp_filtration()

        # initialize progress bar
        pbar = tqdm(
            total=self.n_samples,
            desc=f'{self.__class__.__name__}>Sampling target sites',
            disable=Settings.disable_pbar
        )

        try:
            i = self._sample(pbar)
        finally:
            # the parser is handed back to the caller whatever happens here, so its filtrations are restored
            # even when the sampling pass raises, which would otherwise leave it stripped of its SNP filtration
            pbar.close()
            self._resume_snp_filtration()

        # tear down
        self._teardown()

        # notify on number of sites included in the SFS
        self._logger.info(f"{i} out of {self.n_samples} sampled sites were valid.")

    def _sample(self, pbar: tqdm) -> int:
        """
        Draw monomorphic sites from the FASTA file across the intervals spanned by the parsed variants and
        feed them to the parser, which adds them to the SFS of their type.

        :param pbar: The progress bar to advance once per sampled site.
        :return: The number of sampled sites that were included in the SFS.
        """
        # rewind fasta iterator
        FASTAHandler._rewind(self.parser)

        # initialize random number generator
        rng = np.random.default_rng(self.parser.seed)

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

                # the bounds come from the parsed variants, which may reach beyond the end of the FASTA record
                # if the two files disagree on the contig; sample only from the part backed by the reference
                upper = min(int(bounds[1]), len(record.seq))

                if upper < bounds[1]:
                    self._logger.warning(
                        f"The FASTA record for contig '{contig}' is {len(record.seq)} bp long but the parsed "
                        f"variants reach position {int(bounds[1])}. Sampling target sites only up to "
                        f"{len(record.seq)}."
                    )

                # nothing of the contig's variant span is backed by the reference, so it yields no sites
                if upper <= bounds[0]:
                    self._logger.warning(f"Skipping contig '{contig}' when sampling target sites: its FASTA "
                                         f"record is {len(record.seq)} bp long, which does not reach the first "
                                         f"parsed variant at position {int(bounds[0])}.")
                    continue

                # get positions
                # we sort in ascending order as the parser expects the positions to be sorted
                positions = np.sort(rng.integers(bounds[0], upper, size=n))

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

        return i

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

        # the sampling pass only ever contributes all-ancestral sites, so bin 0 is the only bin it touches.
        # Derive the sampled counts without mutating the spectra, so the bail-outs below can return the input
        # untouched rather than a spectrum the subtraction has already mutilated.
        sampled_monomorphic = spectra.data.iloc[0, :] - before.iloc[0, :]

        # get number of monomorphic and polymorphic sites sampled from the FASTA and VCF file
        n_monomorphic = sampled_monomorphic.sum()
        n_polymorphic = spectra.data.iloc[1:-1, :].sum().sum()

        # every site seen in the input consumes target-site budget, the fixed-derived (divergence) ones included
        n_observed = before.iloc[1:, :].sum()

        # check if we have enough target sites
        if self.n_target_sites < n_polymorphic:
            self._logger.warning(f"Number of polymorphic sites ({n_polymorphic}) exceeds the total "
                                 f"number of target sites ({self.n_target_sites}) which does not make sense. "
                                 f"The number of target sites unchanged is left unchanged.")
            spectra.data = before.astype(float)
        elif n_monomorphic == 0:
            self._logger.warning(f"Number of monomorphic sites is zero which should only happen "
                                 f"if there are very few sites considered. Failed to adjust "
                                 f"the number of monomorphic sites.")
            spectra.data = before.astype(float)
        else:

            # compute multiplicative factor to scale the total number of sites
            # to the number of target sites plus the number of polymorphic sites
            x = (self.n_target_sites + before_n_polymorphic.sum() - n_polymorphic) / n_monomorphic

            # extrapolate the monomorphic counts and subtract the observed sites from them, so that the total
            # number of sites per type is the type's share of the target sites. We do this to correct for the
            # fact that, for a type, we have relatively fewer monomorphic sites if we have more polymorphic ones
            spectra.data.iloc[0, :] = sampled_monomorphic * x - n_observed

            # a type whose observed sites outnumber its share of the target sites would otherwise be assigned a
            # negative mutational opportunity, which is meaningless downstream
            negative = spectra.data.columns[spectra.data.iloc[0, :] < 0].tolist()
            if negative:
                self._logger.warning(f"The number of target sites is too small to accommodate the observed sites "
                                     f"of type(s) {negative}; their monomorphic counts were clipped to zero.")
                spectra.data.iloc[0, :] = spectra.data.iloc[0, :].clip(lower=0)

        return spectra

    def _update_target_sites_joint(self, sfs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extrapolate the monomorphic corner of a joint SFS. All monomorphic (all-ancestral) sites map to the single
        origin ``(0, ..., 0)`` of the joint spectrum, so the number of target sites fixes the mass placed there.
        The sites sampled from the FASTA file are drawn without regard to polymorphism, so their per-type shares
        estimate the composition of all sites: each type is scaled to its share of ``n_target_sites`` in total, and
        the sites it was observed at in the input are subtracted from its monomorphic mass, exactly as in the
        one-dimensional :meth:`_update_target_sites`.

        :param sfs: The per-type joint SFS after sampling (its origin holds the sampled monomorphic mass).
        :return: The per-type joint SFS with the origin set to the extrapolated monomorphic count.
        """
        shape = self.parser._joint_shape
        origin = (0,) * len(shape)
        corner = tuple(n - 1 for n in shape)

        def _before(t: str) -> np.ndarray:
            # the type's polymorphic joint SFS from before sampling (zeros if the type only appears from sampling)
            return np.array(self._sfs_polymorphic.get(t, np.zeros(shape)), dtype=float)

        before = {t: _before(t) for t in sfs}

        # every site seen in the input consumes target-site budget, the fixed-derived (divergence) ones included
        n_observed = {t: float(arr.sum() - arr[origin]) for t, arr in before.items()}

        # the divergence corner is monomorphic, so it is not part of the polymorphic mass
        n_polymorphic = sum(n_observed[t] - float(before[t][corner]) for t in before)

        # monomorphic mass sampled from the FASTA file per type (added at the origin during sampling)
        sampled = {t: float(sfs[t][origin] - before[t][origin]) for t in sfs}
        n_monomorphic = sum(sampled.values())

        if self.n_target_sites < n_polymorphic:
            self._logger.warning(f"Number of polymorphic sites ({n_polymorphic:.0f}) exceeds the total "
                                 f"number of target sites ({self.n_target_sites}) which does not make sense. "
                                 f"The number of target sites unchanged is left unchanged.")
            # ``sfs`` has the sampled monomorphic mass at the origin, so return the pre-sampling spectrum
            return before

        if n_monomorphic == 0:
            self._logger.warning(f"Number of monomorphic sites is zero which should only happen "
                                 f"if there are very few sites considered. Failed to adjust "
                                 f"the number of monomorphic sites.")
            return before

        # scale the sampled monomorphic mass so that each type totals its share of the target sites
        x = self.n_target_sites / n_monomorphic

        updated = {}
        negative = []
        for t in sfs:
            arr = before[t]
            arr[origin] = sampled[t] * x - n_observed[t]

            # a type whose observed sites outnumber its share of the target sites would otherwise be assigned a
            # negative mutational opportunity, which is meaningless downstream
            if arr[origin] < 0:
                negative.append(t)
                arr[origin] = 0

            updated[t] = arr

        if negative:
            self._logger.warning(f"The number of target sites is too small to accommodate the observed sites "
                                 f"of type(s) {negative}; their monomorphic counts were clipped to zero.")

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

        The observed matrix holds mainly the polymorphic-polymorphic pairs. Under a uniform site density the
        ``+/- distance`` window around a site holds on average ``2 * rho_m * distance`` monomorphic sites, where
        ``rho_m = n_monomorphic / region_length`` and ``n_monomorphic = n_target_sites - n_polymorphic``. Each
        polymorphic site of derived count ``j`` therefore pairs with that many monomorphic sites (the ``(0, j)``
        and ``(j, 0)`` entries) and each monomorphic site pairs with that many others (the ``(0, 0)`` entry),
        which the symmetric storage splits evenly between the two slots. The window
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

        # monomorphic-polymorphic pairs: the parse pairs each site with every site within +/- distance, so a
        # polymorphic site of derived count j has 2 * rho_m * distance monomorphic partners; symmetrize()
        # ((A + A.T) / 2) redistributes those between the two symmetric slots without changing their total, so
        # each slot holds rho_m * distance. The (0, 0) entry follows the same accounting for monomorphic pairs.
        contribution = marginal * rho_m * distance
        contribution[0] = 0.0
        contribution[-1] = 0.0
        two_sfs[0, :] += contribution
        two_sfs[:, 0] += contribution
        two_sfs[0, 0] += n_monomorphic * rho_m * distance

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

        #: Buffered ``(position, down-projection vector, type)`` of sites that are still closer than
        #: :attr:`two_sfs_offset` to the current site and thus not yet eligible to form a pair
        self._two_sfs_pending: deque = deque()

        #: Buffered ``(position, down-projection vector, type)`` of the sites inside the eligible band
        #: ``(two_sfs_offset, two_sfs_offset + d]``, i.e. those making up :attr:`_two_sfs_window_sums`
        self._two_sfs_active: deque = deque()

        #: Sum of the down-projection vectors of the sites in the eligible band, per type. Pairing the current
        #: site with this sum in a single outer product is equivalent to pairing it with each site separately
        self._two_sfs_window_sums: Dict[str, np.ndarray] = {}

        #: Number of sites in the eligible band per type, used to drop a type's running sum once its band empties
        self._two_sfs_window_counts: Dict[str, int] = {}

        #: Number of sites processed since the running sums were last rebuilt from the window
        self._two_sfs_since_resync: int = 0

        #: The contig currently held in the two-SFS window
        self._two_sfs_contig: str | None = None

        #: Cache of hypergeometric down-projection vectors keyed by ``(M, n, N)``. The projection depends on
        #: nothing else, and with a fixed sample set only a handful of keys occur, so it is filled lazily
        self._projection_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

    def _reset(self):
        """
        Reset the accumulating state, so that a second :meth:`parse` starts from a clean slate rather than adding
        to the counts of the previous pass. The RNG is re-seeded as well, which keeps repeated parses reproducible.
        The cached reader is discarded too, so that a parse whose predecessor stopped part-way through the input,
        because it raised, starts at the first record rather than wherever the previous pass left off.
        """
        VCFHandler._rewind(self)

        self.n_skipped = 0
        self.n_aa_prob = 0
        self.n_no_ancestral = 0
        self.sfs = defaultdict(lambda: np.zeros(self._joint_shape))
        self._contig_bounds = defaultdict(lambda: (np.inf, -np.inf))
        self._two_sfs_matrices = defaultdict(lambda: np.zeros((self.n + 1, self.n + 1)))
        self._two_sfs_marginal = defaultdict(lambda: np.zeros(self.n + 1))
        self._two_sfs_pending = deque()
        self._two_sfs_active = deque()
        self._two_sfs_window_sums = {}
        self._two_sfs_window_counts = {}
        self._two_sfs_since_resync = 0
        self._two_sfs_contig = None

        # the cached projections are only valid for the sample sizes of the pass that filled them
        self._projection_cache = {}

        self.rng = np.random.default_rng(seed=self.seed)

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
        if self.polarize_probabilistically:
            raw = variant.INFO.get(self.info_ancestral_prob)

            # INFO comes through typed from cyvcf2 but as a plain string from the VCF-Zarr backend, so
            # cast explicitly; a missing value (None) or an empty / '.' sentinel means the site is not
            # probabilistically polarized and is treated as certain (probability 1)
            if raw is not None and raw not in ('', '.'):
                prob = float(raw)

                # the projection mixes the spectrum with its reflection using this as a weight, so anything
                # outside [0, 1] would put negative counts into the SFS
                if not 0.0 <= prob <= 1.0 or not np.isfinite(prob):
                    raise ValueError(f"The ancestral allele probability at {variant.CHROM}:{variant.POS} is "
                                     f"{raw}, which is not a probability in [0, 1].")

                self.n_aa_prob += 1
                return prob

        return 1.0

    def _is_fixed_derived(self, variant: Site) -> bool:
        """
        Whether a site without an alternate allele is fixed for the derived allele, i.e. whether its ancestral
        allele is a base other than the reference allele. Such a site is a fixed difference and its mass belongs
        in the divergence bin, not in the monomorphic-ancestral one. An absent or invalid ancestral allele leaves
        the reference allele as the ancestral one, so that monomorphic sites, which carry no polarization
        information of their own, keep counting as mutational opportunities.

        :param variant: The site.
        :return: Whether the site is fixed for the derived allele.
        """
        aa = variant.INFO.get(self.info_ancestral)

        return aa in bases and aa != variant.REF

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

    def _projection(self, n_samples: int, n_der: int, n: int) -> np.ndarray:
        """
        The hypergeometric down-projection of a site onto ``n`` drawn genotypes: the probability of observing
        ``k = 0, ..., n`` derived alleles when drawing without replacement. The vector depends on nothing but its
        three arguments, so it is looked up in :attr:`_projection_cache` and computed only on the first sighting.

        :param n_samples: The number of called genotypes at the site.
        :param n_der: The number of derived alleles among them.
        :param n: The number of genotypes drawn.
        :return: A read-only array of length ``n + 1``. It is handed out again on every matching site, so callers
            must build a new array instead of writing to it.
        """
        key = (n_samples, n_der, n)
        m = self._projection_cache.get(key)

        if m is None:
            m = hypergeom.pmf(k=range(n + 1), M=n_samples, n=n_der, N=n)
            m.flags.writeable = False
            self._projection_cache[key] = m

        return m

    def _project(self, variant: Site) -> Optional[np.ndarray]:
        """
        Down-project a single site to the one-dimensional SFS, returning the mass over derived-allele counts.

        :param variant: The variant.
        :return: The mass array of length ``n + 1``, or ``None`` if the site is skipped.
        """
        # check `is_snp` property for performance reasons but site may still be monomorphic
        if variant.is_snp:

            # count the called alleles, from the numeric calls where the backend provides them and from the
            # genotype strings otherwise. Multi-character alleles are left to the strings, which count a
            # genotype character at a time and so read an ``AT`` call as two
            site = SiteAlleles.from_site(variant)

            if site is not None and site.single_character:
                counter = site.counts(self._samples_mask)
            else:
                counter = Counter(get_called_bases(variant.gt_bases[self._samples_mask]))

            # number of samples
            n_samples = sum(counter.values())

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

            # determine ancestral allele count
            n_aa = counter.get(aa, 0)

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

                # every site reaching here has at most two alleles, so the reflection is well defined. A site
                # fixed for the derived allele shows a single observed base but is bi-allelic all the same,
                # and its mass is reflected just as at a segregating site
                if aa_prob != 1.0:
                    m[k] += aa_prob
                    m[self.n - k] += 1 - aa_prob
                else:
                    m[k] += 1.0
            else:
                # subsample probabilistically drawing from the hypergeometric distribution (without replacement)
                m = self._projection(n_samples, n_samples - n_aa, self.n)

                # polarize probabilistically, which is well defined for every site reaching here
                if aa_prob != 1.0:
                    m = aa_prob * m + (1 - aa_prob) * m[::-1]

        # if we have a mono-allelic SNPs
        elif is_monomorphic_snp(variant):
            # apply the same coverage requirement as segregating sites, so a low-coverage monomorphic
            # site is not asymmetrically retained and does not inflate the monomorphic:polymorphic ratio
            # (TargetSiteCounter's DummyVariant sites are fully covered by construction and always pass)
            site = SiteAlleles.from_site(variant)

            if site is not None and site.single_character:
                n_called = site.n_called(self._samples_mask)
            else:
                n_called = len(get_called_bases(variant.gt_bases[self._samples_mask]))

            if n_called < self.n:
                self._logger.debug(f'Skipping monomorphic site due to too few samples at {variant.CHROM}:{variant.POS}.')
                return None

            # a site fixed for the derived allele carries all n derived alleles and belongs in the
            # divergence bin, while an ancestral monomorphic site carries none
            n_der = self.n if self._is_fixed_derived(variant) else 0

            aa_prob = self._get_ancestral_prob(variant)

            m = np.zeros(self.n + 1)

            # mispolarization turns one monomorphic bin into the other, so the mass is split between them
            if aa_prob != 1.0:
                m[n_der] = aa_prob
                m[self.n - n_der] = 1 - aa_prob
            else:
                m[n_der] = 1
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

            # count the called alleles per population, from the numeric calls where the backend provides them
            site = SiteAlleles.from_site(variant)

            if site is not None and site.single_character:
                pop_counters = [site.counts(mask) for mask in self._pop_masks]
            else:
                pop_counters = [Counter(get_called_bases(variant.gt_bases[mask])) for mask in self._pop_masks]

            pop_sizes = [sum(c.values()) for c in pop_counters]

            # skip if any population has too few called genotypes
            if any(size < n for size, n in zip(pop_sizes, self._n_per_pop)):
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
            counter: Dict[str, int] = {}
            for c in pop_counters:
                for allele, count in c.items():
                    counter[allele] = counter.get(allele, 0) + count

            n_alleles = len(counter) + (0 if aa in counter else 1)
            if n_alleles > 2:
                self._logger.debug(f'Site has more than two alleles at {variant.CHROM}:{variant.POS} ({dict(counter)}, AA={aa})')
                return None

            # down-project each population independently to a distribution over its derived-allele count
            projections = []
            for c, n_samples, n in zip(pop_counters, pop_sizes, self._n_per_pop):
                n_der = n_samples - c.get(aa, 0)

                if self.subsample_mode == 'random':
                    vec = np.zeros(n + 1)
                    vec[hypergeom.rvs(M=n_samples, n=n_der, N=n, random_state=self.rng)] = 1.0
                else:
                    vec = self._projection(n_samples, n_der, n)

                projections.append(vec)

            # the joint mass is the outer product of the per-population projections
            m = functools.reduce(np.multiply.outer, projections)

            # probabilistically polarize: a mispolarized site flips every population's derived count
            # simultaneously, i.e. reflects the joint array on all axes. This covers sites fixed for the
            # derived allele, which show a single observed base but are bi-allelic all the same
            if aa_prob != 1.0:
                reflected = m[tuple(slice(None, None, -1) for _ in projections)]
                m = aa_prob * m + (1 - aa_prob) * reflected

        # if we have a mono-allelic SNP
        elif is_monomorphic_snp(variant):
            # apply the same per-population coverage requirement as segregating sites
            site = SiteAlleles.from_site(variant)

            if site is not None and site.single_character:
                pop_sizes = [site.n_called(mask) for mask in self._pop_masks]
            else:
                pop_sizes = [len(get_called_bases(variant.gt_bases[mask])) for mask in self._pop_masks]

            if any(size < n for size, n in zip(pop_sizes, self._n_per_pop)):
                self._logger.debug(f'Skipping monomorphic site due to too few samples at {variant.CHROM}:{variant.POS}.')
                return None

            # a site fixed for the derived allele has every population fixed for it, so its mass sits in the
            # all-derived corner, while an ancestral monomorphic site sits at the origin
            origin = (0,) * len(self._joint_shape)
            corner = tuple(n - 1 for n in self._joint_shape)

            fixed_derived = self._is_fixed_derived(variant)
            aa_prob = self._get_ancestral_prob(variant)

            m = np.zeros(self._joint_shape)

            # mispolarization turns one monomorphic corner into the other, so the mass is split between them
            if aa_prob != 1.0:
                m[corner if fixed_derived else origin] = aa_prob
                m[origin if fixed_derived else corner] = 1 - aa_prob
            else:
                m[corner if fixed_derived else origin] = 1.0
        else:
            # skip other types of sites
            self._logger.debug(f'Site is not a valid single nucleotide site at {variant.CHROM}:{variant.POS}.')
            return None

        return m

    def _parse_site_two_sfs(self, variant: Site) -> bool:
        """
        Add a single site to the two-SFS by pairing it with the recently seen sites within the distance window.

        The site's down-projection vector is paired (via an outer product) with that of every buffered site on the
        same contig whose separation falls in ``(two_sfs_offset, two_sfs_offset + d]``. Since the outer product is
        bilinear, all those pairs are accumulated in one outer product with the running sum of the vectors of the
        sites currently in that band (:attr:`_two_sfs_window_sums`), which makes the cost per site independent of
        how many sites the window holds. Sites arrive position-sorted, so the band is maintained with two sliding
        buffers: a site waits in :attr:`_two_sfs_pending` until its separation exceeds ``two_sfs_offset``, then
        moves to :attr:`_two_sfs_active` and joins the running sum, and leaves the sum once its separation exceeds
        ``two_sfs_offset + d``. When stratifications are used, only pairs of sites of the same type are counted,
        into that type's matrix, so the running sums are kept per type.

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

        # reset the window when we move to a new contig (pairs never cross contigs)
        if variant.CHROM != self._two_sfs_contig:
            self._two_sfs_pending.clear()
            self._two_sfs_active.clear()
            self._two_sfs_window_sums.clear()
            self._two_sfs_window_counts.clear()
            self._two_sfs_contig = variant.CHROM

        pos = variant.POS

        # the furthest separation at which a pair is still formed
        max_distance = self.two_sfs_offset + self.d

        # admit the sites whose separation has grown past the offset into the eligible band
        while self._two_sfs_pending and pos - self._two_sfs_pending[0][0] > self.two_sfs_offset:
            entry = self._two_sfs_pending.popleft()
            self._two_sfs_active.append(entry)

            t_prev = entry[2]
            if t_prev in self._two_sfs_window_sums:
                self._two_sfs_window_sums[t_prev] += entry[1]
                self._two_sfs_window_counts[t_prev] += 1
            else:
                # a fresh writable array, as the projection vectors are shared read-only across sites
                self._two_sfs_window_sums[t_prev] = np.array(entry[1], dtype=float)
                self._two_sfs_window_counts[t_prev] = 1

        # retire the sites that are now too far behind to pair with this or any later site
        while self._two_sfs_active and pos - self._two_sfs_active[0][0] > max_distance:
            entry = self._two_sfs_active.popleft()

            t_prev = entry[2]
            self._two_sfs_window_counts[t_prev] -= 1

            # an emptied band is dropped rather than left as a sum of cancelling terms
            if self._two_sfs_window_counts[t_prev] == 0:
                del self._two_sfs_window_counts[t_prev]
                del self._two_sfs_window_sums[t_prev]
            else:
                self._two_sfs_window_sums[t_prev] -= entry[1]

        self._two_sfs_since_resync += 1

        if self._two_sfs_since_resync >= _TWO_SFS_RESYNC_INTERVAL:
            self._resync_two_sfs_window()

        window_sum = self._two_sfs_window_sums.get(t)

        # accumulate the (forward) within-stratum pairs; the matrix is symmetrized once at the end
        if window_sum is not None:
            self._two_sfs_matrices[t] += np.multiply.outer(window_sum, m)

        self._two_sfs_pending.append((pos, m, t))

        return True

    def _resync_two_sfs_window(self):
        """
        Rebuild the two-SFS running window sums from the sites currently in the eligible band, discarding the
        rounding error accumulated by the incremental additions and subtractions (see
        :data:`_TWO_SFS_RESYNC_INTERVAL`).
        """
        sums: Dict[str, np.ndarray] = {}
        counts: Dict[str, int] = {}

        for _, m_prev, t_prev in self._two_sfs_active:
            if t_prev in sums:
                sums[t_prev] += m_prev
                counts[t_prev] += 1
            else:
                sums[t_prev] = np.array(m_prev, dtype=float)
                counts[t_prev] = 1

        self._two_sfs_window_sums = sums
        self._two_sfs_window_counts = counts
        self._two_sfs_since_resync = 0

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
        # restore the per-pass state of the components, which otherwise carries over into a second parse
        self._rewind()

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
                requested = set(self.pops[name])
                mask = np.isin(samples, self.pops[name])

                # a partial match would silently shrink the population, changing the projection without warning
                missing = sorted(requested - set(samples))
                if missing:
                    raise ValueError(f"Samples for population '{name}' were not found in the input: {missing}.")

                self._pop_masks.append(mask)

            # masked filtrations copy this mask, so without it they would fall back to their unmasked branch and
            # decide from ALT / is_snp rather than from the genotypes of the populations being parsed
            self._samples_mask = np.logical_or.reduce(self._pop_masks)

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
        # discard the state of any previous pass before setting up
        self._reset()
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

        # the region length is read off the still-open reader, as closing it below discards the cached reader
        # and asking afterwards would load the whole source a second time
        region_length = self._region_length() if self.two_sfs and self.target_site_counter is not None else None

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
                        matrix, self._two_sfs_marginal['all'], region_length, self.d
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
            # extrapolate the monomorphic corner from the target-site count. The dict is built after the
            # sampling pass, which adds the sampled monomorphic sites to self.sfs and may introduce types
            # that no polymorphic site of the input carried
            if self.target_site_counter is not None and self.n_skipped < self.n_sites:
                self.target_site_counter.count()
                sfs = self.target_site_counter._update_target_sites_joint(dict(self.sfs))
            else:
                sfs = dict(self.sfs)

            return JointSpectra(
                {t: JointSFS(sfs[t], self._pop_names) for t in sorted(sfs)}
            )

        # count target sites
        if self.target_site_counter is not None and self.n_skipped < self.n_sites:
            # count target sites
            self.target_site_counter.count()

            # update target sites (the sampled monomorphic sites have been added to self.sfs by now)
            spectra = self.target_site_counter._update_target_sites(Spectra(dict(self.sfs)))
        else:
            spectra = Spectra(dict(self.sfs))

        return spectra.sort_types()
