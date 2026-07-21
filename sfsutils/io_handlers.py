"""
Handles the reading of VCF, VCF-Zarr, tree-sequence, GFF and FASTA files.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-29"

import atexit
import gzip
import hashlib
import logging
import os
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from functools import cached_property
from typing import Any, List, Iterable, Iterator, TextIO, Dict, Optional, Set, Tuple, Union, Sequence, \
    Protocol, runtime_checkable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from Bio.SeqIO.FastaIO import FastaIterator
from Bio.SeqRecord import SeqRecord
from pandas.errors import SettingWithCopyWarning
from tqdm import tqdm

from .settings import Settings

#: The DNA bases
bases = ["A", "C", "G", "T"]

# logger
logger = logging.getLogger('sfsutils')


@runtime_checkable
class Site(Protocol):
    """
    Structural interface for a single streamed variant site: the abstraction layer over the input backends.
    Both :class:`cyvcf2.Variant` (the VCF backend) and the concrete :class:`~sfsutils.io_handlers.Variant` emitted by the
    tree-sequence and VCF-Zarr backends satisfy it structurally, so the parser, filtrations, annotations and
    stratifications are typed against :class:`~sfsutils.io_handlers.Site` alone rather than a union of the concrete backend types.
    """

    #: The contig.
    CHROM: str

    #: The 1-based position.
    POS: int

    #: The reference allele.
    REF: str

    #: The alternate alleles.
    ALT: List[str]

    #: The INFO field.
    INFO: Dict[str, object]

    #: The per-sample genotype strings (e.g. ``"A/T"``), as for ``cyvcf2.Variant.gt_bases``.
    gt_bases: 'np.ndarray'

    #: The per-haplotype allele indices into ``[REF] + ALT``, of shape ``(n_samples, ploidy)`` and integer
    #: dtype, with ``-1`` for a missing call. Optional: it is ``None`` (or absent) on backends that carry
    #: the genotypes as strings only, in which case consumers read :attr:`gt_bases` instead.
    allele_indices: Optional['np.ndarray']

    #: Whether the site is an SNP.
    is_snp: bool

    #: Whether the site is an MNP.
    is_mnp: bool

    #: Whether the site is an indel.
    is_indel: bool

    #: Whether the site is a deletion.
    is_deletion: bool

    #: Whether the site is a structural variant.
    is_sv: bool


def get_called_bases(genotypes: Sequence[str]) -> np.ndarray:
    """
    Get the called bases from a list of calls.

    :param genotypes: Array of genotypes in the form of strings.
    :return: Array of called bases.
    """
    # join genotypes
    joined_genotypes = ''.join(genotypes).replace('|', '/')

    # convert to numpy array of characters
    char_array = np.array(list(joined_genotypes))

    # return only characters that are in the bases list
    return char_array[np.isin(char_array, bases)]


#: The bases as a set, for fast membership tests in the per-site filtration hot path
_base_set = frozenset(bases)


def get_distinct_called_bases(genotypes: Sequence[str]) -> set:
    """
    Get the set of distinct called bases. Equivalent to ``set(get_called_bases(genotypes))`` but avoids building
    the intermediate arrays, which matters because the filtrations run this on every site.

    :param genotypes: Array of genotypes in the form of strings.
    :return: The distinct called bases.
    """
    return set(''.join(genotypes)) & _base_set


def get_distinct_called_alleles(genotypes: Sequence[str]) -> set:
    """
    Get the set of distinct called alleles, keeping multi-character alleles intact.

    :param genotypes: Array of genotypes in the form of strings.
    :return: The distinct called alleles.
    """
    return {
        allele
        for genotype in genotypes
        for allele in genotype.replace('|', '/').split('/')
        if allele != '' and set(allele) <= _base_set
    }


def get_called_alleles(genotypes: Sequence[str]) -> np.ndarray:
    """
    Get the distinct called alleles from a list of calls. Multi-character alleles stay intact, so an MNP
    contributes one allele per haplotype rather than one per base.

    :param genotypes: Array of genotypes in the form of strings.
    :return: Array of distinct called alleles.
    """
    return np.array(sorted(get_distinct_called_alleles(genotypes)), dtype=object)


class SiteAlleles:
    """
    Numeric view of a site's genotypes: the per-haplotype allele indices paired with the site's allele
    strings. The backends hold the calls as indices already, so reading them here spares both the join into
    genotype strings and the split back out of them, which dominates the per-site cost of the filtrations
    and of the down-projection.

    An allele is called only where its string consists of DNA bases alone, matching :func:`get_called_bases`
    and :func:`get_distinct_called_alleles`: ``N``, ``*``, ``<NON_REF>`` and a missing call contribute
    nothing. Counting haplotypes and counting genotype characters coincide only where every allele of the
    site is a single character, which :attr:`single_character` reports.
    """

    def __init__(self, indices: np.ndarray, alleles: Sequence[str]):
        """
        Initialize the view.

        :param indices: The per-haplotype allele indices, of shape ``(n_samples, ploidy)``, with ``-1``
            for a missing call.
        :param alleles: The site's allele strings, the reference first.
        """
        #: The site's allele strings, the reference first
        self.alleles: List[str] = list(alleles)

        #: The per-haplotype allele indices
        self.indices: np.ndarray = indices

        #: Whether every allele of the site is a single character
        self.single_character: bool = all(len(a) == 1 for a in self.alleles)

        #: Whether the allele at each index is called, with a trailing slot absorbing the missing and
        #: out-of-range calls so that validity is a single lookup
        self._called: np.ndarray = np.zeros(len(self.alleles) + 1, dtype=bool)

        for i, allele in enumerate(self.alleles):
            self._called[i] = allele != '' and set(allele) <= _base_set

    #: The site the last view was built for, alongside that view. Every filtration and the parser itself
    #: ask for the view of the site they are handed, and the site is handed on to the next of them
    #: unchanged, so it is built once and shared. The site is held rather than its identity alone, since
    #: the sites are transient and the address of one that has been released is handed out again.
    _cached: Tuple[Optional['Site'], Optional['SiteAlleles']] = (None, None)

    @classmethod
    def from_site(cls, variant: 'Site') -> Optional['SiteAlleles']:
        """
        The view for a site, where its backend provides the numeric calls, reusing the view of the site
        this was last called for.

        :param variant: The site.
        :return: The view, or ``None`` where the backend carries the genotypes as strings only.
        """
        # read the pair in one go: it is replaced as a whole, so the site and the view cannot be observed
        # out of step with one another
        site, view = cls._cached

        if site is variant:
            return view

        view = cls._build(variant)
        cls._cached = (variant, view)

        return view

    @classmethod
    def _build(cls, variant: 'Site') -> Optional['SiteAlleles']:
        """
        Build the view for a site.

        :param variant: The site.
        :return: The view, or ``None`` where the backend carries the genotypes as strings only.
        """
        indices = getattr(variant, 'allele_indices', None)

        if indices is None:
            # cyvcf2 holds the calls on its Genotypes object, whose last column carries the phase flag
            genotype = getattr(variant, 'genotype', None)

            if genotype is None:
                return None

            indices = np.asarray(genotype.array())

            if indices.ndim != 2 or indices.shape[1] < 2:
                return None

            indices = indices[:, :-1]

        return cls(indices, [variant.REF or ''] + list(variant.ALT))

    def _bins(self, mask: Optional[np.ndarray]) -> np.ndarray:
        """
        The number of haplotypes of the selected samples carrying each allele index, shifted by the two
        negative sentinels so that a missing call (``-1``) and the fill of a shorter genotype (``-2``)
        land in the two leading bins and an out-of-range call past the alleles. Binning the raw indices
        spares the clamp and the intermediate arrays it needs, which matters because the parser runs this
        on every site.

        :param mask: The boolean samples mask, or ``None`` for every sample.
        :return: The per-index counts, allele ``i`` in bin ``i + 2``.
        """
        indices = self.indices if mask is None else self.indices[mask]

        # the calls come in the narrowest dtype that holds the site's alleles, in which the shift of the
        # sentinels would wrap the highest allele index round to a negative code; bincount widens to
        # intp regardless, so widening here first costs nothing
        codes = np.asarray(indices, dtype=np.intp).ravel() + 2

        # a sentinel beyond the two the VCF-Zarr spec defines would index before the leading bin
        if codes.size and codes.min() < 0:
            codes = codes[codes >= 0]

        return np.bincount(codes, minlength=len(self.alleles) + 3)

    def n_called(self, mask: Optional[np.ndarray] = None) -> int:
        """
        The number of called haplotypes among the selected samples.

        :param mask: The boolean samples mask, or ``None`` for every sample.
        :return: The number of called haplotypes.
        """
        n = len(self.alleles)
        bins = self._bins(mask)

        return int(bins[2:n + 2] @ self._called[:n])

    def counts(self, mask: Optional[np.ndarray] = None) -> Dict[str, int]:
        """
        The number of called haplotypes carrying each allele among the selected samples. Alleles that no
        selected haplotype carries are absent from the mapping, and two indices sharing an allele string
        are summed into one entry.

        :param mask: The boolean samples mask, or ``None`` for every sample.
        :return: The per-allele haplotype counts.
        """
        bins = self._bins(mask)

        observed: Dict[str, int] = {}
        for i, allele in enumerate(self.alleles):
            count = int(bins[i + 2]) if self._called[i] else 0

            if count:
                observed[allele] = observed.get(allele, 0) + count

        return observed

    def distinct(self, mask: Optional[np.ndarray] = None) -> Set[str]:
        """
        The distinct alleles called among the selected samples.

        :param mask: The boolean samples mask, or ``None`` for every sample.
        :return: The distinct called alleles.
        """
        bins = self._bins(mask)

        return {allele for i, allele in enumerate(self.alleles) if bins[i + 2] and self._called[i]}


def get_major_base(genotypes: Sequence[str]) -> str | None:
    """
    Get the major base from a list of calls.

    :param genotypes: Array of genotypes in the form of strings.
    :return: Major base.
    """
    # get the called bases
    bases = get_called_bases(genotypes)

    if len(bases) > 0:
        return Counter(bases).most_common()[0][0]


def _is_snp(ref: str, alt: Sequence[str]) -> bool:
    """
    Whether a site is a single-nucleotide polymorphism, under the rule htslib applies and cyvcf2 surfaces
    through ``is_snp``: a single-character reference, which may be ``N`` or another IUPAC code, together
    with at least one alternate allele, each of them a single base. The spanning deletion ``*``, the
    missing allele ``.`` and a symbolic ``<...>`` allele are thereby excluded, and the non-VCF backends
    classify a site exactly as the same records read through cyvcf2 do.

    :param ref: The reference allele.
    :param alt: The alternate alleles.
    :return: Whether the site is a single-nucleotide polymorphism.
    """
    return len(ref) == 1 and len(alt) > 0 and all(a in _base_set for a in alt)


def is_monomorphic_snp(variant: Site) -> bool:
    """
    Whether the given variant is a monomorphic SNP.

    :param variant: The site
    :return: Whether the site is a monomorphic SNP
    """
    return (not (variant.is_snp or variant.is_mnp or variant.is_indel or variant.is_deletion or variant.is_sv)
            and not variant.ALT and variant.REF in bases)


def count_indexed_sites(vcf: str) -> int | None:
    """
    Read the number of records of a VCF from its tabix or CSI index, which htslib stores alongside the
    per-contig offsets, so the file itself does not have to be decompressed a second time.

    :param vcf: The path to the input
    :return: The number of records, or ``None`` where the file carries no readable index
    """
    # only look for an index next to a local file: a URL or a directory (a VCF-Zarr store) has none, and
    # probing for one would cost a request
    if not any(os.path.exists(vcf + extension) for extension in ('.tbi', '.csi')):
        return None

    try:
        from cyvcf2 import VCF
    except ImportError:
        return None

    try:
        reader = VCF(vcf)

        try:
            n = int(reader.num_records)
        finally:
            reader.close()
    except Exception as e:
        logger.debug(f"Could not read the record count of '{vcf}' from its index: {e}")
        return None

    # an index written without the record-count metadata reports zero, which is indistinguishable from an
    # empty file, so the caller falls back to counting
    return n if n > 0 else None


def count_sites(
        vcf: str | Iterable['cyvcf2.Variant'],
        max_sites: int = np.inf,
        desc: str = 'Counting sites'
) -> int:
    """
    Count the number of sites in the input. Where the input is an indexed VCF, the count comes from the
    index rather than from a pass over the records.

    :param vcf: The path to the input or an iterable of variants
    :param max_sites: Maximum number of sites to consider
    :param desc: Description for the progress bar
    :return: Number of sites
    """

    # if we don't have a file path, we can just count the number of variants
    if not isinstance(vcf, str):
        return len(list(vcf))

    indexed = count_indexed_sites(vcf)
    if indexed is not None:
        return int(min(indexed, max_sites))

    i = 0
    with open_file(vcf) as f:

        with tqdm(disable=Settings.disable_pbar, desc=desc) as pbar:

            for line in f:
                if not line.startswith('#'):
                    i += 1
                    pbar.update()

                # stop counting if max_sites was reached
                if i >= max_sites:
                    break

    return i


def download_if_url(path: str, cache: bool = True, desc: str = 'Downloading file') -> str:
    """
    Download the file if it is a URL.

    :param path: The path to the file.
    :param cache: Whether to cache the file.
    :param desc: Description for the progress bar
    :return: The path to the downloaded file or the original path.
    """
    if FileHandler.is_url(path):
        # download the file and return path
        return FileHandler.download_file(path, cache=cache, desc=desc)

    return path


def open_file(file: str) -> TextIO:
    """
    Open a file, either gzipped or not.

    :param file: File to open
    :return: stream
    """
    if file.endswith('.gz'):
        return gzip.open(file, "rt")

    return open(file, 'r')


class FileHandler:
    """
    Base class for file handling.
    """

    #: The logger instance
    _logger = logger.getChild(__qualname__)

    #: Cache of gzipped file path -> its decompressed temporary copy, so a rewind reuses the copy
    _unzipped: Dict[str, str] = {}

    def __init__(self, cache: bool = True, aliases: Dict[str, List[str]] = {}):
        """
        Create a new FileHandler instance.

        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        #: Whether to cache files that are downloaded from URLs
        self.cache: bool = cache

        #: The contig mappings
        self._alias_mappings, self.aliases = self._expand_aliases(aliases)

    @staticmethod
    def _expand_aliases(alias_dict: Dict[str, List[str]]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """
        Expand the contig aliases.
        """
        # map alias to primary alias
        mappings = {}

        # map primary alias to all aliases
        aliases = {}

        for contig, alias_list in alias_dict.items():
            all_aliases = alias_list + [contig]
            aliases[contig] = all_aliases

            for alias in all_aliases:
                mappings[alias] = contig

        return mappings, aliases

    @staticmethod
    def is_url(path: str) -> bool:
        """
        Check if the given path is a URL.

        :param path: The path to check.
        :return: ``True`` if the path is a URL, ``False`` otherwise.
        """
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def download_if_url(self, path: str) -> str:
        """
        Download the file if it is a URL.

        :param path: The path to the file.
        :return: The path to the downloaded file or the original path.
        """
        return download_if_url(path, cache=self.cache, desc=f'{self.__class__.__name__}>Downloading file')

    @staticmethod
    def unzip_if_zipped(file: str):
        """
        If the given file is gzipped, unzip it and return the path to the unzipped file.
        If the file is not gzipped, return the path to the original file.

        :param file: The path to the file.
        :return: The path to the unzipped file, or the original file if it was not gzipped.
        """
        # check if the file extension is .gz
        if file.endswith('.gz'):
            # reuse a previous decompression of the same file: a rewind would otherwise decompress the
            # whole reference again and leak another full-size temporary copy
            cached = FileHandler._unzipped.get(file)
            if cached is not None and os.path.exists(cached):
                return cached

            suffix = os.path.splitext(file[:-3])[1] or '.tmp'
            fd, unzipped = tempfile.mkstemp(suffix=suffix)

            logger.info(f'Unzipping {file} to {unzipped}')

            with gzip.open(file, 'rb') as f_in:
                with os.fdopen(fd, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            FileHandler._unzipped[file] = unzipped
            atexit.register(lambda p=unzipped: os.path.exists(p) and os.remove(p))

            return unzipped

        return file

    @staticmethod
    def get_filename(url: str):
        """
        Return the file extension of a URL.

        :param url: The URL to get the file extension from.
        :return: The file extension.
        """
        return os.path.basename(urlparse(url).path)

    @staticmethod
    def hash(s: str) -> str:
        """
        Return a truncated SHA1 hash of a string.

        :param s: The string to hash.
        :return: The SHA1 hash.
        """
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    @classmethod
    def download_file(cls, url: str, cache: bool = True, desc: str = 'Downloading file') -> str:
        """
        Download a file from a URL.

        :param cache: Whether to cache the file.
        :param url: The URL to download the file from.
        :param desc: Description for the progress bar
        :return: The path to the downloaded file.
        """
        # extract the file extension from the URL
        filename = FileHandler.get_filename(url)

        # create a temporary file path
        path = tempfile.gettempdir() + '/' + FileHandler.hash(url) + '.' + filename

        # check if the file is already cached
        if cache and os.path.exists(path):
            cls._logger.info(f'Using cached file at {path}')
            return path

        cls._logger.info(f'Downloading file from {url}')

        # start the stream
        response = requests.get(url, stream=True)

        # check if the request was successful
        response.raise_for_status()

        # create a temporary file with the original file extension
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192

            with tqdm(total=total_size,
                      unit='B',
                      unit_scale=True,
                      desc=desc,
                      disable=Settings.disable_pbar) as pbar:

                # write the file to disk
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        tmp.write(chunk)
                        pbar.update(len(chunk))

        # rename the file to the original file extension
        os.rename(tmp.name, path)

        if cache:
            cls._logger.info(f'Cached file at {path}')

        return path

    def get_aliases(self, contig: str) -> List[str]:
        """
        Get all aliases for the given contig alias including the primary alias.

        :param contig: The contig.
        :return: The aliases.
        """
        if contig in self._alias_mappings:
            return self.aliases[self._alias_mappings[contig]]

        return [contig]


class FASTAHandler(FileHandler):

    def __init__(self, fasta: str | None, cache: bool = True, aliases: Dict[str, List[str]] = {}):
        """
        Create a new FASTAHandler instance.

        :param fasta: The path to the FASTA file.
        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        FileHandler.__init__(self, cache=cache, aliases=aliases)

        #: The path to the FASTA file.
        self.fasta: str = fasta

        #: The current contig.
        self._contig: SeqRecord | None = None

        #: The alias sets the FASTA carries no contig for, so a site on an absent contig costs one
        #: set lookup rather than a scan over the whole file
        self._absent: Set[frozenset] = set()

    @cached_property
    def _local(self) -> str | None:
        """
        The path to the FASTA on the local filesystem, downloaded and decompressed where necessary, so
        that it can be seeked into.

        :return: The local path.
        """
        if self.fasta is None:
            return None

        return self.unzip_if_zipped(self.download_if_url(self.fasta))

    @cached_property
    def _offsets(self) -> Dict[str, int]:
        """
        The byte offset of every contig's header line, in file order. Only the header lines are looked at,
        so building this costs one sequential pass and no sequence parsing, and it turns a lookup into a
        seek rather than into a scan from wherever the previous one stopped.

        :return: The offset of each contig.
        """
        offsets: Dict[str, int] = {}

        if self._local is None:
            return offsets

        with open(self._local, 'rb') as f:
            offset = 0

            for line in f:
                if line[:1] == b'>':
                    # the record id is the header up to the first whitespace, as Biopython parses it
                    parts = line[1:].split()
                    name = parts[0].decode() if parts else ''

                    # a duplicated name resolves to its first record, as a forward scan would
                    offsets.setdefault(name, offset)

                offset += len(line)

        return offsets

    @cached_property
    def _handle(self) -> TextIO:
        """
        The handle the contigs are read through, held open across lookups so that seeking to a contig does
        not reopen the file.

        :return: The handle.
        """
        return open(self._local, 'r')

    @cached_property
    def _ref(self) -> FastaIterator | None:
        """
        Get the reference reader.

        :return: The reference reader.
        """
        if self.fasta is None:
            return

        return self.load_fasta(self.fasta)

    def load_fasta(self, file: str) -> FastaIterator:
        """
        Load a FASTA file into a dictionary.

        :param file: The path to The FASTA file path, possibly gzipped or a URL
        :return: Iterator over the sequences.
        """
        self._logger.info("Loading FASTA file")

        # download and unzip if necessary
        local_file = self.unzip_if_zipped(self.download_if_url(file))

        return SeqIO.parse(local_file, 'fasta')

    def get_contig(self, aliases, rewind: bool = True, notify: bool = True) -> SeqRecord:
        """
        Get the contig from the FASTA file. The contig is looked up in the header index and read by
        seeking to it, so the sites may visit the contigs in any order at the same cost.

        Note that ``pyfaidx`` would be more efficient here, but there were problems when running it in parallel.

        :param aliases: The contig aliases.
        :param rewind: Unused, the lookup does not depend on the position of a cursor.
        :param notify: Unused, the lookup does not depend on the position of a cursor.
        :return: The contig.
        :raises LookupError: Where the FASTA carries none of the aliases.
        """
        # if the contig is already loaded, we can just return it
        if self._contig is not None and self._contig.id in aliases:
            return self._contig

        key = frozenset(aliases)

        if key not in self._absent:
            offsets = self._offsets

            for alias in aliases:
                offset = offsets.get(alias)

                if offset is not None:
                    self._handle.seek(offset)
                    self._contig = next(SeqIO.parse(self._handle, 'fasta'))

                    return self._contig

            self._absent.add(key)

        raise LookupError(f'None of the contig aliases {aliases} were found in the FASTA file.')

    def get_contig_names(self) -> List[str]:
        """
        Get the names of the contigs in the FASTA file.

        :return: The contig names.
        """
        if self.fasta is None:
            return []

        return list(self._offsets)

    def _rewind(self):
        """
        Rewind the fasta iterator.
        """
        # check the instance cache directly: hasattr() would fire the cached_property and open (and
        # for a URL download and decompress) the very reference we are about to discard
        if '_ref' in self.__dict__:
            # noinspection all
            del self._ref


class GFFHandler(FileHandler):
    """
    GFF handler.
    """

    #: The number of GFF lines read at a time, which bounds the memory the attributes column occupies
    _gff_block: int = 50000

    def __init__(self, gff: str | None, cache: bool = True, aliases: Dict[str, List[str]] = {}):
        """
        Constructor.

        :param gff: The path to the GFF file.
        :param cache: Whether to cache the file.
        :param aliases: The contig aliases.
        """
        FileHandler.__init__(self, cache=cache, aliases=aliases)

        #: The logger
        self._logger = logger.getChild(self.__class__.__name__)

        #: The GFF file path
        self.gff = gff

    @cached_property
    def _cds(self) -> pd.DataFrame | None:
        """
        The coding sequences.

        :return: Dataframe with coding sequences.
        """
        if self.gff is None:
            return

        return self._load_cds()

    @staticmethod
    def _cds_of_block(block: pd.DataFrame) -> pd.DataFrame:
        """
        The coding sequences of one block of GFF lines, with their coordinates typed and their parent
        transcript extracted.

        :param block: The block of GFF lines.
        :return: The coding sequences of the block.
        """
        # filter for coding sequences
        cds = block[block['type'] == 'CDS']

        # drop rows with NA values
        cds = cds.dropna()

        # convert start and end to int
        cds = cds.assign(start=cds['start'].astype(int), end=cds['end'].astype(int))

        # the transcript each coding sequence belongs to. Codons that span a CDS boundary are completed from the
        # adjacent CDS, which is only meaningful within one transcript: the nearest CDS by coordinate may belong
        # to an unrelated gene, possibly on the opposite strand. GFF3 spells this Parent=, GTF transcript_id "..".
        cds['parent'] = cds['attributes'].str.extract(
            r'(?:^|;)\s*(?:Parent|transcript_id)[=\s]"?([^;"]+)', expand=False)

        # drop type and attributes columns
        cds = cds.drop(columns=['type', 'attributes'])

        # the coding sequences of the transcripts of one gene repeat the same coordinates, so most of a
        # block is already redundant; the surviving rows are deduplicated again across the blocks, which
        # keeps the same first occurrence a single-pass read would
        return cds.drop_duplicates(subset=['seqid', 'start', 'end'])

    def _load_cds(self) -> pd.DataFrame:
        """
        Load coding sequences from a GFF file.

        :return: The DataFrame.
        """
        self._logger.info(f'Loading GFF file')

        # download and unzip if necessary
        local_file = self.unzip_if_zipped(self.download_if_url(self.gff))

        # column labels for GFF file
        col_labels = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']

        dtypes = dict(
            seqid='category',
            type='category',
            start=float,  # temporarily load as float to handle NA values
            end=float,  # temporarily load as float to handle NA values
            strand='category',
            phase='category'
        )

        # a whole-genome annotation is dominated by the attributes of the many non-CDS records, so the
        # file is read a block of lines at a time and only the coding sequences of each block, with their
        # attributes already reduced to the parent transcript, are carried over
        blocks = pd.read_csv(
            local_file,
            sep='\t',
            comment='#',
            names=col_labels,
            dtype=dtypes,
            usecols=['seqid', 'type', 'start', 'end', 'strand', 'phase', 'attributes'],
            chunksize=self._gff_block
        )

        categorical = ('seqid', 'strand', 'phase')
        categories: Dict[str, Dict[str, None]] = {column: {} for column in categorical}
        kept = []

        for block in blocks:
            # a category the block does not carry is not in its own dtype, so the categories of every
            # block are collected: a contig without any coding sequence keeps its category and thereby
            # its (empty) group in a per-contig count, as it has when the file is read in one pass
            for column in categorical:
                categories[column].update(dict.fromkeys(block[column].cat.categories))

            kept.append(self._cds_of_block(block))

        df = pd.concat(kept, ignore_index=True) if kept else self._cds_of_block(
            pd.DataFrame(columns=col_labels))

        # the categories are ordered by name rather than by the block they first appear in, so that the
        # frame, which is sorted on the contig, does not depend on where the blocks fall
        for column in categorical:
            df[column] = pd.Categorical(df[column], categories=sorted(categories[column]))

        # remove duplicates
        df = df.drop_duplicates(subset=['seqid', 'start', 'end'])

        # sort by seqid and start
        df.sort_values(by=['seqid', 'start'], inplace=True)

        return df

    def _count_target_sites(self, remove_overlaps: bool = False, contigs: List[str] = None) -> Dict[str, int]:
        """
        Count the number of target sites in a GFF file.

        :param remove_overlaps: Whether to remove overlapping coding sequences.
        :param contigs: The contigs to consider.
        :return: The number of target sites per chromosome/contig.
        """
        cds = self._add_lengths(
            cds=self._load_cds(),
            remove_overlaps=remove_overlaps,
            contigs=contigs
        )

        # group by 'seqid' and calculate the sum of 'length'
        target_sites = cds.groupby('seqid', observed=False)['length'].sum().to_dict()

        # filter explicitly for contigs if necessary
        # as seqid is a categorical variable, groups were retained even if they were filtered out
        if contigs is not None:
            target_sites = {k: v for k, v in target_sites.items() if k in contigs}

        return target_sites

    @staticmethod
    def remove_overlaps(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove overlapping coding sequences.

        :param df: The coding sequences.
        :return: The coding sequences without overlaps.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SettingWithCopyWarning)

            # shift within each contig, so the last CDS of one contig is not compared against the first
            # CDS of the next (whose start coordinate restarts low) and spuriously dropped as overlapping
            df['overlap'] = df.groupby('seqid')['start'].shift(-1) <= df['end']

        df = df[~df['overlap']]

        return df.drop(columns=['overlap'])

    @staticmethod
    def _add_lengths(cds: pd.DataFrame, remove_overlaps: bool = False, contigs: List[str] = None) -> pd.DataFrame:
        """
        Compute coding sequences lengths.

        :param cds: The coding sequences.
        :param remove_overlaps: Whether to remove overlapping coding sequences.
        :param contigs: The contigs to consider.
        :return: The coding sequences with lengths.
        """
        # filter for contigs if necessary
        if contigs is not None:
            cds = cds[cds['seqid'].isin(contigs)]

        # remove duplicates
        cds = cds.drop_duplicates(subset=['seqid', 'start'])

        # remove overlaps
        if remove_overlaps:
            cds = GFFHandler.remove_overlaps(cds)

        # catch warning when adding a new column to a slice of a DataFrame
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SettingWithCopyWarning)

            # create a new column for the difference between 'end' and 'start'
            cds['length'] = cds['end'] - cds['start'] + 1

        return cds


class VCFHandler(FileHandler):
    """
    Base class for variant source handling.
    """

    def __init__(
            self,
            vcf: "str | 'tskit.TreeSequence'",
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {}
    ):
        """
        Create a new variant handler instance.

        :param vcf: The variant source: a VCF path (gzipped or a URL), a VCF-Zarr store (.vcz/.zarr), or a tskit tree sequence (a .trees path or a TreeSequence object)
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        FileHandler.__init__(self, cache=cache, aliases=aliases)

        #: The variant source (a path or a tskit TreeSequence object)
        self.vcf = os.fspath(vcf) if isinstance(vcf, os.PathLike) else vcf

        #: The tag in the INFO field that contains the ancestral allele
        self.info_ancestral: str = info_ancestral

        #: Maximum number of sites to consider
        self.max_sites: int = int(max_sites) if not np.isinf(max_sites) else np.inf

        #: Seed for the random number generator
        self.seed: Optional[int] = int(seed) if seed is not None else None

        #: Random generator instance
        self.rng = np.random.default_rng(seed=seed)

    @cached_property
    def _reader(self):
        """
        Get the variant reader for the configured source (VCF, VCF-Zarr, or tree sequence).

        :return: The variant reader.
        """
        return self._open_reader()

    def _rewind(self):
        """
        Rewind the variant iterator.
        """
        # check the instance cache directly: hasattr() would fire the cached_property and open the very
        # reader we are about to discard
        if '_reader' in self.__dict__:
            # noinspection all
            del self._reader

    @staticmethod
    def _is_tree_sequence(source) -> bool:
        """
        Whether the source is a tskit tree sequence (a ``TreeSequence`` object or a ``.trees`` path).

        :param source: The variant source.
        :return: Whether it is a tree sequence.
        """
        if isinstance(source, str):
            return source.endswith('.trees')

        return type(source).__name__ == 'TreeSequence' and type(source).__module__.split('.')[0] == 'tskit'

    @staticmethod
    def _is_zarr_store(source) -> bool:
        """
        Whether the source is a VCF-Zarr store path (a ``.vcz`` or ``.zarr`` directory).

        :param source: The variant source.
        :return: Whether it is a VCF-Zarr store.
        """
        return isinstance(source, str) and source.rstrip('/').endswith(('.vcz', '.zarr'))

    def _open_reader(self):
        """
        Open the appropriate variant reader for the configured source. A tskit tree sequence and a
        VCF-Zarr store are read through :class:`TskitVariantReader` and :class:`~sfsutils.io_handlers.ZarrVariantReader`;
        everything else is read from VCF via cyvcf2.

        :return: The variant reader.
        """
        if self._is_tree_sequence(self.vcf):
            return TskitVariantReader(self._load_tree_sequence(self.vcf))

        if self._is_zarr_store(self.vcf):
            return ZarrVariantReader(self.vcf, info_ancestral=self.info_ancestral)

        # a pre-built VariantReader passed directly as the source. We require a VariantReader rather
        # than an arbitrary iterable of sites because the parser needs a re-iterable source that also
        # exposes ``samples``, ``seqnames`` and ``count_sites()``; a bare iterable (or a cyvcf2.VCF
        # object) provides none of these and would fail later with an opaque error.
        if not isinstance(self.vcf, str):
            if isinstance(self.vcf, VariantReader):
                return self.vcf

            raise TypeError(
                f"Unsupported variant source of type '{type(self.vcf).__name__}'. A non-path source must be a "
                f"VariantReader (e.g. TskitVariantReader or ZarrVariantReader), which exposes 'samples', "
                f"'seqnames' and 'count_sites()' and is re-iterable. A bare iterable of sites or a cyvcf2.VCF "
                f"object is not supported; pass a VCF path/URL or wrap your sites in a VariantReader instead."
            )

        return self.load_vcf()

    @staticmethod
    def _load_tree_sequence(source):
        """
        Resolve a tree-sequence source (a ``TreeSequence`` object or a ``.trees`` path) to a loaded
        tree sequence.

        :param source: The tree-sequence source.
        :return: The loaded tree sequence.
        """
        if not isinstance(source, str):
            return source

        try:
            import tskit
        except ImportError:
            raise ImportError(
                "Reading tree sequences in sfsutils requires the optional 'tskit' package. "
                "Please install sfsutils with the 'arg' extra: pip install sfsutils[arg]"
            )

        return tskit.load(source)

    def load_vcf(self) -> 'cyvcf2.VCF':
        """
        Open a VCF file for streaming.

        :return: The VCF reader.
        """
        try:
            from cyvcf2 import VCF
        except ImportError:
            raise ImportError(
                "VCF support in sfsutils requires the optional 'cyvcf2' package. "
                "Please install sfsutils with the 'vcf' extra: pip install sfsutils[vcf]"
            )

        self._logger.info("Loading VCF file")

        return VCF(self.download_if_url(self.vcf))

    @cached_property
    def n_sites(self) -> int:
        """
        Get the number of sites in the input.

        :return: Number of sites
        """
        return self.count_sites()

    def count_sites(self) -> int:
        """
        Count the number of sites in the source.

        :return: Number of sites
        """
        # tree sequences and VCF-Zarr stores expose their site count directly
        if self._is_tree_sequence(self.vcf) or self._is_zarr_store(self.vcf):
            return int(min(self._reader.count_sites(), self.max_sites))

        # a pre-built VariantReader passed directly as the source. VariantReader.count_sites() opens a
        # fresh iteration pass, so it does not exhaust the source and the later parse() pass still sees
        # every site (a plain generator would have been consumed here). Non-VariantReader sources are
        # rejected in _open_reader, so self._reader is guaranteed to be a VariantReader.
        if not isinstance(self.vcf, str):
            return int(min(self._reader.count_sites(), self.max_sites))

        return count_sites(
            vcf=self.download_if_url(self.vcf),
            max_sites=self.max_sites,
            desc=f'{self.__class__.__name__}>Counting sites'
        )

    def get_pbar(self, desc: str = "Processing sites", total: int | None = 0) -> tqdm:
        """
        Return a progress bar for the number of sites.

        :param desc: Description for the progress bar
        :param total: Total number of items
        :return: tqdm
        """
        return tqdm(
            total=self.n_sites if total == 0 else total,
            disable=Settings.disable_pbar,
            desc=desc
        )


class MultiHandler(VCFHandler, FASTAHandler, GFFHandler):
    """
    Handle variant sources, FASTA and GFF files.
    """

    def __init__(
            self,
            source: "str | os.PathLike | 'tskit.TreeSequence' | VariantReader | None" = None,
            fasta: str | None = None,
            gff: str | None = None,
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {},
            vcf: "str | os.PathLike | 'tskit.TreeSequence' | VariantReader | None" = None
    ):
        """
        Create a new MultiHandler instance.

        :param source: The variant source: a VCF path (gzipped or a URL), a VCF-Zarr store (.vcz/.zarr), a
            tskit tree sequence (a .trees path or a TreeSequence object), or a pre-built :class:`~sfsutils.io_handlers.VariantReader`.
        :param fasta: The path to the FASTA file.
        :param gff: The path to the GFF file.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        :param vcf: Deprecated alias for ``source``, kept for backward compatibility. Provide either
            ``source`` or ``vcf``, not both.
        :raises ValueError: If both ``source`` and ``vcf`` are given, or if neither is given.
        """
        source = self._resolve_source(source, vcf)

        # initialize vcf handler
        VCFHandler.__init__(
            self,
            vcf=source,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed,
            cache=cache,
            aliases=aliases
        )

        # initialize fasta handler
        FASTAHandler.__init__(
            self,
            fasta=fasta,
            cache=cache,
            aliases=aliases
        )

        # initialize gff handler
        GFFHandler.__init__(
            self,
            gff=gff,
            cache=cache,
            aliases=aliases
        )

        #: The variant source. Alias of :attr:`vcf`, which is kept for backward compatibility.
        self.source = self.vcf

    @staticmethod
    def _resolve_source(source, vcf):
        """
        Reconcile the ``source`` parameter with its deprecated ``vcf`` alias.

        :param source: The variant source.
        :param vcf: The deprecated ``vcf`` alias.
        :return: The resolved variant source.
        :raises ValueError: If both ``source`` and ``vcf`` are given, or if neither is given.
        """
        if source is not None and vcf is not None:
            raise ValueError(
                "Provide either 'source' or the deprecated 'vcf' alias, not both."
            )

        if source is None and vcf is None:
            raise ValueError(
                "A variant source must be provided via 'source' (or the deprecated 'vcf' alias)."
            )

        if vcf is not None:
            warnings.warn(
                "The 'vcf' argument is deprecated; use 'source' instead.",
                DeprecationWarning,
                stacklevel=3
            )

        return source if source is not None else vcf

    def _require_fasta(self, class_name: str):
        """
        Raise an exception if no FASTA file was provided.

        :param class_name: The name of the class that requires a FASTA file.
        """
        if self.fasta is None:
            raise ValueError(f'{class_name} requires a FASTA file to be specified.')

    def _require_gff(self, class_name: str):
        """
        Raise an exception if no GFF file was provided.

        :param class_name: The name of the class that requires a GFF file.
        """
        if self.gff is None:
            raise ValueError(f'{class_name} requires a GFF file to be specified.')

    def _rewind(self):
        """
        Rewind the fasta and vcf handler.
        """
        FASTAHandler._rewind(self)
        VCFHandler._rewind(self)


class NoTypeException(BaseException):
    """
    Exception thrown when no type can be determined.
    """
    pass


class Variant:
    """
    Minimal concrete implementation of the :class:`~sfsutils.io_handlers.Site` interface: a duck-typed stand-in for a
    :class:`cyvcf2.Variant` exposing the subset of its interface that the parser, filtrations, annotations and
    stratifications rely on: ``CHROM``, ``POS``, ``REF``, ``ALT``, ``INFO``, the ``is_*`` type flags and the
    per-sample ``gt_bases``, alongside the numeric ``allele_indices`` these backends hold natively.
    Non-VCF backends (tree sequences, VCF-Zarr stores) emit these objects.
    """

    #: Whether the variant is an SNP
    is_snp: bool = False

    #: Whether the variant is an MNP
    is_mnp: bool = False

    #: Whether the variant is an indel
    is_indel: bool = False

    #: Whether the variant is a deletion
    is_deletion: bool = False

    #: Whether the variant is a structural variant
    is_sv: bool = False

    #: The per-haplotype allele indices, absent unless the backend supplies them
    allele_indices: Optional[np.ndarray] = None

    def __init__(
            self,
            ref: str,
            pos: int,
            chrom: str,
            gt_bases: Sequence[str] | np.ndarray | None = None,
            alt: Sequence[str] | None = None,
            is_snp: bool = False,
            is_mnp: bool = False,
            info: Dict[str, object] | None = None,
            allele_indices: np.ndarray | None = None,
            phased: bool | Sequence[bool] | np.ndarray | None = None,
    ):
        """
        Initialize the variant.

        :param ref: The reference allele.
        :param pos: The position.
        :param chrom: The contig.
        :param gt_bases: The per-sample genotype strings (e.g. ``"A/T"``), as for ``cyvcf2.Variant.gt_bases``.
        :param alt: The alternate alleles.
        :param is_snp: Whether the site is a single-nucleotide polymorphism.
        :param is_mnp: Whether the site is a multi-nucleotide polymorphism.
        :param info: The INFO field.
        :param allele_indices: The per-haplotype allele indices into ``[ref] + alt``, of shape
            ``(n_samples, ploidy)``, with ``-1`` for a missing call. Where ``gt_bases`` is omitted the
            genotype strings are assembled from these on demand.
        :param phased: Whether each sample's genotype is phased, either per sample or for the site as a
            whole, which decides the separator of the assembled genotype strings.
        """
        #: The reference allele
        self.REF: str = ref

        #: The position
        self.POS: int = int(pos)

        #: The contig
        self.CHROM: str = chrom

        #: The alternate alleles
        self.ALT: List[str] = list(alt) if alt is not None else []

        # supplied genotype strings shadow the cached property that would otherwise assemble them
        if gt_bases is not None:
            self.gt_bases = np.asarray(gt_bases)

        if allele_indices is not None:
            self.allele_indices = allele_indices

        #: Whether each sample's genotype is phased
        self._phased: bool | Sequence[bool] | np.ndarray | None = phased

        #: Whether the site is an SNP
        self.is_snp: bool = is_snp

        #: Whether the site is an MNP
        self.is_mnp: bool = is_mnp

        #: Info field
        self.INFO: Dict[str, object] = dict(info) if info else {}

    @cached_property
    def gt_bases(self) -> np.ndarray:
        """
        The per-sample genotype strings, assembled from the allele indices. Most consumers read the indices
        instead, so a backend that holds the calls numerically supplies those alone and the strings are
        built only where something actually asks for them.

        :return: The genotype strings, one per sample.
        """
        indices = self.allele_indices

        if indices is None:
            return np.array([], dtype=object)

        alleles = [self.REF] + list(self.ALT)
        n_alleles = len(alleles)

        if self._phased is None:
            separators = ['/'] * len(indices)
        elif np.ndim(self._phased) == 0:
            separators = ['|' if self._phased else '/'] * len(indices)
        else:
            separators = ['|' if p else '/' for p in self._phased]

        return np.array([
            separators[i].join(alleles[a] if 0 <= a < n_alleles else '.' for a in row)
            for i, row in enumerate(indices)
        ], dtype=object)


class DummyVariant(Variant):
    """
    Synthetic monomorphic reference site used by the :class:`~sfsutils.parser.TargetSiteCounter` to
    represent a sampled target site. It exposes the full :class:`Site` interface; every sample is
    homozygous for the reference allele. The per-sample ``gt_bases`` array is built lazily (there may be
    many such sites and most consumers never read the genotypes) via a cached property.
    """

    def __init__(self, ref: str, pos: int, chrom: str, n_samples: int = 0, ploidy: int = 2):
        """
        Initialize the dummy variant.

        :param ref: The reference allele.
        :param pos: The position.
        :param chrom: The contig.
        :param n_samples: The number of samples, so ``gt_bases`` aligns with the parser's sample masks
            (default ``0`` for a genotype-less placeholder used where the genotypes are not read).
        :param ploidy: The ploidy of each sample's genotype.
        """
        super().__init__(ref=ref, pos=pos, chrom=chrom)

        self._n_samples: int = int(n_samples)
        self._ploidy: int = int(ploidy)

    @cached_property
    def gt_bases(self) -> np.ndarray:
        """
        The per-sample genotypes: every sample homozygous for the reference allele.

        :return: The genotype strings, one per sample.
        """
        return np.array(['/'.join([self.REF] * self._ploidy)] * self._n_samples, dtype=object)

    @cached_property
    def allele_indices(self) -> np.ndarray:
        """
        The per-haplotype allele indices: every haplotype carries the reference allele.

        :return: The allele indices, of shape ``(n_samples, ploidy)``.
        """
        return np.zeros((self._n_samples, self._ploidy), dtype=int)


class VariantReader(Iterable, ABC):
    """
    Common streaming interface over a variant source. Concrete readers wrap a VCF-Zarr store or a tskit
    tree sequence and yield :class:`~sfsutils.io_handlers.Variant` objects in file order, so the parser can consume any input
    format through a single ``for variant in reader`` loop. Readers are re-iterable: each ``iter(reader)``
    starts a fresh pass over the source.
    """

    @property
    @abstractmethod
    def samples(self) -> List[str]:
        """
        The sample names, in genotype-column order.

        :return: The sample names.
        """
        pass

    @property
    @abstractmethod
    def seqnames(self) -> List[str]:
        """
        The contig (sequence) names present in the source, matching ``cyvcf2.VCF.seqnames``.

        :return: The contig names.
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Variant]:
        """
        Iterate over the sites of the source.

        :return: An iterator over variants.
        """
        pass

    @property
    def sequence_length(self) -> Optional[float]:
        """
        The total length (bp) of the source region, when the source knows it (as a tree sequence does).
        Used to estimate the site density when extrapolating monomorphic sites from a target-site count.
        Returns ``None`` when no reliable length is available (e.g. a VCF-Zarr store), in which case the
        observed variant span is used instead.

        :return: The sequence length, or ``None``.
        """
        return None

    def add_info_to_header(self, data: dict):
        """
        Declare an INFO field, registering it with the underlying VCF header. Does nothing for
        non-VCF sources, which have no header.

        :param data: The INFO field definition.
        """
        pass

    def count_sites(self) -> int:
        """
        Count the number of sites in the source.

        :return: The number of sites.
        """
        return sum(1 for _ in self)

    def close(self):
        """
        Release any resources held by the reader.
        """
        pass


class TskitVariantReader(VariantReader):
    """
    Stream variants from a tskit tree sequence (e.g. an inferred ARG or an msprime simulation). Sample
    haplotype nodes are grouped into diploid (or higher-ploidy) samples by individual, exactly as
    :meth:`tskit.TreeSequence.write_vcf` does, so parsing a ``.trees`` file yields the same spectrum as
    parsing the VCF written from it. tskit stores the ancestral state as allele ``0``, which becomes the
    reference allele, so the parser recovers the correct polarisation with ``skip_non_polarized=False``.
    """

    def __init__(self, ts: 'tskit.TreeSequence', contig: str = '1'):
        """
        Initialize the reader.

        :param ts: The tree sequence.
        :param contig: The contig name to report (tree sequences have no contig concept).
        """
        #: The tree sequence
        self._ts = ts

        #: The reported contig name
        self._contig = str(contig)

        sample_nodes = list(int(n) for n in ts.samples())

        #: Genotype-column index for each sample node
        self._node_col = {node: i for i, node in enumerate(sample_nodes)}

        sample_set = set(sample_nodes)

        # group sample nodes into VCF-style samples by individual (matching write_vcf); fall back to one
        # haploid sample per node when the tree sequence has no individuals
        individuals = [ind for ind in ts.individuals() if sample_set.intersection(int(n) for n in ind.nodes)]

        if individuals:
            self._sample_names = [f"tsk_{ind.id}" for ind in individuals]
            self._groups = [[int(n) for n in ind.nodes if int(n) in sample_set] for ind in individuals]
        else:
            self._sample_names = [f"tsk_{i}" for i in range(len(sample_nodes))]
            self._groups = [[node] for node in sample_nodes]

        ploidy = max((len(group) for group in self._groups), default=0)

        #: Genotype column of each sample's haplotypes, of shape ``(n_samples, ploidy)``. A sample of
        #: lower ploidy than the site's widest is padded with ``-1``, which reads back as a missing call
        #: and so contributes nothing, exactly as its shorter genotype string does
        self._cols = np.full((len(self._groups), ploidy), -1, dtype=np.intp)

        for i, group in enumerate(self._groups):
            self._cols[i, :len(group)] = [self._node_col[node] for node in group]

        #: Whether the samples differ in ploidy
        self._ragged: bool = any(len(group) != ploidy for group in self._groups)

    @property
    def samples(self) -> List[str]:
        """
        The sample names.

        :return: The sample names.
        """
        return list(self._sample_names)

    @property
    def seqnames(self) -> List[str]:
        """
        The contig name (tree sequences have a single synthetic contig).

        :return: The contig names.
        """
        return [self._contig]

    @property
    def tree_sequence(self) -> 'tskit.TreeSequence':
        """
        The underlying tree sequence, used by :class:`TskitVariantWriter` to write a site-subset ``.trees``.

        :return: The tree sequence.
        """
        return self._ts

    @property
    def sequence_length(self) -> float:
        """
        The length of the tree-sequence genome, i.e. the region over which sites are distributed. This is the
        true extent even when the observed (polymorphic) sites do not reach the ends.

        :return: The sequence length.
        """
        return float(self._ts.sequence_length)

    def count_sites(self) -> int:
        """
        The number of sites in the tree sequence.

        :return: The number of sites.
        """
        return int(self._ts.num_sites)

    def __iter__(self) -> Iterator[Variant]:
        """
        Iterate over the sites of the tree sequence.

        :return: An iterator over variants.
        """
        # the tskit Site is materialised afresh on every access, decoding the metadata of each of its
        # mutations, so the positions are taken from the table column instead
        positions = self._ts.sites_position

        for var in self._ts.variants():
            alleles = var.alleles
            genotypes = var.genotypes

            is_snp = _is_snp(alleles[0], [a for a in alleles[1:] if a])

            codes = np.asarray(genotypes)[self._cols]

            if all(alleles):
                # the tskit numbering is already that of ``[REF] + ALT`` and its missing sentinel is the
                # one the allele indices use, so the calls carry over as they are
                allele_indices = codes
            else:
                # an empty allele is dropped from ALT, so the tskit allele numbering is re-indexed onto
                # ``[REF] + ALT``; a dropped allele reads back as a missing call, as its genotype string does
                remap = np.full(len(alleles) + 1, -1, dtype=np.intp)
                remap[0] = 0
                next_index = 1
                for j, allele in enumerate(alleles[1:], start=1):
                    if allele:
                        remap[j] = next_index
                        next_index += 1

                allele_indices = remap[np.where((codes >= 0) & (codes < len(alleles)), codes, len(alleles))]

            if self._ragged:
                # a sample of lower ploidy than the widest has no call in its trailing columns, which the
                # fill sentinel marks apart from a call that is genuinely missing
                allele_indices = np.where(self._cols >= 0, allele_indices, -2)

            # a sample of lower ploidy than the widest carries a padding column, which would show up as an
            # extra missing allele in the assembled genotype string, so those are written out here instead
            gt_bases = np.array([
                "|".join(
                    alleles[genotypes[self._node_col[node]]]
                    if genotypes[self._node_col[node]] >= 0 and alleles[genotypes[self._node_col[node]]]
                    else "."
                    for node in group
                )
                for group in self._groups
            ], dtype=object) if self._ragged else None

            position = float(positions[var.index])

            variant = Variant(
                ref=alleles[0],
                pos=int(position) + 1,  # tskit positions are 0-based, VCF POS is 1-based
                chrom=self._contig,
                gt_bases=gt_bases,
                alt=[a for a in alleles[1:] if a],
                is_snp=is_snp,
                allele_indices=allele_indices,
                # tree-sequence haplotypes within an individual are ordered, hence phased, as in write_vcf
                phased=True,
            )

            # carry the exact (possibly non-integer) tskit position so TskitVariantWriter can identify the
            # site without relying on the lossy integer POS, which collides on continuous-genome sequences
            variant._tskit_position = position

            yield variant


class ZarrVariantReader(VariantReader):
    """
    Stream variants from a VCF-Zarr store in the `vcf2zarr <https://sgkit-dev.github.io/bio2zarr>`_ (VCZ)
    layout. Genotypes are read in variant chunks to bound memory. An ancestral-allele INFO field, if
    encoded as a ``variant_<tag>`` array, is surfaced under ``INFO`` so the usual polarisation logic
    applies.
    """

    #: The number of variants read per batch where the store does not declare its own chunk grid
    _default_chunk_size: int = 1000

    def __init__(self, path: str, info_ancestral: str = 'AA', chunk_size: Optional[int] = None):
        """
        Initialize the reader.

        :param path: The path to the VCF-Zarr store.
        :param info_ancestral: The INFO tag holding the ancestral allele.
        :param chunk_size: The number of variants read per batch, or ``None`` to follow the store's own
            chunking along the ``variants`` axis.
        """
        try:
            import zarr
        except ImportError:
            raise ImportError(
                "VCF-Zarr support in sfsutils requires the optional 'zarr' package. "
                "Please install sfsutils with the 'zarr' extra: pip install sfsutils[zarr]"
            )

        #: The Zarr store root
        self._root = zarr.open(path, mode='r')

        #: The INFO tag holding the ancestral allele
        self._info_ancestral = info_ancestral

        #: The number of variants read per batch
        self._chunk_size = int(chunk_size) if chunk_size else self._store_chunk_size()

        #: The sample names
        self._sample_ids = [self._decode(s) for s in self._root['sample_id'][:]]

        #: The contig names
        self._contig_ids = [self._decode(c) for c in self._root['contig_id'][:]]

    def _store_chunk_size(self) -> int:
        """
        The store's own chunk length along the ``variants`` axis, which every variant and call array
        shares. A batch that straddles the stored chunks makes each of them be fetched and decompressed
        once per batch overlapping it, so the batch follows the grid rather than a fixed length.

        :return: The number of variants to read per batch.
        """
        for name in ('call_genotype', 'variant_position'):
            if name in self._root:
                chunks = getattr(self._root[name], 'chunks', None)

                if chunks:
                    return max(1, int(chunks[0]))

        return self._default_chunk_size

    @staticmethod
    def _decode(value) -> str:
        """
        Decode a Zarr string scalar (bytes or str) to ``str``.

        :param value: The value.
        :return: The decoded string.
        """
        return value.decode() if isinstance(value, bytes) else str(value)

    #: The allele entries a VCF-Zarr store uses to pad a site to the widest allele count, and which
    #: tskit2zarr also writes for a mutation to the empty allele
    _empty_alleles = frozenset(('', '.'))

    @classmethod
    def _info_scalar(cls, value, kind: str):
        """
        One entry of a ``variant_<key>`` array as an INFO value, with the types cyvcf2 surfaces
        (str/float/int/bool) and the store's missing markers reported as absent.

        :param value: The stored entry.
        :param kind: The dtype kind of the array it comes from.
        :return: The value, or ``None`` where the store marks the entry as absent.
        """
        if kind in ('U', 'S', 'O', 'T'):
            decoded = cls._decode(value)

            return None if decoded in cls._empty_alleles else decoded

        if kind == 'f':
            # NaN (a plain NaN or the VCF-Zarr missing sentinel) means an absent value
            return None if np.isnan(value) else float(value)

        if kind == 'b':
            # a bool array encodes a VCF Flag: surface it only when set, as cyvcf2 does (an absent flag
            # is stored False and must not read back as present-False)
            return True if bool(value) else None

        # the VCF-Zarr missing (-1) and fill (-2) sentinels of an integer array, which every writer
        # emits for a field a site does not carry
        entry = int(value)

        return None if entry in (-1, -2) else entry

    @classmethod
    def _info_row(cls, values, kind: str):
        """
        One row of a two-dimensional ``variant_<key>`` array as an INFO value. A field of ``Number != 1``
        is stored one value per column, padded to the widest site, which is collapsed back to the form
        cyvcf2 hands out: the comma-separated string for a string field, the scalar or the tuple of
        values for a numeric one.

        :param values: The stored row.
        :param kind: The dtype kind of the array it comes from.
        :return: The value, or ``None`` where the site carries no entry at all.
        """
        entries = [entry for entry in (cls._info_scalar(v, kind) for v in values) if entry is not None]

        if not entries:
            return None

        if kind in ('U', 'S', 'O', 'T'):
            return ','.join(entries)

        return entries[0] if len(entries) == 1 else tuple(entries)

    @classmethod
    def _alleles(cls, stored, rows: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """
        The allele strings of one site alongside its calls. An empty entry is either the padding a site
        with fewer alleles than the widest carries or, from a tree sequence, a mutation to the empty
        allele; both are dropped from ``[REF] + ALT``, so the calls are re-indexed onto the alleles that
        remain and a call of a dropped allele reads back as missing, exactly as the tree sequence itself
        streams it.

        :param stored: The stored allele entries of the site.
        :param rows: The stored allele indices of the site.
        :return: The allele strings and the re-indexed allele indices.
        """
        alleles = [cls._decode(x) for x in stored]

        if not any(a in cls._empty_alleles for a in alleles):
            return alleles, rows

        remap = np.full(len(alleles) + 1, -1, dtype=np.intp)
        remap[0] = 0
        kept = [alleles[0]] if alleles else []

        for j, allele in enumerate(alleles[1:], start=1):
            if allele not in cls._empty_alleles:
                remap[j] = len(kept)
                kept.append(allele)

        remapped = remap[np.where((rows >= 0) & (rows < len(alleles)), rows, len(alleles))]

        # the fill of a sample below the widest ploidy is distinct from a call that is missing, so it
        # survives the re-indexing rather than collapsing onto -1
        return kept, np.where(rows == -2, -2, remapped)

    @property
    def samples(self) -> List[str]:
        """
        The sample names.

        :return: The sample names.
        """
        return list(self._sample_ids)

    @property
    def seqnames(self) -> List[str]:
        """
        The contig names declared in the store.

        :return: The contig names.
        """
        return list(self._contig_ids)

    def count_sites(self) -> int:
        """
        The number of sites in the store.

        :return: The number of sites.
        """
        return int(self._root['variant_position'].shape[0])

    def __iter__(self) -> Iterator[Variant]:
        """
        Iterate over the sites of the store.

        :return: An iterator over variants.
        """
        root = self._root
        position = root['variant_position']
        allele = root['variant_allele']
        contig = root['variant_contig']
        arrays = list(root.array_keys())
        # a store written from a sample-less VCF carries no call arrays at all, and streams as a series of
        # sites without genotypes, exactly as the same VCF does
        genotype = root['call_genotype'] if 'call_genotype' in arrays else None
        phased = root['call_genotype_phased'] if 'call_genotype_phased' in arrays else None
        # surface every INFO field the writer persisted as a variant_<key> string array (the ancestral
        # tag, but also e.g. an annotated Degeneracy/Synonymy), skipping the VCF fixed columns that
        # vcf2zarr stores as reserved variant_* arrays (CHROM/POS/ID/REF+ALT/QUAL/FILTER and their
        # length/mask companions); otherwise a store re-parsed stratified by an annotated field would
        # see no INFO, while a plain vcf2zarr store would fabricate INFO from its reserved metadata
        reserved_arrays = {'variant_position', 'variant_contig', 'variant_allele', 'variant_id',
                           'variant_id_mask', 'variant_quality', 'variant_filter', 'variant_length'}
        # a field of Number != 1 (AC/DP4, but also the VEP CSQ and SnpEff ANN of a multi-transcript
        # annotation) is stored as a 2-D variant_<key> array, one column per value, and is surfaced in
        # the comma-separated form cyvcf2 hands out
        info_arrays = {k[len('variant_'):]: root[k]
                       for k in root.array_keys()
                       if k.startswith('variant_') and k not in reserved_arrays and root[k].ndim in (1, 2)}

        n = position.shape[0]

        for start in range(0, n, self._chunk_size):
            end = min(start + self._chunk_size, n)

            pos_batch = position[start:end]
            allele_batch = allele[start:end]
            contig_batch = contig[start:end]
            gt_batch = np.asarray(genotype[start:end]) if genotype is not None else None
            phased_batch = np.asarray(phased[start:end]) if phased is not None else None
            info_batches = {key: arr[start:end] for key, arr in info_arrays.items()}

            for i in range(end - start):
                rows = gt_batch[i] if gt_batch is not None else np.zeros((0, 0), dtype=np.int8)
                seps = phased_batch[i] if phased_batch is not None else None

                site_alleles, rows = self._alleles(allele_batch[i], rows)

                is_snp = _is_snp(site_alleles[0] if site_alleles else '', site_alleles[1:])

                # surface INFO with native types matching cyvcf2 (float/int/bool/str), so a numeric field
                # stored typed (by vcf2zarr, or by our own writer) is not silently a string
                info = {}
                for key, batch in info_batches.items():
                    kind = batch.dtype.kind
                    value = (self._info_scalar(batch[i], kind) if batch.ndim == 1
                             else self._info_row(batch[i], kind))

                    if value is not None:
                        info[key] = value

                yield Variant(
                    ref=site_alleles[0] if site_alleles else '.',
                    pos=int(pos_batch[i]),  # vcf2zarr stores the 1-based VCF position
                    chrom=self._contig_ids[int(contig_batch[i])],
                    alt=site_alleles[1:],
                    is_snp=is_snp,
                    info=info,
                    allele_indices=rows,
                    phased=seps,
                )


class VariantWriter(ABC):
    """
    Abstract writer mirroring :class:`~sfsutils.io_handlers.VariantReader`: it consumes the same streamed
    :class:`~sfsutils.io_handlers.Variant` interface and writes it to a concrete on-disk format. The output
    format follows the output file's extension (see :meth:`VariantWriter.open`).
    """

    @staticmethod
    def open(
            output: str,
            reader: Union['cyvcf2.VCF', VariantReader],
            info_ancestral: str = 'AA'
    ) -> 'VariantWriter':
        """
        Open the variant writer matching the output file's extension: a VCF-Zarr store for ``.vcz``/``.zarr`` (from
        any input), a tskit tree sequence for ``.trees`` (only when the input is itself a tree sequence, since a
        genealogy cannot be reconstructed from genotype data), and a VCF otherwise.

        :param output: The output path; its extension selects the format.
        :param reader: The open input reader (a cyvcf2 VCF or a :class:`~sfsutils.io_handlers.VariantReader`).
        :param info_ancestral: The INFO tag holding the ancestral allele, for the VCF-Zarr writer.
        :return: The writer.
        :raises ValueError: If a ``.trees`` output is requested from a non-tree-sequence input.
        """
        fmt = _output_format(output)

        if fmt == 'zarr':
            return ZarrVariantWriter(output, samples=list(reader.samples), seqnames=list(reader.seqnames),
                                     info_ancestral=info_ancestral)

        if fmt == 'tskit':
            if not isinstance(reader, TskitVariantReader):
                raise ValueError(
                    "Writing a tree sequence (.trees) is only supported when the input is itself a tree sequence: "
                    "a genealogy cannot be reconstructed from genotype data without ARG inference. Use a .vcz/.zarr "
                    "or VCF output instead."
                )

            return TskitVariantWriter(reader.tree_sequence, output)

        return VCFVariantWriter(output, reader)

    def write(self, variant: Site) -> None:
        """
        Write a single variant.

        :param variant: The variant to write.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Flush and release any resources held by the writer.
        """
        pass


class VCFVariantWriter(VariantWriter):
    """
    Write variants to a VCF (optionally gzipped) via cyvcf2, copying the header from the input VCF. Because it
    reuses the input's cyvcf2 header, this writer requires a VCF input; writing VCF output from a tree sequence
    or VCF-Zarr store is not supported (use a ``.vcz``/``.zarr`` output instead).
    """

    def __init__(self, output: str, template: 'cyvcf2.VCF'):
        """
        Open the writer.

        :param output: The output VCF path.
        :param template: The input cyvcf2 VCF whose header is copied.
        :raises ValueError: If the input is not a VCF (no cyvcf2 header to copy).
        """
        try:
            from cyvcf2 import Writer
        except ImportError:
            raise ImportError(
                "VCF support in sfsutils requires the optional 'cyvcf2' package. "
                "Please install sfsutils with the 'vcf' extra: pip install sfsutils[vcf]"
            )

        if isinstance(template, VariantReader):
            raise ValueError(
                "Writing VCF output is only supported from a VCF input, since the header is copied from it. "
                "To write from a tree sequence or VCF-Zarr store, use a .vcz/.zarr output instead."
            )

        self._writer = Writer(output, template)

    def write(self, variant: 'cyvcf2.Variant') -> None:
        """
        Write a single record.

        :param variant: The cyvcf2 variant to write.
        """
        self._writer.write_record(variant)

    def close(self) -> None:
        """
        Close the underlying writer.
        """
        self._writer.close()


class ZarrVariantWriter(VariantWriter):
    """
    Write variants to a VCF-Zarr store in the `vcf2zarr <https://sgkit-dev.github.io/bio2zarr>`_ (VCZ) layout
    read back by :class:`~sfsutils.io_handlers.ZarrVariantReader`, so the output can come from any input (VCF, tree sequence or
    another VCF-Zarr store). Any INFO fields present on the variants (for example an annotated ancestral
    allele) are persisted as ``variant_<tag>`` arrays.

    Positions, genotypes, alleles and INFO values accumulate in chunk-sized buffers and each complete
    chunk is flushed to the store, so the memory held is one chunk rather than the whole input. The two
    ragged axes (the ploidy and the allele width) are only known once every variant has been seen, so the
    arrays are created from the first chunk and rebuilt on the rare site that widens them. The same holds
    of the type of an INFO field, which is taken from the chunk it first appears in and rewritten on the
    rare chunk that widens it (an integer field first seen without a value, a numeric one that turns out
    to carry a string).
    """

    #: Number of variants per chunk along the ``variants`` axis, as written by vcf2zarr
    _variant_chunk: int = 10000

    #: Number of samples per chunk along the ``samples`` axis, as written by vcf2zarr
    _sample_chunk: int = 1000

    #: The coordinate, allele and genotype datasets an INFO field must not collide with
    _reserved = frozenset({'variant_position', 'variant_contig', 'variant_allele',
                           'call_genotype', 'call_genotype_phased', 'sample_id', 'contig_id'})

    #: Marker for an INFO field a variant does not carry
    _missing = object()

    #: The genotype sentinel of a haplotype a call does not reach, which the VCF-Zarr spec separates from
    #: the ``-1`` of a call that is present but missing, so a shorter call exports at its own ploidy
    _fill: int = -2

    #: The VCF-Zarr missing-float sentinel (a specific NaN bit pattern), which vcztools/sgkit emit as a
    #: missing INFO value; a plain NaN would be exported as the literal token 'nan'
    _float_missing: float = np.array([0x7FF0000000000001], dtype=np.uint64).view(np.float64)[0]

    #: How an INFO field is encoded, from the narrowest to the widest. A field is written in the
    #: encoding of the chunk it first appears in and promoted where a later chunk does not fit it.
    _info_dtypes: Dict[str, Any] = {'bool': bool, 'int': np.int64, 'float': np.float64, 'str': str}

    def __init__(self, output: str, samples: List[str], seqnames: List[str], info_ancestral: str = 'AA'):
        """
        Open the writer.

        :param output: The output store path (``.vcz`` or ``.zarr``).
        :param samples: The sample names.
        :param seqnames: The contig names.
        :param info_ancestral: The INFO tag holding the ancestral allele (written as ``variant_<tag>``).
        """
        try:
            import zarr  # noqa: F401
        except ImportError:
            raise ImportError(
                "VCF-Zarr support in sfsutils requires the optional 'zarr' package. "
                "Please install sfsutils with the 'zarr' extra: pip install sfsutils[zarr]"
            )

        self._output = output
        self._logger = logger.getChild(self.__class__.__name__)

        #: Whether the store has been finalised, so that closing again is a no-op
        self._closed: bool = False
        self._samples = list(samples)
        self._contig_ids = list(seqnames)
        self._contig_index = {c: i for i, c in enumerate(self._contig_ids)}
        self._info_ancestral = info_ancestral

        # per-contig lengths (the last position seen on each), so a ##contig header can be emitted
        self._contig_length: List[int] = [1] * len(self._contig_ids)

        # the INFO values of the variants of the current chunk, one list per field, padded with the
        # missing marker where a variant does not carry the field
        self._info: Dict[str, List] = {}

        # the encoding each field has been written in so far, so a chunk that does not fit it promotes it
        self._info_kind: Dict[str, str] = {}

        # whether every value a field has carried so far was an integer, which a numeric array no longer
        # says once it holds them and which decides how they are spelled if the field turns out to be one
        self._info_integral: Dict[str, bool] = {}
        self._skipped: Set[str] = set()

        self._root = None
        self._n: int = 0
        self._flushed: int = 0
        self._row: int = 0
        self._ploidy: Optional[int] = None
        self._max_alleles: Optional[int] = None

        self._buf_position: Optional[np.ndarray] = None
        self._buf_contig: Optional[np.ndarray] = None
        self._buf_genotype: Optional[np.ndarray] = None
        self._buf_phased: Optional[np.ndarray] = None
        self._buf_allele: Optional[np.ndarray] = None

    @property
    def _axis_ploidy(self) -> int:
        """
        The extent of the ploidy axis: the widest call seen, or two where no variant carries any call.

        :return: The ploidy.
        """
        return self._ploidy if self._ploidy else 2

    @property
    def _axis_alleles(self) -> int:
        """
        The extent of the allele axis: the most alleles seen at any variant.

        :return: The allele count.
        """
        return self._max_alleles if self._max_alleles else 1

    @property
    def _gt_dtype(self) -> np.dtype:
        """
        The genotype dtype, sized to the data so many-allele sites do not wrap.

        :return: The dtype.
        """
        return np.dtype(np.int8 if self._axis_alleles <= np.iinfo(np.int8).max else np.int32)

    def write(self, variant: Site) -> None:
        """
        Buffer a single variant, flushing the current chunk to the store once it is full.

        :param variant: The variant to write.
        """
        alleles = [variant.REF] + list(variant.ALT)

        indices, phased = self._calls(variant, alleles)

        chrom = variant.CHROM
        contig = self._contig_index.get(chrom)
        if contig is None:
            contig = self._contig_index[chrom] = len(self._contig_ids)
            self._contig_ids.append(chrom)
            self._contig_length.append(1)

        pos = int(variant.POS)
        if pos > self._contig_length[contig]:
            self._contig_length[contig] = pos

        n_calls, width = indices.shape
        self._reserve(width, len(alleles))

        row = self._row

        # the buffers are reused across chunks, so the row is cleared before it is filled
        self._buf_position[row] = pos
        self._buf_contig[row] = contig
        self._buf_allele[row] = ''
        self._buf_allele[row, :len(alleles)] = alleles
        self._buf_genotype[row] = self._fill
        self._buf_genotype[row, :n_calls, :width] = indices
        self._buf_phased[row] = False
        self._buf_phased[row, :n_calls] = phased

        self._record_info(variant)

        self._n += 1
        self._row += 1

        if self._row == self._variant_chunk:
            self._flush()

    def _calls(self, variant: Site, alleles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        The numeric calls of a variant: the per-haplotype allele indices of shape ``(n_samples, ploidy)``
        alongside the per-sample phase. Every backend carries the calls numerically, so they are taken as
        they are; assembling and splitting the genotype strings would both cost the bulk of the write and
        lose the calls those strings cannot express, such as a haploid call of a third allele or any call
        of ploidy above two.

        :param variant: The variant.
        :param alleles: The site's allele strings, the reference first.
        :return: The allele indices and the per-sample phase.
        """
        indices = getattr(variant, 'allele_indices', None)

        if indices is not None:
            return np.atleast_2d(np.asarray(indices)), self._phases(getattr(variant, '_phased', None),
                                                                    len(indices))

        # cyvcf2 holds the calls on its Genotypes object, whose last column carries the phase flag
        genotype = getattr(variant, 'genotype', None)
        array = np.asarray(genotype.array()) if genotype is not None else None

        if array is not None and array.ndim == 2 and array.shape[1] >= 2:
            return array[:, :-1], array[:, -1].astype(bool)

        return self._split(variant, alleles)

    @staticmethod
    def _phases(phased, n_samples: int) -> np.ndarray:
        """
        The per-sample phase of a variant that declares it either per sample or for the site as a whole.

        :param phased: The phase declaration.
        :param n_samples: The number of samples.
        :return: The per-sample phase.
        """
        if phased is None:
            return np.zeros(n_samples, dtype=bool)

        if np.ndim(phased) == 0:
            return np.full(n_samples, bool(phased), dtype=bool)

        return np.asarray(phased, dtype=bool)

    @staticmethod
    def _split(variant: Site, alleles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        The numeric calls of a variant that carries its genotypes as strings alone.

        :param variant: The variant.
        :param alleles: The site's allele strings, the reference first.
        :return: The allele indices and the per-sample phase.
        """
        index = {a: i for i, a in enumerate(alleles)}

        # the same genotype string recurs across most samples, so each distinct one is split and looked
        # up once and the samples only carry a row index into the table of distinct calls
        table: List[List[int]] = []
        phased: List[bool] = []
        codes: Dict[str, int] = {}
        gt_bases = np.asarray(variant.gt_bases)
        rows = np.empty(len(gt_bases), dtype=np.intp)

        for i, gt in enumerate(gt_bases):
            gt = str(gt)
            code = codes.get(gt)
            if code is None:
                code = codes[gt] = len(table)
                table.append([index.get(c, -1) for c in gt.replace('|', '/').split('/')])
                phased.append('|' in gt)
            rows[i] = code

        width = max((len(call) for call in table), default=0)
        calls = np.full((len(table), width), ZarrVariantWriter._fill, dtype=np.intp)

        for i, call in enumerate(table):
            calls[i, :len(call)] = call

        return calls[rows], np.asarray(phased, dtype=bool)[rows]

    def _record_info(self, variant: Site) -> None:
        """
        Buffer the INFO values of a variant, keeping one list per field aligned with the rows of the
        chunk being filled.

        :param variant: The variant whose INFO fields to record.
        """
        info = dict(variant.INFO) if getattr(variant, 'INFO', None) else {}

        for key, value in info.items():
            values = self._info.get(key)

            if values is None:
                if f'variant_{key}' in self._reserved:
                    self._skipped.add(key)
                    continue

                # a field may first appear at any variant, so the earlier rows of the chunk are marked as
                # missing (the chunks already flushed are backfilled when the array is created)
                values = self._info[key] = [self._missing] * self._row

            # cyvcf2 returns a tuple for a Number=A/R/G/. field; write the comma-separated form the VCF
            # uses, so the value stays parseable instead of becoming a Python repr such as "(12, 3)"
            if isinstance(value, (tuple, list, np.ndarray)):
                value = ','.join(str(x) for x in value)

            values.append(value)

        for values in self._info.values():
            if len(values) == self._row:
                values.append(self._missing)

    def _reserve(self, ploidy: int, alleles: int) -> None:
        """
        Make room in the buffers for a variant of the given ploidy and allele count, allocating them on
        the first variant and widening them where a ragged axis grows.

        :param ploidy: The widest call of the variant.
        :param alleles: The number of alleles the variant carries.
        """
        self._ploidy = ploidy if self._ploidy is None else max(self._ploidy, ploidy)
        self._max_alleles = alleles if self._max_alleles is None else max(self._max_alleles, alleles)

        n_samples = len(self._samples)

        if self._buf_position is None:
            self._buf_position = np.empty(self._variant_chunk, dtype=np.int64)
            self._buf_contig = np.empty(self._variant_chunk, dtype=np.int64)
            self._buf_phased = np.zeros((self._variant_chunk, n_samples), dtype=bool)
            self._buf_genotype = np.full((self._variant_chunk, n_samples, self._axis_ploidy), self._fill,
                                         dtype=self._gt_dtype)
            self._buf_allele = np.full((self._variant_chunk, self._axis_alleles), '', dtype=object)
            return

        if self._buf_genotype.shape[2] != self._axis_ploidy or self._buf_genotype.dtype != self._gt_dtype:
            widened = np.full((self._variant_chunk, n_samples, self._axis_ploidy), self._fill,
                              dtype=self._gt_dtype)

            # the axis narrows where the variants so far carry no call at all and the default ploidy of
            # two gives way to the first call seen, whose columns beyond it hold nothing but the fill
            width = min(self._buf_genotype.shape[2], self._axis_ploidy)
            widened[:self._row, :, :width] = self._buf_genotype[:self._row, :, :width]
            self._buf_genotype = widened

        if self._buf_allele.shape[1] != self._axis_alleles:
            widened = np.full((self._variant_chunk, self._axis_alleles), '', dtype=object)
            widened[:self._row, :self._buf_allele.shape[1]] = self._buf_allele[:self._row]
            self._buf_allele = widened

    def _chunks(self, shape: Tuple[int, ...], dimensions: List[str]) -> Tuple[int, ...]:
        """
        The chunk grid of an array. The spec requires a single chunk size along ``variants`` across all
        variant and call arrays, and a single one along ``samples`` across the call arrays, so readers
        such as vcztools can align the chunk grids. Zarr's own auto-chunking would pick a different grid
        per array and leave the store readable only by us.

        :param shape: The final shape of the array.
        :param dimensions: The names of its axes.
        :return: The chunk shape.
        """
        sizes = {'variants': self._variant_chunk, 'samples': self._sample_chunk}

        # any remaining axis (ploidy, alleles, contigs) stays in a single chunk, and an empty axis
        # still needs a positive chunk length
        return tuple(max(1, min(sizes.get(dim, extent), extent)) for dim, extent in zip(dimensions, shape))

    def _create(self, name: str, shape: Tuple[int, ...], dtype, dimensions: List[str], extent=None):
        """
        Create an array in the store. Every array carries an ``_ARRAY_DIMENSIONS`` attribute naming its
        axes, so the store is a spec-compliant VCF-Zarr readable by vcztools / sgkit and not only by our
        own reader. Variable-length strings use the native zarr-3 ``str`` dtype.

        :param name: The array name.
        :param shape: The shape to create it with.
        :param dtype: The dtype.
        :param dimensions: The names of its axes.
        :param extent: The final shape the chunk grid is clamped to, where it differs from ``shape``.
        :return: The created array.
        """
        array = self._root.create_array(name, shape=shape, dtype=dtype,
                                        chunks=self._chunks(shape if extent is None else extent, dimensions))
        array.attrs['_ARRAY_DIMENSIONS'] = list(dimensions)

        return array

    def _write(self, name: str, data, dimensions: List[str]) -> None:
        """
        Create an array and write it whole.

        :param name: The array name.
        :param data: The data.
        :param dimensions: The names of its axes.
        """
        data = np.asarray(data)
        array = self._create(name, data.shape, str if data.dtype == object else data.dtype, dimensions)
        array[...] = data

    @staticmethod
    def _str_data(values) -> np.ndarray:
        """
        Cast values to an object array of strings, which is written as a variable-length string array.

        :param values: The values.
        :return: The object array.
        """
        return np.asarray([str(v) for v in values], dtype=object)

    def _open(self) -> None:
        """
        Open the store and create the streamed arrays. A mid-stream flush only happens once the first
        chunk is complete, so clamping the grid to the variants seen so far gives the same result as
        clamping it to the final count.
        """
        import zarr

        self._root = zarr.open(self._output, mode='w')

        n_samples = len(self._samples)
        genotype = (self._n, n_samples, self._axis_ploidy)

        self._create('variant_position', (0,), np.int64, ['variants'], extent=(self._n,))
        self._create('variant_contig', (0,), np.int64, ['variants'], extent=(self._n,))
        self._create('call_genotype', (0, n_samples, self._axis_ploidy), self._gt_dtype,
                     ['variants', 'samples', 'ploidy'], extent=genotype)
        self._create('call_genotype_phased', (0, n_samples), bool, ['variants', 'samples'],
                     extent=(self._n, n_samples))
        self._create('variant_allele', (0, self._axis_alleles), str, ['variants', 'alleles'],
                     extent=(self._n, self._axis_alleles))

    def _widen(self) -> None:
        """
        Grow the ragged axes of the arrays already in the store to their current extent, so the buffered
        chunk fits.
        """
        genotype = self._root['call_genotype']

        if genotype.shape[2] != self._axis_ploidy or genotype.dtype != self._gt_dtype:
            self._rebuild('call_genotype', (len(self._samples), self._axis_ploidy), self._gt_dtype,
                          self._fill, ['variants', 'samples', 'ploidy'])

        if self._root['variant_allele'].shape[1] != self._axis_alleles:
            self._rebuild('variant_allele', (self._axis_alleles,), str, '', ['variants', 'alleles'])

    def _rebuild(self, name: str, tail: Tuple[int, ...], dtype, fill, dimensions: List[str]) -> None:
        """
        Recreate an array with a wider trailing shape, carrying the data over a chunk at a time so only
        one chunk is held in memory. Zarr fixes the chunk grid at creation, so the widened axis needs a
        fresh array rather than a resize.

        :param name: The array name.
        :param tail: The shape after the ``variants`` axis.
        :param dtype: The dtype.
        :param fill: The value the widened columns take.
        :param dimensions: The names of its axes.
        """
        staging = f'{name}_staging'
        shape = (self._root[name].shape[0],) + tail

        self._copy(self._root[name], self._create(staging, shape, dtype, dimensions), fill)
        del self._root[name]

        self._copy(self._root[staging], self._create(name, shape, dtype, dimensions), fill)
        del self._root[staging]

    def _copy(self, source, target, fill) -> None:
        """
        Copy an array chunk by chunk into a target whose trailing axis is at least as wide.

        :param source: The array to read.
        :param target: The array to write.
        :param fill: The value the columns beyond the source's width take.
        """
        dtype = object if isinstance(fill, str) else target.dtype
        width = min(source.shape[-1], target.shape[-1])

        for start in range(0, source.shape[0], self._variant_chunk):
            stop = min(start + self._variant_chunk, source.shape[0])
            block = np.full((stop - start,) + target.shape[1:], fill, dtype=dtype)
            block[..., :width] = source[start:stop][..., :width]
            target[start:stop] = block

    def _flush(self) -> None:
        """
        Write the buffered chunk to the store, creating or widening the arrays as needed.
        """
        if self._root is None:
            self._open()
        else:
            self._widen()

        if self._row == 0:
            return

        start, stop = self._flushed, self._flushed + self._row

        for name, buffer in (('variant_position', self._buf_position),
                             ('variant_contig', self._buf_contig),
                             ('call_genotype', self._buf_genotype),
                             ('call_genotype_phased', self._buf_phased),
                             ('variant_allele', self._buf_allele)):
            array = self._root[name]
            array.resize((stop,) + array.shape[1:])
            array[start:stop] = buffer[:self._row]

        self._flush_info(start, stop)

        self._flushed = stop
        self._row = 0

    def close(self) -> None:
        """
        Write the remaining variants and finalise the store.
        """
        # the arrays are created as the store is finalised, so finalising twice would fail on them; a
        # teardown that runs again after an error must not turn that into a second, confusing exception
        if self._closed:
            return

        self._closed = True

        self._flush()

        self._write('sample_id', self._str_data(self._samples), ['samples'])
        self._write('contig_id', self._str_data(self._contig_ids), ['contigs'])
        self._write('contig_length', np.array(self._contig_length, dtype=np.int64), ['contigs'])

        for key in sorted(self._skipped):
            self._logger.warning(f"Skipping INFO field '{key}': it collides with the reserved VCF-Zarr "
                                 f"dataset 'variant_{key}'.")

        # mark the store as a spec-compliant VCF-Zarr so external readers (vcztools/sgkit) accept it
        from . import __version__
        self._root.attrs['vcf_zarr_version'] = '0.5'
        self._root.attrs['source'] = f'sfsutils-{__version__}'

    @staticmethod
    def _is_bool(value) -> bool:
        """
        Whether a value is a boolean, i.e. a VCF Flag.

        :param value: The value.
        :return: Whether it is a boolean.
        """
        return isinstance(value, (bool, np.bool_))

    @classmethod
    def _is_int(cls, value) -> bool:
        """
        Whether a value is an integer that an int64 array holds and that no reader mistakes for the
        VCF-Zarr missing or fill sentinel.

        :param value: The value.
        :return: Whether it is such an integer.
        """
        return (isinstance(value, (int, np.integer)) and not cls._is_bool(value)
                and -(2 ** 63) <= int(value) < 2 ** 63 and int(value) not in (-1, -2))

    @classmethod
    def _is_numeric(cls, value) -> bool:
        """
        Whether a value is one a float64 array holds without losing precision. An integer beyond the
        range of an int64 would, so it is written as a string instead.

        :param value: The value.
        :return: Whether it is such a number.
        """
        return (isinstance(value, (float, np.floating))
                or (isinstance(value, (int, np.integer)) and not cls._is_bool(value)
                    and -(2 ** 63) <= int(value) < 2 ** 63))

    def _absent(self, value) -> bool:
        """
        Whether a variant carries no value for a field: the value is unset, the empty string, or the VCF
        ``.`` missing marker (which annotations such as DegeneracyAnnotation write for sites the field
        does not apply to). cyvcf2 hands back ``None`` for a Number=1 numeric field whose value is ``.``.

        :param value: The value.
        :return: Whether the value is absent.
        """
        if value is self._missing or value is None:
            return True

        if isinstance(value, float) and np.isnan(value):
            return True

        return isinstance(value, str) and value in ('', '.')

    def _kind(self, values: List) -> str:
        """
        The narrowest encoding holding one chunk of the values of an INFO field.

        :param values: The values of the chunk.
        :return: The encoding, or ``empty`` where the chunk carries no value at all.
        """
        present = [v for v in values if not self._absent(v)]

        if not present:
            return 'empty'

        if all(self._is_bool(v) for v in present):
            return 'bool'

        # an integer array has no way of marking a missing value apart from the sentinels, so a field
        # that is absent anywhere in the chunk is carried as a float
        if all(self._is_int(v) for v in present) and len(present) == len(values):
            return 'int'

        if all(self._is_numeric(v) for v in present):
            return 'float'

        return 'str'

    @staticmethod
    def _widest(left: str, right: str) -> str:
        """
        The encoding holding the values of two encodings at once. A flag and a number have none in
        common, so both are written out as the strings the VCF spells them with.

        :param left: The one encoding.
        :param right: The other encoding.
        :return: The encoding holding both.
        """
        if left == 'empty' or left == right:
            return right

        if right == 'empty':
            return left

        return 'float' if {left, right} == {'int', 'float'} else 'str'

    def _encode(self, values: List, kind: str) -> np.ndarray:
        """
        One chunk of the values of an INFO field in its encoding, with the marker the encoding uses for
        an absent value.

        :param values: The values of the chunk.
        :param kind: The encoding.
        :return: The encoded chunk.
        """
        if kind == 'bool':
            # an absent flag is a flag that is not set
            return np.array([False if self._absent(v) else bool(v) for v in values], dtype=bool)

        if kind == 'int':
            return np.array([int(v) for v in values], dtype=np.int64)

        if kind == 'float':
            return np.array([self._float_missing if self._absent(v) else float(v) for v in values],
                            dtype=np.float64)

        return self._str_data(['' if self._absent(v) else str(v) for v in values])

    def _recode(self, data: np.ndarray, source: str, kind: str, integral: bool) -> np.ndarray:
        """
        One chunk of an INFO field already in the store, in a wider encoding.

        :param data: The stored chunk.
        :param source: The encoding it is stored in.
        :param kind: The encoding to carry it over to.
        :param integral: Whether every value the array holds reached it as an integer.
        :return: The re-encoded chunk.
        """
        if source == kind:
            return data

        if kind == 'float':
            return data.astype(np.float64)

        if source == 'float':
            # a field whose values all reached the store as integers is spelled without a fractional part
            # again, so that one mixing integers and strings reads back as the VCF wrote it
            return self._str_data(['' if np.isnan(v) else str(int(v)) if integral else str(float(v))
                                   for v in data])

        if source == 'int':
            return self._str_data([str(int(v)) for v in data])

        return self._str_data([str(bool(v)) for v in data])

    def _flush_info(self, start: int, stop: int) -> None:
        """
        Write the INFO values of the buffered chunk as ``variant_<tag>`` arrays, typed to match the
        values so a numeric field round-trips as a number rather than a string. A field first appearing
        in this chunk is created here, with the variants before it marked as carrying no value, and a
        field whose values no longer fit its encoding is rewritten in a wider one.

        :param start: The first row of the chunk.
        :param stop: The row past the chunk.
        """
        for key, values in self._info.items():
            name = f'variant_{key}'

            # a field the last variants of the chunk do not carry is missing on them
            values.extend([self._missing] * (self._row - len(values)))

            kind = self._widest(self._info_kind.get(key, 'empty'), self._kind(values))
            integral = self._info_integral.get(key, True) and all(
                isinstance(v, (int, np.integer)) and not self._is_bool(v)
                for v in values if not self._absent(v))

            if name not in self._root:
                # the variants before the field first appears carry no value, and an integer array cannot
                # say so, so a field appearing late is numeric rather than integral
                if start and kind == 'int':
                    kind = 'float'

                self._create(name, (start,), self._info_dtypes[kind], ['variants'], extent=(self._n,))
                self._root[name][:] = self._encode([self._missing] * start, kind)
            elif kind != self._info_kind[key]:
                self._rebuild_info(name, self._info_kind[key], kind, self._info_integral[key])

            self._info_kind[key] = kind
            self._info_integral[key] = integral

            array = self._root[name]
            array.resize((stop,))
            array[start:stop] = self._encode(values[:self._row], kind)

            values.clear()

    def _rebuild_info(self, name: str, source: str, kind: str, integral: bool) -> None:
        """
        Rewrite an INFO array in a wider encoding, a chunk at a time so only one chunk is held in memory.
        Zarr fixes the dtype at creation, so the wider encoding needs a fresh array rather than a cast.

        :param name: The array name.
        :param source: The encoding the array is stored in.
        :param kind: The encoding to rewrite it in.
        :param integral: Whether every value the array holds reached it as an integer.
        """
        staging = f'{name}_staging'
        shape = self._root[name].shape

        self._recast(self._root[name], self._create(staging, shape, self._info_dtypes[kind], ['variants']),
                     source, kind, integral)
        del self._root[name]

        self._recast(self._root[staging], self._create(name, shape, self._info_dtypes[kind], ['variants']),
                     kind, kind, integral)
        del self._root[staging]

    def _recast(self, source, target, source_kind: str, kind: str, integral: bool) -> None:
        """
        Copy an INFO array chunk by chunk into a target of a wider encoding.

        :param source: The array to read.
        :param target: The array to write.
        :param source_kind: The encoding the source is stored in.
        :param kind: The encoding of the target.
        :param integral: Whether every value the source holds reached it as an integer.
        """
        for start in range(0, source.shape[0], self._variant_chunk):
            stop = min(start + self._variant_chunk, source.shape[0])
            target[start:stop] = self._recode(source[start:stop], source_kind, kind, integral)


class TskitVariantWriter(VariantWriter):
    """
    Write a site-subset of an input tree sequence to a ``.trees`` file. Only the sites whose variants are
    written (i.e. survive filtering) are kept, via :meth:`tskit.TreeSequence.delete_sites`, leaving the
    genealogy untouched. This is the only well-defined ``.trees`` output: a genealogy cannot be reconstructed
    from genotype data, so a tree-sequence output requires a tree-sequence input. INFO fields added by
    annotations are attached to the kept sites as JSON metadata on a best-effort basis.
    """

    def __init__(self, ts: 'tskit.TreeSequence', output: str):
        """
        Open the writer.

        :param ts: The source tree sequence.
        :param output: The output ``.trees`` path.
        """
        self._ts = ts
        self._output = output
        self._logger = logger.getChild(self.__class__.__name__)

        #: The exact tskit positions of the sites to keep.
        self._kept: set = set()

        #: INFO metadata keyed by exact tskit position, for kept sites.
        self._info_by_pos: Dict[float, Dict[str, object]] = {}

    def _position(self, variant: Site) -> float:
        """
        The exact tskit (0-based) position of a variant. :class:`TskitVariantReader` attaches it so that
        continuous-genome sequences, whose positions collide under the integer VCF ``POS``, are identified
        unambiguously; for any other source we fall back to reconstructing it from ``POS``.

        :param variant: The variant.
        :return: The exact 0-based position.
        """
        pos = getattr(variant, '_tskit_position', None)

        return float(pos) if pos is not None else float(int(variant.POS) - 1)

    def write(self, variant: Site) -> None:
        """
        Mark the variant's site as kept.

        :param variant: The variant to keep.
        """
        pos = self._position(variant)
        self._kept.add(pos)

        if getattr(variant, 'INFO', None):
            self._info_by_pos[pos] = dict(variant.INFO)

    def close(self) -> None:
        """
        Delete the dropped sites and dump the resulting tree sequence.
        """
        drop = [site.id for site in self._ts.sites() if site.position not in self._kept]

        sub = self._ts.delete_sites(drop)

        if self._info_by_pos:
            sub = self._attach_site_metadata(sub)

        sub.dump(self._output)

    def _attach_site_metadata(self, ts: 'tskit.TreeSequence') -> 'tskit.TreeSequence':
        """
        Attach the collected INFO fields to the kept sites as permissive-JSON metadata.

        :param ts: The site-subset tree sequence.
        :return: The tree sequence with site metadata, or the input unchanged if metadata could not be encoded.
        """
        import tskit

        try:
            tables = ts.dump_tables()
            sites = tables.sites.copy()
            tables.sites.clear()
            tables.sites.metadata_schema = tskit.MetadataSchema.permissive_json()

            for row in sites:
                info = self._info_by_pos.get(row.position, {})
                metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in info.items()}
                tables.sites.add_row(position=row.position, ancestral_state=row.ancestral_state, metadata=metadata)

            return tables.tree_sequence()
        except Exception as e:
            self._logger.warning(f"Could not attach INFO metadata to the tree sequence: {e}")
            return ts


def _output_format(output: str) -> str:
    """
    Infer the output format from a file name.

    :param output: The output path.
    :return: One of ``'zarr'``, ``'tskit'`` or ``'vcf'``.
    """
    lowered = output.rstrip('/').lower()

    if lowered.endswith(('.vcz', '.zarr')):
        return 'zarr'

    if lowered.endswith('.trees'):
        return 'tskit'

    return 'vcf'
