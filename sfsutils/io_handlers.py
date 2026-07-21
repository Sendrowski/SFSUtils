"""
Handles the reading of VCF, VCF-Zarr, tree-sequence, GFF and FASTA files.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-29"

import gzip
import hashlib
import logging
import os
import re
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from functools import cached_property
from typing import List, Iterable, Iterator, TextIO, Dict, Optional, Tuple, Union, Sequence, Protocol, \
    runtime_checkable
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


def is_monomorphic_snp(variant: Site) -> bool:
    """
    Whether the given variant is a monomorphic SNP.

    :param variant: The site
    :return: Whether the site is a monomorphic SNP
    """
    return (not (variant.is_snp or variant.is_mnp or variant.is_indel or variant.is_deletion or variant.is_sv)
            and not variant.ALT and variant.REF in bases)


def count_sites(
        vcf: str | Iterable['cyvcf2.Variant'],
        max_sites: int = np.inf,
        desc: str = 'Counting sites'
) -> int:
    """
    Count the number of sites in the input.

    :param vcf: The path to the input or an iterable of variants
    :param max_sites: Maximum number of sites to consider
    :param desc: Description for the progress bar
    :return: Number of sites
    """

    # if we don't have a file path, we can just count the number of variants
    if not isinstance(vcf, str):
        return len(list(vcf))

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
            suffix = os.path.splitext(file[:-3])[1] or '.tmp'
            fd, unzipped = tempfile.mkstemp(suffix=suffix)

            logger.info(f'Unzipping {file} to {unzipped}')

            with gzip.open(file, 'rb') as f_in:
                with os.fdopen(fd, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

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
        Get the contig from the FASTA file.

        Note that ``pyfaidx`` would be more efficient here, but there were problems when running it in parallel.

        :param aliases: The contig aliases.
        :param rewind: Whether to allow for rewinding the iterator if the contig is not found.
        :param notify: Whether to notify the user when rewinding the iterator.
        :return: The contig.
        """
        # if the contig is already loaded, we can just return it
        if self._contig is not None and self._contig.id in aliases:
            return self._contig

        # if the contig is not loaded, we can try to load it
        try:
            self._contig = next(self._ref)

            # iterate until we find the contig
            while self._contig.id not in aliases:
                self._contig = next(self._ref)

        except StopIteration:

            # if rewind is ``True``, we can rewind the iterator and try again
            if rewind:
                if notify:
                    self._logger.info("Rewinding FASTA iterator.")

                # renew fasta iterator
                FASTAHandler._rewind(self)

                return self.get_contig(aliases, rewind=False)

            raise LookupError(f'None of the contig aliases {aliases} were found in the FASTA file.')

        return self._contig

    def get_contig_names(self) -> List[str]:
        """
        Get the names of the contigs in the FASTA file.

        :return: The contig names.
        """
        return [contig.id for contig in self._ref]

    def _rewind(self):
        """
        Rewind the fasta iterator.
        """
        if hasattr(self, '_ref'):
            # noinspection all
            del self._ref


class GFFHandler(FileHandler):
    """
    GFF handler.
    """

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

        # load GFF file
        df = pd.read_csv(
            local_file,
            sep='\t',
            comment='#',
            names=col_labels,
            dtype=dtypes,
            usecols=['seqid', 'type', 'start', 'end', 'strand', 'phase']
        )

        # filter for coding sequences
        df = df[df['type'] == 'CDS']

        # drop rows with NA values
        df = df.dropna()

        # convert start and end to int
        df['start'] = df['start'].astype(int)
        df['end'] = df['end'].astype(int)

        # drop type column
        df.drop(columns=['type'], inplace=True)

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

            df['overlap'] = df['start'].shift(-1) <= df['end']

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
        if hasattr(self, '_reader'):
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
    per-sample ``gt_bases``. Non-VCF backends (tree sequences, VCF-Zarr stores) emit these objects.
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
        """
        #: The reference allele
        self.REF: str = ref

        #: The position
        self.POS: int = int(pos)

        #: The contig
        self.CHROM: str = chrom

        #: The alternate alleles
        self.ALT: List[str] = list(alt) if alt is not None else []

        #: The per-sample genotype strings
        self.gt_bases: np.ndarray = np.asarray(gt_bases) if gt_bases is not None else np.array([], dtype=object)

        #: Whether the site is an SNP
        self.is_snp: bool = is_snp

        #: Whether the site is an MNP
        self.is_mnp: bool = is_mnp

        #: Info field
        self.INFO: Dict[str, object] = dict(info) if info else {}


class DummyVariant(Variant):
    """
    Dummy variant class to emulate a mono-allelic site.
    """

    def __init__(self, ref: str, pos: int, chrom: str):
        """
        Initialize the dummy variant.

        :param ref: The reference allele
        :param pos: The position
        :param chrom: The contig
        """
        super().__init__(ref=ref, pos=pos, chrom=chrom)


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
        for var in self._ts.variants():
            alleles = var.alleles
            genotypes = var.genotypes

            gt_bases = np.array([
                # tree-sequence haplotypes within an individual are ordered, hence phased ('|'), as in write_vcf
                "|".join(
                    alleles[genotypes[self._node_col[node]]]
                    if genotypes[self._node_col[node]] >= 0 and alleles[genotypes[self._node_col[node]]]
                    else "."
                    for node in group
                )
                for group in self._groups
            ], dtype=object)

            observed = [a for a in alleles if a]
            is_snp = len(observed) >= 2 and all(len(a) == 1 for a in observed)

            variant = Variant(
                ref=alleles[0],
                pos=int(var.site.position) + 1,  # tskit positions are 0-based, VCF POS is 1-based
                chrom=self._contig,
                gt_bases=gt_bases,
                alt=[a for a in alleles[1:] if a],
                is_snp=is_snp,
            )

            # carry the exact (possibly non-integer) tskit position so TskitVariantWriter can identify the
            # site without relying on the lossy integer POS, which collides on continuous-genome sequences
            variant._tskit_position = var.site.position

            yield variant


class ZarrVariantReader(VariantReader):
    """
    Stream variants from a VCF-Zarr store in the `vcf2zarr <https://sgkit-dev.github.io/bio2zarr>`_ (VCZ)
    layout. Genotypes are read in variant chunks to bound memory. An ancestral-allele INFO field, if
    encoded as a ``variant_<tag>`` array, is surfaced under ``INFO`` so the usual polarisation logic
    applies.
    """

    def __init__(self, path: str, info_ancestral: str = 'AA', chunk_size: int = 1000):
        """
        Initialize the reader.

        :param path: The path to the VCF-Zarr store.
        :param info_ancestral: The INFO tag holding the ancestral allele.
        :param chunk_size: The number of variants read per chunk.
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

        #: The number of variants read per chunk
        self._chunk_size = int(chunk_size)

        #: The sample names
        self._sample_ids = [self._decode(s) for s in self._root['sample_id'][:]]

        #: The contig names
        self._contig_ids = [self._decode(c) for c in self._root['contig_id'][:]]

    @staticmethod
    def _decode(value) -> str:
        """
        Decode a Zarr string scalar (bytes or str) to ``str``.

        :param value: The value.
        :return: The decoded string.
        """
        return value.decode() if isinstance(value, bytes) else str(value)

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
        genotype = root['call_genotype']
        phased = root['call_genotype_phased'] if 'call_genotype_phased' in list(root.array_keys()) else None
        # surface every INFO field the writer persisted as a variant_<key> string array (the ancestral
        # tag, but also e.g. an annotated Degeneracy/Synonymy), skipping the VCF fixed columns that
        # vcf2zarr stores as reserved variant_* arrays (CHROM/POS/ID/REF+ALT/QUAL/FILTER and their
        # length/mask companions); otherwise a store re-parsed stratified by an annotated field would
        # see no INFO, while a plain vcf2zarr store would fabricate INFO from its reserved metadata
        reserved_arrays = {'variant_position', 'variant_contig', 'variant_allele', 'variant_id',
                           'variant_id_mask', 'variant_quality', 'variant_filter', 'variant_length'}
        info_arrays = {k[len('variant_'):]: root[k]
                       for k in root.array_keys()
                       if k.startswith('variant_') and k not in reserved_arrays}

        n = position.shape[0]

        for start in range(0, n, self._chunk_size):
            end = min(start + self._chunk_size, n)

            pos_batch = position[start:end]
            allele_batch = allele[start:end]
            contig_batch = contig[start:end]
            gt_batch = np.asarray(genotype[start:end])
            phased_batch = np.asarray(phased[start:end]) if phased is not None else None
            info_batches = {key: arr[start:end] for key, arr in info_arrays.items()}

            for i in range(end - start):
                site_alleles = [a for a in (self._decode(x) for x in allele_batch[i]) if a not in ('', '.')]

                rows = gt_batch[i]
                seps = phased_batch[i] if phased_batch is not None else None

                gt_bases = np.array([
                    ('|' if seps is not None and seps[s] else '/').join(
                        site_alleles[a] if 0 <= a < len(site_alleles) else '.' for a in rows[s]
                    )
                    for s in range(rows.shape[0])
                ], dtype=object)

                observed = [a for a in site_alleles if a]
                is_snp = len(observed) >= 2 and all(len(a) == 1 for a in observed)

                # every INFO value is decoded to a string (unlike cyvcf2, which types them from the VCF
                # header); a consumer that needs a number must cast explicitly (see _get_ancestral_prob)
                info = {key: self._decode(batch[i]) for key, batch in info_batches.items()}

                yield Variant(
                    ref=site_alleles[0] if site_alleles else '.',
                    pos=int(pos_batch[i]),  # vcf2zarr stores the 1-based VCF position
                    chrom=self._contig_ids[int(contig_batch[i])],
                    gt_bases=gt_bases,
                    alt=site_alleles[1:],
                    is_snp=is_snp,
                    info=info,
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
    allele) are persisted as ``variant_<tag>`` arrays. Variants are buffered in memory and the arrays written
    on :meth:`close`, since the ragged allele dimension needs a global maximum.
    """

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
            import numcodecs  # noqa: F401
        except ImportError:
            raise ImportError(
                "VCF-Zarr support in sfsutils requires the optional 'zarr' package. "
                "Please install sfsutils with the 'zarr' extra: pip install sfsutils[zarr]"
            )

        self._output = output
        self._logger = logger.getChild(self.__class__.__name__)
        self._samples = list(samples)
        self._contig_ids = list(seqnames)
        self._contig_index = {c: i for i, c in enumerate(self._contig_ids)}
        self._info_ancestral = info_ancestral

        self._positions: List[int] = []
        self._contigs: List[int] = []
        self._alleles: List[List[str]] = []
        self._genotypes: List[List[List[int]]] = []
        self._phased: List[List[bool]] = []
        self._infos: List[Dict[str, object]] = []

    def write(self, variant: Site) -> None:
        """
        Buffer a single variant.

        :param variant: The variant to write.
        """
        alleles = [variant.REF] + list(variant.ALT)
        index = {a: i for i, a in enumerate(alleles)}

        rows, phased = [], []
        for gt in np.asarray(variant.gt_bases):
            gt = str(gt)
            phased.append('|' in gt)
            rows.append([index.get(c, -1) for c in re.split(r'[|/]', gt)])

        chrom = variant.CHROM
        if chrom not in self._contig_index:
            self._contig_index[chrom] = len(self._contig_ids)
            self._contig_ids.append(chrom)

        self._positions.append(int(variant.POS))
        self._contigs.append(self._contig_index[chrom])
        self._alleles.append(alleles)
        self._genotypes.append(rows)
        self._phased.append(phased)
        self._infos.append(dict(variant.INFO) if getattr(variant, 'INFO', None) else {})

    def close(self) -> None:
        """
        Write the buffered variants to the store.
        """
        import zarr
        import numcodecs

        root = zarr.open(self._output, mode='w')
        codec = numcodecs.VLenUTF8()

        def _str_array(name: str, values):
            root.create_dataset(name, data=np.asarray([str(v) for v in values], dtype=object),
                                object_codec=codec, overwrite=True)

        _str_array('sample_id', self._samples)
        _str_array('contig_id', self._contig_ids)

        n = len(self._positions)
        n_samples = len(self._samples)
        ploidy = max((len(call) for rows in self._genotypes for call in rows), default=2)
        max_alleles = max((len(a) for a in self._alleles), default=1)

        root.create_dataset('variant_position', data=np.array(self._positions, dtype=np.int64), overwrite=True)
        root.create_dataset('variant_contig', data=np.array(self._contigs, dtype=np.int64), overwrite=True)

        # size the allele-index dtype to the data so many-allele sites do not wrap
        gt_dtype = np.int8 if max_alleles <= np.iinfo(np.int8).max else np.int32
        genotype = np.full((n, n_samples, ploidy), -1, dtype=gt_dtype)
        phased = np.zeros((n, n_samples), dtype=bool)
        for vi, (rows, ph) in enumerate(zip(self._genotypes, self._phased)):
            for si, call in enumerate(rows):
                genotype[vi, si, :len(call)] = call
                phased[vi, si] = ph[si]

        root.create_dataset('call_genotype', data=genotype, overwrite=True)
        root.create_dataset('call_genotype_phased', data=phased, overwrite=True)

        allele = np.full((n, max_alleles), '', dtype=object)
        for vi, a in enumerate(self._alleles):
            allele[vi, :len(a)] = a
        root.create_dataset('variant_allele', data=allele, object_codec=codec, overwrite=True)

        # persist any INFO fields (e.g. an annotated ancestral allele) as variant_<tag> string arrays,
        # skipping any whose name would collide with a reserved coordinate/allele/genotype dataset
        reserved = {'variant_position', 'variant_contig', 'variant_allele',
                    'call_genotype', 'call_genotype_phased', 'sample_id', 'contig_id'}
        info_keys = sorted({k for info in self._infos for k in info})
        for key in info_keys:
            name = f'variant_{key}'

            if name in reserved:
                self._logger.warning(f"Skipping INFO field '{key}': it collides with the reserved VCF-Zarr "
                                     f"dataset '{name}'.")
                continue

            _str_array(name, [str(info.get(key, '')) for info in self._infos])


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
