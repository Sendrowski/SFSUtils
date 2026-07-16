"""
Handlers the reading of VCF, GFF and FASTA files.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-29"

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
from typing import List, Iterable, Iterator, TextIO, Dict, Optional, Tuple, Union, Sequence
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


def is_monomorphic_snp(variant: Union['cyvcf2.Variant', 'DummyVariant']) -> bool:
    """
    Whether the given variant is a monomorphic SNP.

    :param variant: The vcf site
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
    Count the number of sites in the VCF.

    :param vcf: The path to the VCF file or an iterable of variants
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
    Download the VCF file if it is a URL.

    :param path: The path to the VCF file.
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
        Download the VCF file if it is a URL.

        :param path: The path to the VCF file.
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
    Base class for VCF handling.
    """

    def __init__(
            self,
            vcf: str | Iterable['cyvcf2.Variant'],
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {}
    ):
        """
        Create a new VCF instance.

        :param vcf: The variant source: a VCF path (gzipped or a URL), a VCF-Zarr store (.vcz/.zarr), a tskit tree sequence (.trees) or TreeSequence, or an iterable of variants
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        FileHandler.__init__(self, cache=cache, aliases=aliases)

        #: The path to the VCF file or an iterable of variants
        self.vcf = vcf

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
        VCF-Zarr store are read through :class:`TskitVariantReader` and :class:`ZarrVariantReader`;
        everything else is read from VCF via cyvcf2.

        :return: The variant reader.
        """
        if self._is_tree_sequence(self.vcf):
            return TskitVariantReader(self._load_tree_sequence(self.vcf))

        if self._is_zarr_store(self.vcf):
            return ZarrVariantReader(self.vcf, info_ancestral=self.info_ancestral)

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
        Load a VCF file into a dictionary.

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
        Get the number of sites in the VCF.

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
    Handle VCF, FASTA and GFF files.
    """

    def __init__(
            self,
            vcf: str | Iterable['cyvcf2.Variant'],
            fasta: str | None = None,
            gff: str | None = None,
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {}
    ):
        """
        Create a new MultiHandler instance.

        :param vcf: The variant source: a VCF path (gzipped or a URL), a VCF-Zarr store (.vcz/.zarr), a tskit tree sequence (.trees) or TreeSequence, or an iterable of variants
        :param fasta: The path to the FASTA file.
        :param gff: The path to the GFF file.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        # initialize vcf handler
        VCFHandler.__init__(
            self,
            vcf=vcf,
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
    Minimal duck-typed stand-in for a :class:`cyvcf2.Variant`, exposing the subset of its interface that
    the parser, filtrations, annotations and stratifications rely on: ``CHROM``, ``POS``, ``REF``,
    ``ALT``, ``INFO``, the ``is_*`` type flags and the per-sample ``gt_bases``. Non-VCF backends (tree
    sequences, VCF-Zarr stores) emit these objects so they feed the same streaming site interface as
    cyvcf2, without sfsutils having to special-case the input format downstream.
    """

    #: Whether the variant is an SNP
    is_snp = False

    #: Whether the variant is an MNP
    is_mnp = False

    #: Whether the variant is an indel
    is_indel = False

    #: Whether the variant is a deletion
    is_deletion = False

    #: Whether the variant is a structural variant
    is_sv = False

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
    tree sequence and yield :class:`Variant` objects in file order, so the parser can consume any input
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

    def add_info_to_header(self, data: dict):
        """
        Declare an INFO field. On-the-fly annotations that write to :attr:`Variant.INFO` call this to
        register the field with the underlying VCF header; for non-VCF sources (which are never written
        back out) there is no header to update, so this is a no-op.

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
                "/".join(
                    alleles[genotypes[self._node_col[node]]]
                    if genotypes[self._node_col[node]] >= 0 and alleles[genotypes[self._node_col[node]]]
                    else "."
                    for node in group
                )
                for group in self._groups
            ], dtype=object)

            observed = [a for a in alleles if a]
            is_snp = len(observed) >= 2 and all(len(a) == 1 for a in observed)

            yield Variant(
                ref=alleles[0],
                pos=int(var.site.position) + 1,  # tskit positions are 0-based, VCF POS is 1-based
                chrom=self._contig,
                gt_bases=gt_bases,
                alt=[a for a in alleles[1:] if a],
                is_snp=is_snp,
            )


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
        aa_key = f'variant_{self._info_ancestral}'
        aa = root[aa_key] if aa_key in list(root.array_keys()) else None

        n = position.shape[0]

        for start in range(0, n, self._chunk_size):
            end = min(start + self._chunk_size, n)

            pos_batch = position[start:end]
            allele_batch = allele[start:end]
            contig_batch = contig[start:end]
            gt_batch = np.asarray(genotype[start:end])
            phased_batch = np.asarray(phased[start:end]) if phased is not None else None
            aa_batch = aa[start:end] if aa is not None else None

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

                info = {}
                if aa_batch is not None:
                    info[self._info_ancestral] = self._decode(aa_batch[i])

                yield Variant(
                    ref=site_alleles[0] if site_alleles else '.',
                    pos=int(pos_batch[i]),  # vcf2zarr stores the 1-based VCF position
                    chrom=self._contig_ids[int(contig_batch[i])],
                    gt_bases=gt_bases,
                    alt=site_alleles[1:],
                    is_snp=is_snp,
                    info=info,
                )
