"""
SFS utilities.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2022-07-24"

import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Iterable, Iterator, Any, Literal, Sequence, Tuple, TypeVar
from numpy.random import Generator

import jsonpickle
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .io_handlers import download_if_url

# get logger
logger = logging.getLogger('sfsutils')


def standard_kingman(n: int) -> 'Spectrum':
    """
    Get standard Kingman SFS for theta = 1.

    :param n: Sample size (number of haplotypes)
    :return: Spectrum
    """
    return Spectrum(pad(1 / np.arange(1, int(n))))


def pad(counts: Sequence) -> np.ndarray:
    """
    Pad array with monomorphic counts.

    :param counts: SFS counts to pad
    :return: Padded array
    """
    return np.array([0] + list(counts) + [0])


#: Type variable bound to :class:`AbstractSpectrum`, so that inherited self-returning methods (``copy``,
#: ``from_file``, ``from_json``) are typed as the concrete subclass rather than the base.
_S = TypeVar("_S", bound="AbstractSpectrum")


class AbstractSpectrum(ABC):
    """
    Abstract base class for site-frequency spectrum containers.

    A concrete spectrum wraps a numpy array in :attr:`data`: the one-dimensional :class:`Spectrum`, the
    two-dimensional :class:`TwoSFS` (with its :class:`TwoLocusSFS` specialization), and the multi-population
    :class:`JointSFS`. This base supplies the shared array interface and JSON serialization; subclasses add
    dimension-specific behaviour such as folding and plotting.
    """

    #: The underlying array holding the spectrum.
    data: np.ndarray

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The shape of the underlying array.
        """
        return self.data.shape

    @property
    def n_sites(self) -> float:
        """
        The total number of sites, i.e. the sum of all entries.
        """
        return float(np.sum(self.data))

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Numpy array interface so the spectrum can be used directly in numpy operations.

        :param dtype: Optional dtype.
        :return: The underlying array.
        """
        return self.data if dtype is None else self.data.astype(dtype)

    def __iter__(self) -> Iterator:
        """
        Iterate over the first axis of the spectrum.

        :return: Iterator
        """
        return self.data.__iter__()

    def copy(self: _S) -> _S:
        """
        Create a deep copy.

        :return: Deep copy.
        """
        return copy.deepcopy(self)

    def to_file(self, file: str) -> None:
        """
        Save to file (in JSON format).

        :param file: File path.
        """
        with open(file, 'w') as f:
            f.write(self.to_json())

    def to_json(self) -> str:
        """
        Convert to a JSON string.

        :return: JSON string
        """
        obj = copy.deepcopy(self)

        # convert numpy array to list
        obj.data = np.asarray(obj.data).tolist()

        return jsonpickle.encode(obj)

    @classmethod
    def from_file(cls: type[_S], file: str) -> _S:
        """
        Load from file.

        :param file: File path.
        :return: Spectrum
        """
        with open(file, 'r') as f:
            return cls.from_json(f.read())

    @classmethod
    def from_json(cls: type[_S], json: str) -> _S:
        """
        Load from a JSON string.

        :param json: JSON string.
        :return: Spectrum
        """
        obj = jsonpickle.decode(json)

        # convert list back to numpy array
        obj.data = np.array(obj.data)

        return obj

    @abstractmethod
    def plot(self, *args, **kwargs) -> 'plt.Axes':
        """
        Plot the spectrum.

        :return: Axes.
        """
        pass


class Spectrum(AbstractSpectrum):
    """
    Class for holding and manipulating a site-frequency spectrum.
    """

    def __init__(self, data: Sequence[float]):
        """
        Initialize spectrum.

        :param data: SFS counts
        """
        self.data: np.ndarray = np.array(data, dtype=float)

    @property
    def n(self) -> int:
        """
        The sample size.

        :return: Sample size
        """
        return self.data.shape[0] - 1

    @property
    def n_sites(self) -> float:
        """
        The total number of sites.

        :return: Total number of sites
        """
        return sum(self.data)

    @property
    def n_div(self) -> float:
        """
        Number of divergence counts.

        :return: Number of divergence counts
        """
        return self.data[-1]

    @property
    def has_div(self) -> bool:
        """
        Whether n_div was specified.

        :return: Whether n_div was specified
        """
        return self.n_div != 0

    @property
    def n_monomorphic(self) -> float:
        """
        Number of monomorphic sites.

        :return: Number of monomorphic sites
        """
        return self.data[0] + self.data[-1]

    @property
    def polymorphic(self) -> np.ndarray:
        """
        Get the polymorphic counts.

        :return: Polymorphic counts
        """
        return self.data[1:-1]

    @property
    def n_polymorphic(self) -> float:
        """
        The total number of polymorphic sites.

        :return: Total number of polymorphic sites
        """
        return np.sum(self.polymorphic)

    def to_list(self) -> list:
        """
        Convert to list.

        :return: SFS counts
        """
        return list(self.data)

    def to_spectra(self) -> 'Spectra':
        """
        Convert to Spectra object.

        :return: Spectra object
        """
        return Spectra.from_spectrum(self)

    def to_file(self, file: str):
        """
        Save object to file.

        :param file: File name
        """
        self.to_spectra().to_file(file)

    @staticmethod
    def from_file(file: str) -> 'Spectrum':
        """
        Load object from file.

        :param file: File name
        :return: Spectrum object
        """
        return Spectra.from_file(file).to_spectrum()

    def to_numpy(self) -> np.ndarray:
        """
        Convert to array.

        :return: SFS counts
        """
        return self.data

    @property
    def theta(self) -> float:
        """
        Calculate site-wise population mutation rate using Watterson's estimator.
        Note that theta is given per site, i.e. Watterson's estimator is divided by the
        total number of sites (:attr:`n_sites`).
        """
        return self.Theta / self.n_sites

    @property
    def Theta(self) -> float:
        """
        Calculate genome-wide population mutation rate using Watterson's estimator.

        .. note:: Property :attr:`Theta` is not normalized by the total number of sites, unlike :attr:`theta`.
        """
        return self.n_polymorphic / np.sum(1 / np.arange(1, self.n))

    def fold(self) -> 'Spectrum':
        """
        Fold the site-frequency spectrum.

        :return: Folded spectrum
        """
        mid = (self.n + 1) // 2
        data = self.data.copy()

        data[:mid] += data[-mid:][::-1]
        data[-mid:] = 0

        return Spectrum(data)

    def misidentify(self, epsilon: float) -> 'Spectrum':
        """
        Introduce ancestral misidentification at rate epsilon. Note that monomorphic counts won't be affected.

        :param epsilon: Misidentification rate (0 <= epsilon <= 1)
        :return: Spectrum with misidentification applied
        :raise ValueError: If epsilon is not between 0 and 1
        """
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")

        data = self.data.copy()
        n = self.n

        flipped = epsilon * data[1:n][::-1]
        retained = (1 - epsilon) * data[1:n]
        data[1:n] = retained + flipped

        return Spectrum(data)

    def subsample(
            self,
            n: int,
            mode: Literal['random', 'probabilistic'] = 'probabilistic',
            seed: int | Generator = None
    ) -> 'Spectrum':
        """
        Subsample spectrum to a given sample size.

        .. warning::
            If using the 'random' mode, The SFS counts are cast to integers before subsampling so this will
            only provide sensible results if the SFS counts are integers or if they are large enough to be
            approximated well by integers. The 'probabilistic' mode does not have this limitation.

        :param n: Sample size
        :param mode: Subsampling mode. Either 'random' or 'probabilistic'.
        :param seed: Random state or seed. Only for 'random' mode.
        :return: Subsampled spectrum
        """
        if n >= self.n:
            raise ValueError(f'Subsampled sample size {n} must be smaller than original sample size {self.n}.')

        if mode not in ['random', 'probabilistic']:
            raise ValueError(f'Unknown subsampling mode {mode}.')

        subsample = np.zeros(n + 1, dtype=float)

        if mode == 'random':
            # add monomorphic counts
            subsample[0] = self.data[0]
            subsample[-1] = self.data[-1]

            # build a single random state up front so an int seed does not restart the stream for every
            # frequency class (which would make the per-class draws non-independent); thread it through all draws
            if isinstance(seed, Generator):
                rng = seed
            else:
                rng = np.random.default_rng(None if seed is None else int(seed))

            # iterate over spectrum and subsample hypergeometrically
            for i, m in enumerate(self.polymorphic.astype(int)):
                # get subsampled counts
                samples = hypergeom.rvs(M=self.n, n=i + 1, N=n, size=m, random_state=rng)

                # add subsampled counts
                subsample += np.histogram(samples, bins=np.arange(n + 2))[0]
        else:
            for i, m in enumerate(self.data):
                probs = hypergeom.pmf(k=range(n + 1), M=self.n, n=i, N=n)

                # add subsampled counts
                subsample += m * probs

        sfs = Spectrum(subsample)

        # fold if original spectrum was folded
        if self.is_folded():
            sfs = sfs.fold()

        return sfs

    def resample(self, seed: int | Generator = None) -> 'Spectrum':
        """
        Resample SFS assuming independent Poisson counts.

        :param seed: Random state or seed
        :return: Resampled spectrum.
        """
        if isinstance(seed, Generator):
            rng = seed
        else:
            rng = np.random.default_rng(None if seed is None else int(seed))

        return Spectrum.from_polydfe(
            # resample polymorphic sites only
            polymorphic=rng.poisson(lam=self.polymorphic),
            n_sites=self.n_sites,
            n_div=rng.poisson(self.n_div)
        )

    def is_folded(self) -> bool:
        """
        Check if the site-frequency spectrum is folded.

        :return: True if folded, False otherwise
        """
        mid = (self.n + 1) // 2

        return np.all(self.data[-mid:] == 0)

    def normalize(self) -> 'Spectrum':
        """
        Normalize SFS so that all non-monomorphic counts add up to 1.

        :return: Normalized spectrum
        """
        # copy array
        data = self.data.copy()

        # normalize counts; a spectrum with no polymorphic sites has nothing to normalize, so leave the
        # (all-zero) interior as is rather than dividing by zero into NaNs
        total = data[1:-1].sum()
        if total > 0:
            data[1:-1] /= total

        return Spectrum(data)

    def copy(self) -> 'Spectrum':
        """
        Copy the spectrum.

        :return: Copy of the spectrum
        """
        return Spectrum(self.data.copy())

    @staticmethod
    def from_polymorphic(data: Sequence) -> 'Spectrum':
        """
        Create Spectrum from polymorphic counts only.

        :param data: Polymorphic counts
        :return: Spectrum
        """
        return Spectrum([0] + list(data) + [0])

    @staticmethod
    def from_list(data: Sequence) -> 'Spectrum':
        """
        Create Spectrum from list.

        :param data: SFS counts
        :return: Spectrum
        """
        return Spectrum(data)

    @staticmethod
    def from_polydfe(
            polymorphic: Sequence,
            n_sites: float,
            n_div: float
    ) -> 'Spectrum':
        """
        Create Spectrum from polyDFE specification which treats the number
        of mutational target sites and the divergence counts separately.

        :param polymorphic: Polymorphic counts
        :param n_sites: Total number of sites
        :param n_div: Number of divergence counts
        :return: Spectrum
        """
        # determine number of monomorphic ancestral counts
        n_monomorphic = n_sites - np.sum(list(polymorphic) + [n_div])

        data = [n_monomorphic] + list(polymorphic) + [n_div]

        return Spectrum(data)

    @staticmethod
    def _array_or_scalar(data: Iterable | float) -> np.ndarray | float:
        """
        Convert to array if iterable or return scalar otherwise.

        :param data: Iterable or scalar.
        :return: Array or scalar
        """
        if isinstance(data, Iterable):
            return np.array(list(data))

        return data

    def __mul__(self, other: Iterable | float) -> 'Spectrum':
        """
        Multiply spectrum.

        :param other: Iterable or scalar
        :return: Spectrum
        """
        return Spectrum(self.data * self._array_or_scalar(other))

    __rmul__ = __mul__

    def __add__(self, other: Iterable | float) -> 'Spectrum':
        """
        Add spectrum.

        :param other: Iterable or scalar
        :return: Spectrum
        """
        return Spectrum(self.data + self._array_or_scalar(other))

    def __sub__(self, other: Iterable | float) -> 'Spectrum':
        """
        Subtract spectrum.

        :param other: Iterable or scalar
        :return: Spectrum
        """
        return Spectrum(self.data - self._array_or_scalar(other))

    def __pow__(self, other: Iterable | float) -> 'Spectrum':
        """
        Power operator.

        :param other: Iterable or scalar
        :return: Spectrum
        """
        return Spectrum(self.data ** self._array_or_scalar(other))

    def __floordiv__(self, other: Iterable | float) -> 'Spectrum':
        """
        Divide spectrum.

        :param other: Iterable or scalar
        :return: Spectrum
        """
        return Spectrum(self.data // self._array_or_scalar(other))

    def __truediv__(self, other: Iterable | float) -> 'Spectrum':
        """
        Divide spectrum.

        :param other: Iterable or scalar
        :return: Spectrum
        """
        return Spectrum(self.data / self._array_or_scalar(other))

    def __iter__(self):
        """
        Get iterator.

        :return: Iterator
        """
        return self.data.__iter__()

    def __len__(self) -> int:
        """
        Get length of spectrum (including monomorphic and fixed counts).

        :return: Length of spectrum
        """
        return self.data.shape[0]

    def plot(
            self,
            show: bool = True,
            file: str = None,
            title: str = None,
            log_scale: bool = False,
            show_monomorphic: bool = False,
            kwargs_legend: dict = dict(prop=dict(size=8)),
            ax: 'plt.Axes' = None
    ) -> 'plt.Axes':
        """
        Plot spectrum.

        :param show: Whether to show plot.
        :param file: File to save plot to.
        :param title: Title of plot.
        :param log_scale: Whether to use log scale on y-axis.
        :param show_monomorphic: Whether to show monomorphic counts.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :return: Axes
        """
        from .visualization import Visualization

        return Visualization.plot_spectra(
            spectra=[self.to_list()],
            file=file,
            show=show,
            title=title,
            log_scale=log_scale,
            show_monomorphic=show_monomorphic,
            ax=ax,
            kwargs_legend=kwargs_legend
        )

    @staticmethod
    def kingman(n: int, n_monomorphic: int = 0) -> 'Spectrum':
        """
        The standard (Kingman) neutral site-frequency spectrum for a sample of size ``n``.

        :param n: sample size
        :param n_monomorphic: number of monomorphic sites placed in the zero-frequency bin
        :return: the Kingman SFS
        """
        sfs = standard_kingman(n)
        sfs.data[0] = n_monomorphic

        return sfs

    @staticmethod
    def standard_kingman(n: int, n_monomorphic: int = 0) -> 'Spectrum':
        """
        Alias of :meth:`kingman`.

        .. deprecated:: 1.0.0
            Use :meth:`kingman` instead.

        :param n: sample size
        :param n_monomorphic: Number of monomorphic sites.
        :return: Standard Kingman SFS
        """
        return Spectrum.kingman(n, n_monomorphic)

    @staticmethod
    def get_neutral(
            theta: float,
            n_sites: float,
            n: int,
            r: Sequence[float] = None
    ) -> 'Spectrum':
        """
        Obtain a standard neutral SFS for a given theta and number of sites.

        :param theta: Population mutation rate
        :param n_sites: Number of total sites
        :param n: Number of frequency classes
        :param r: Nuisance parameters that account for demography. An array of length ``n-1`` whose elements are
            multiplied element-wise with the polymorphic counts of the Kingman SFS. By default, no demography effects
            are considered which is equivalent to ``r = [1] * (n-1)``. Note that non-default values of ``r`` will also
            affect estimates of the population mutation rate.
        :return: Neutral SFS
        """
        n = int(n)

        if r is None:
            r = np.ones(n + 1)
        else:
            r = list(r)

            if len(r) != n - 1:
                raise ValueError(f"The length of r must be n - 1 = {n - 1}; got {len(r)}.")

            r = np.array([1] + r + [1])

        sfs: Spectrum = Spectrum.standard_kingman(n=n) * theta * n_sites

        # add demography
        sfs.data *= r

        # add monomorphic counts
        sfs.data[0] = n_sites - sfs.n_sites

        return sfs

    def scale_theta(self, theta: float) -> 'Spectrum':
        """
        Scale the spectrum to a different theta value by

        :param theta: New theta value
        :return: Scaled spectrum
        """
        data = self.data.copy()

        # an all-monomorphic spectrum has theta == 0 and no polymorphic mass to rescale; guard the
        # division so the interior stays zero rather than becoming NaN via theta / 0
        if self.theta > 0:
            data[1:-1] *= theta / self.theta
        data[0] = self.n_sites - data[1:-1].sum() - data[-1]

        return Spectrum(data)


class AbstractSpectra(ABC):
    """
    Abstract base class for a collection of site-frequency spectra keyed by type, for example the SFS stratified
    into neutral and selected sites. The concrete backings share this interface but not their storage:
    :class:`~sfsutils.spectrum.Spectra` is DataFrame-backed and one-dimensional, while :class:`JointSpectra` is dict-backed and holds
    multi-population :class:`JointSFS` objects. Code needing only the common collection operations can be written
    against this base regardless of dimensionality.
    """

    @property
    @abstractmethod
    def types(self) -> List[str]:
        """
        The types.
        """
        pass

    @property
    @abstractmethod
    def all(self) -> 'AbstractSpectrum':
        """
        The ``all`` type, equal to the sum of all spectra.
        """
        pass

    @abstractmethod
    def __getitem__(self, key):
        """
        Get the spectrum (or sub-collection) for a given type.

        :param key: Type.
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        """
        Iterate over the types.

        :return: Iterator
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        The number of types.

        :return: Number of types
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Convert to a dictionary keyed by type.

        :return: Dictionary keyed by type
        """
        pass

    @abstractmethod
    def to_file(self, file: str) -> None:
        """
        Save to file.

        :param file: File path.
        """
        pass


class Spectra(AbstractSpectra):
    """
    Class for holding and manipulating site-frequency spectra of multiple types.
    """

    def __init__(self, data: Dict[str, Iterable]):
        """
        Initialize spectra.

        :param data: Dictionary of SFS counts keyed by type
        """
        self.data: pd.DataFrame = pd.DataFrame(data)

    @property
    def n(self) -> int:
        """
        The sample size.

        :return: Sample size
        """
        return self.data.shape[0] - 1

    @property
    def k(self) -> int:
        """
        The number of types.

        :return: Number of types
        """
        return self.data.shape[1]

    @property
    def n_monomorphic(self) -> pd.Series:
        """
        The number of monomorphic sites.

        :return: Number of monomorphic sites
        """
        return self.data.iloc[0] + self.data.iloc[-1]

    @property
    def polymorphic(self) -> np.ndarray:
        """
        The polymorphic counts.

        :return: Polymorphic counts
        """
        return self.data[1:-1]

    @property
    def n_polymorphic(self) -> pd.Series:
        """
        The total number of polymorphic sites per type.

        :return: Total number of polymorphic sites for each type
        """
        return self.polymorphic.sum()

    @staticmethod
    def from_list(data: Sequence, types: List) -> 'Spectra':
        """
        Create from array of spectra.
        Note that data.ndim needs to be 2.

        :param data: Array of spectra
        :param types: Types
        :return: Spectra
        """
        return Spectra(dict((t, d) for t, d in zip(types, data)))

    @property
    def types(self) -> List[str]:
        """
        The types.

        :return: Types
        """
        return self.data.columns.to_list()

    @property
    def n_sites(self) -> pd.Series:
        """
        The number of mutational target sites which is the sum of all SFS entries.

        :return: Number of mutational target sites for each type
        """
        return self.data.sum()

    @property
    def n_div(self) -> pd.Series:
        """
        The number of divergence counts.

        :return: Number of divergence counts for each type
        """
        return self.data.iloc[-1]

    @property
    def has_div(self) -> pd.Series:
        """
        Whether n_div was specified.

        :return: Whether n_div was specified for each type
        """
        # noinspection PyTypeChecker
        return self.n_div != 0

    @property
    def theta(self) -> pd.Series:
        """
        Calculate site-wise population mutation rate using Watterson's estimator.
        Note that theta is given per site, i.e. Watterson's estimator is divided by the
        total number of sites (:attr:`n_sites`).
        """
        return self.Theta / self.n_sites

    @property
    def Theta(self) -> pd.Series:
        """
        Calculate genome-wide population mutation rate using Watterson's estimator.

        .. note:: Property :attr:`Theta` is not normalized by the total number of sites, unlike :attr:`theta`.
        """
        return self.n_polymorphic / np.sum(1 / np.arange(1, self.n))

    def normalize(self) -> 'Spectra':
        """
        Normalize spectra by sum of all entries.

        :return: Normalized spectra
        """
        # normalize each type by its own column sum; an all-zero (empty) type has nothing to normalize,
        # so divide it by one instead of by zero, leaving its column zero rather than NaN
        sums = self.data.sum().replace(0, 1)

        return self / sums

    def to_file(self, file: str):
        """
        Save object to file.

        :param file: File name
        """
        self.data.to_csv(file, index=False)

    def to_spectra(self) -> Dict[str, Spectrum]:
        """
        Convert to dictionary of spectrum objects.

        :return: Dictionary of spectrum objects
        """
        return dict((t, self.select(t, use_regex=False)) for t in self.types)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get representation as dataframe.

        :return: Dataframe
        """
        return self.data

    def to_numpy(self) -> np.ndarray:
        """
        Convert to numpy array.

        :return: Numpy array
        """
        return self.data.to_numpy().T

    def to_list(self) -> list:
        """
        Convert to nested list.

        :return: Nested list
        """
        return list(list(d) for d in self.to_numpy())

    def to_dict(self) -> dict:
        """
        Convert to dictionary.

        :return: Dictionary of lists
        """
        # return dictionary of lists
        return dict((k, list(v.values())) for k, v in self.data.to_dict().items())

    def __mul__(self, other: Any) -> 'Spectra':
        """
        Multiply Spectra.

        :param other: Scalar
        :return: Spectra
        """
        return Spectra.from_dataframe(self.data * other)

    __rmul__ = __mul__

    def __floordiv__(self, other: Any) -> 'Spectra':
        """
        Divide Spectra.

        :param other: Scalar
        :return: Spectra
        """
        return Spectra.from_dataframe(self.data // other)

    def __truediv__(self, other: Any) -> 'Spectra':
        """
        Divide Spectra.

        :param other: Scalar
        :return: Spectra
        """
        return Spectra.from_dataframe(self.data / other)

    def __len__(self) -> int:
        """
        Get number of spectra.

        :return: Number of spectra
        """
        return self.k

    def __add__(self, other: 'Spectra') -> 'Spectra':
        """
        Merge types of two spectra objects by adding up their counts entry-wise.

        :param other: Spectra object
        :return: Spectra with merged types
        """
        return Spectra.from_dataframe(self.data.add(other.data, fill_value=0))

    def __getitem__(
            self,
            keys: str | List[str] | np.ndarray | tuple,
            use_regex: bool = True
    ) -> Union['Spectrum', 'Spectra']:
        """
        Get item.

        :param keys: String or list of strings, possibly regex to match type names
        :param use_regex: Whether to use regex to match type names
        :return: Spectrum or Spectra object depending on the number of matches
        """
        # whether the input in an array
        is_array = isinstance(keys, (np.ndarray, list, tuple))

        if use_regex:
            # subset dataframe using column names using regex
            subset = self.data.loc[:, self.data.columns.str.fullmatch('|'.join(keys) if is_array else keys)]
        else:
            # subset dataframe using column names
            subset = self.data.loc[:, keys]

        # return spectrum object if we have a series
        if isinstance(subset, pd.Series):
            return Spectrum(list(subset))

        # return spectrum object if only one column is left
        # and if not multiple keys were supplied
        if subset.shape[1] == 1 and not is_array:
            return Spectrum(list(subset.iloc[:, 0]))

        # wrap subset dataframe in spectra object
        return Spectra.from_dataframe(subset)

    def __setitem__(self, key: str, s: Spectrum):
        """
        Save new spectrum as type.

        :param key: Type
        :param s: Spectrum
        """
        self.data[key] = s.to_list()

    def __iter__(self):
        """
        Get iterator.

        :return: Iterator
        """
        return self.data.__iter__()

    def select(
            self,
            keys: str | List[str] | np.ndarray | tuple,
            use_regex: bool = True
    ) -> 'Spectra':
        """
        Select types. Alias for __getitem__.

        :param keys: String or list of strings, possibly regex to match type names
        :param use_regex: Whether to use regex to match type names
        :return: Spectrum or Spectra depending on the number of matches
        """
        return self.__getitem__(keys, use_regex=use_regex)

    def copy(self) -> 'Spectra':
        """
        Copy object.

        :return: Copy of object
        """
        return Spectra.from_dataframe(self.data.copy())

    def _to_multi_index(self) -> 'Spectra':
        """
        Convert to Spectra object with multi-indexed columns.

        :return: Spectra object with multi-indexed columns
        """
        other = self.copy()
        columns = [tuple(col.split('.')) for col in other.data.columns]
        other.data.columns = pd.MultiIndex.from_tuples(columns)

        return other

    def _to_single_index(self) -> 'Spectra':
        """
        Convert to Spectra object with single-indexed columns (using dot notation).

        :return: Spectra object with single-indexed columns
        """
        other = self.copy()

        if other.data.columns.nlevels > 1:
            columns = other.data.columns.map('.'.join)
            other.data.columns = columns

        return other

    def get_empty(self) -> 'Spectra':
        """
        Get a Spectra object with zero counts but having the same shape and types as self.

        :return: Spectra object with zero counts
        """
        return Spectra.from_dataframe(pd.DataFrame(0, index=self.data.index, columns=self.data.columns))

    def merge_groups(self, level: List[int] | int = 0) -> 'Spectra':
        """
        Group over given levels and sum up spectra so the spectra
        are summed over the levels that were not specified.

        :param level: Level(s) to group over
        :return: Spectra object with merged groups
        """
        # cast to int
        level = [int(l) for l in level] if isinstance(level, Iterable) else int(level)

        return Spectra.from_dataframe(self._to_multi_index().data.T.groupby(level=level).sum().T)._to_single_index()

    def has_dots(self) -> bool:
        """
        Check whether column names contain dots.

        :return: True if column names contain dots, False otherwise
        """
        return any('.' in col for col in self.data.columns)

    def replace_dots(self, replacement: str = '_') -> 'Spectra':
        """
        Replace dots in column names with a given string.

        :param replacement: Replacement string
        :return: Spectra object with replaced dots
        """
        other = self.copy()
        other.data.columns = other.data.columns.str.replace('.', replacement)

        return other

    @property
    def all(self) -> 'Spectrum':
        """
        The 'all' type equals the sum of all spectra.

        :return: Spectrum object
        """
        return Spectrum(self.data.sum(axis=1).to_list())

    def combine(self, s: 'Spectra') -> 'Spectra':
        """
        Merge types of two Spectra objects.

        :param s: Other Spectra object
        :return: Merged Spectra object
        """
        return Spectra(self.to_dict() | s.to_dict())

    @staticmethod
    def from_dict(data: dict) -> 'Spectra':
        """
        Load from nested dictionary first indexed by types and then by samples.

        :param data: Dictionary of lists indexed by types
        :return: Spectra object
        """
        lists = [list(v.values() if isinstance(v, dict) else v) for v in data.values()]

        return Spectra.from_list(lists, types=list(data.keys()))

    @staticmethod
    def from_dataframe(data: pd.DataFrame) -> 'Spectra':
        """
        Load Spectra object from dataframe.

        :param data: Dataframe
        :return: Spectra object
        """
        return Spectra.from_dict(data.to_dict())

    @classmethod
    def from_file(cls, file: str) -> 'Spectra':
        """
        Save object to file.

        :param file: Path to file, possibly URL
        :return: Spectra object
        """
        return Spectra.from_dataframe(pd.read_csv(download_if_url(
            file,
            desc=f'{cls.__name__}>Downloading file'))
        )

    @staticmethod
    def from_spectra(spectra: Dict[str, Spectrum]) -> 'Spectra':
        """
        Create from dict of spectrum objects indexed by type.

        :param spectra: Dictionary of spectrum objects indexed by type
        :return: Spectra object
        """
        return Spectra.from_list(
            [sfs.to_list() for sfs in spectra.values()],
            types=list(spectra.keys())
        )

    @staticmethod
    def from_spectrum(sfs: Spectrum) -> 'Spectra':
        """
        Create from single spectrum object. The type of the spectrum is set to 'all'.

        :param sfs: Spectrum
        :return: Spectra object
        """
        return Spectra.from_spectra(dict(all=sfs))

    def to_spectrum(self) -> Spectrum:
        """
        Convert to Spectrum object by summing over all types.

        :return: Spectrum object
        """
        return self.all

    def plot(
            self,
            show: bool = True,
            file: str = None,
            title: str = None,
            log_scale: bool = False,
            use_subplots: bool = False,
            show_monomorphic: bool = False,
            kwargs_legend: dict = dict(prop=dict(size=8)),
            ax: 'plt.Axes' = None
    ) -> 'plt.Axes':
        """
        Visualize spectra.

        :param show: Whether to show the plot.
        :param file: File name to save the plot to.
        :param title: Plot title.
        :param log_scale: Whether to use log scale on y-axis.
        :param use_subplots: Whether to use subplots. Only for Python visualization backend.
        :param show_monomorphic: Whether to show monomorphic sites.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :param ax: Axes to plot on. Only for Python visualization backend and if ``use_subplots`` is ``False``.
        :return: Axes
        """
        from .visualization import Visualization

        return Visualization.plot_spectra(
            spectra=list(list(v) for v in self.to_spectra().values()),
            labels=self.types,
            file=file,
            show=show,
            title=title,
            log_scale=log_scale,
            use_subplots=use_subplots,
            show_monomorphic=show_monomorphic,
            kwargs_legend=kwargs_legend,
            ax=ax
        )

    def drop_empty(self) -> 'Spectra':
        """
        Remove types whose spectra have no counts.

        :return: Spectra with non-empty types
        """
        return Spectra.from_dataframe(self.data.loc[:, self.data.any()])

    def drop_zero_entries(self) -> 'Spectra':
        """
        Remove types whose spectra have some zero entries.
        Note that we ignore zero counts in the last entry i.e. fixed derived alleles.

        :return: Spectra with non-zero entries
        """
        return Spectra.from_dataframe(self.data.loc[:, self.data[:-1].all()])

    def drop_sparse(self, n_polymorphic: int) -> 'Spectra':
        """
        Remove types whose spectra have fewer than equal ``n_polymorphic`` polymorphic sites.

        :return: Spectra
        """
        return Spectra.from_dataframe(self.data.loc[:, self.data[1:-1].sum() > int(n_polymorphic)])

    def rename(self, names: List[str]) -> 'Spectra':
        """
        Rename types.

        :param names: New names
        :return: Spectra with renamed types
        """
        other = self.copy()
        other.data.columns = names

        return other

    def prefix(self, prefix: str) -> 'Spectra':
        """
        Prefix types, i.e. 'type' -> 'prefix.type' for all types.

        :param prefix: Prefix
        :return: Spectra with prefixed types
        """
        return self.rename([prefix + '.' + col for col in self.types])

    def reorder_levels(self, levels: List[int]) -> 'Spectra':
        """
        Reorder levels.

        :param levels: New order of levels
        :return: Spectra with reordered levels
        """
        s = self._to_multi_index()
        s.data.columns = s.data.columns.reorder_levels(levels)
        s = s._to_single_index()

        return s

    def print(self):
        """
        Print spectra.
        """
        print(self.data.T)

    def fold(self):
        """
        Fold spectra.

        :return: Folded spectra
        """
        return Spectra.from_spectra({t: s.fold() for t, s in self.to_spectra().items()})

    def subsample(
            self,
            n: int,
            mode: Literal['random', 'probabilistic'] = 'probabilistic',
            seed: int | Generator = None
    ) -> 'Spectra':
        """
        Subsample spectra to a given sample size.

        .. warning::
            If using the 'random' mode, The SFS counts are cast to integers before subsampling so this will
            only provide sensible results if the SFS counts are integers or if they are large enough to be
            approximated by integers. The 'probabilistic' mode does not have this limitation.

        :param n: Sample size
        :param mode: Subsampling mode. Either 'random' or 'probabilistic'.
        :param seed: Random state or seed. Only for 'random' mode.
        :return: Subsampled spectra
        """
        spectra = self.to_spectra()
        rngs = self._spawn_rngs(seed, len(spectra))

        return Spectra.from_spectra(
            {t: s.subsample(n, mode, rng) for (t, s), rng in zip(spectra.items(), rngs)}
        )

    def resample(self, seed: int | Generator = None) -> 'Spectra':
        """
        Resample SFS assuming independent Poisson counts.

        :param seed: Random state or seed
        :return: Resampled spectra.
        """
        spectra = self.to_spectra()
        rngs = self._spawn_rngs(seed, len(spectra))

        return Spectra.from_spectra(
            {t: s.resample(rng) for (t, s), rng in zip(spectra.items(), rngs)}
        )

    @staticmethod
    def _spawn_rngs(seed: int | Generator, n: int) -> List[Generator]:
        """
        Spawn independent child generators, one per type, so that with an int seed the types are not resampled
        or subsampled from an identical stream (which would break bootstrap independence across types) while
        the result stays reproducible for a given seed.

        :param seed: Random state or seed shared by all types
        :param n: Number of independent generators to produce
        :return: List of independent generators
        """
        if isinstance(seed, Generator):
            return list(seed.spawn(n))

        return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(n)]

    def is_folded(self) -> Dict[str, bool]:
        """
        Check whether spectra are folded.

        :return: Dictionary of types and whether they are folded
        """
        return {t: s.is_folded() for t, s in self.to_spectra().items()}

    def sort_types(self) -> 'Spectra':
        """
        Sort types alphabetically.

        :return: Sorted spectra object
        """
        return Spectra.from_dataframe(self.data.sort_index(axis=1))


class TwoSFS(AbstractSpectrum):
    """
    A 2-dimensional site-frequency spectrum, i.e. a square matrix whose entry ``(i, j)`` relates the number of
    derived alleles at a pair of frequency classes ``i`` and ``j`` of a single population (for example the
    second moment of the SFS). For the joint spectrum *across* populations, which is generally rectangular, see
    :class:`JointSFS`.
    """

    # class-level defaults so jsonpickle can restore spectra serialized without these attributes
    n: int = None
    w: int = None

    def __init__(self, data: np.ndarray | list) -> None:
        """
        Construct from a data matrix.

        :param data: A square 2-dimensional array.
        :raises ValueError: If the data is not a square 2-dimensional array.
        """
        # store as float so operations that assign a NaN fill value (fill_monomorphic, mask_diagonal, mask_upper)
        # work regardless of the input dtype
        data = np.array(data, dtype=float)

        if data.ndim != 2:
            raise ValueError('Data has to be 2-dimensional.')

        if data.shape[0] != data.shape[1]:
            raise ValueError('Matrix has to be square.')

        #: The matrix dimension along one axis (the sample size plus one, i.e. the number of derived-count bins).
        self.n: int = data.shape[0]

        #: The width of one folded half.
        self.w: int = self.n // 2 + 1 if self.n % 2 == 1 else self.n // 2

        #: The 2-SFS matrix.
        self.data: np.ndarray = data

    def is_folded(self) -> bool:
        """
        Check if the 2-SFS is folded.

        :return: Whether the 2-SFS is folded.
        """
        return np.all(self.data == self.fold().data)

    def __add__(self, other) -> 'TwoSFS':
        """
        Add to the 2-SFS.

        :param other: Another 2-SFS or a scalar or array.
        :return: 2-SFS
        """
        if isinstance(other, TwoSFS):
            return self + other.data

        return TwoSFS(self.data + other)

    def __sub__(self, other) -> 'TwoSFS':
        """
        Subtract from the 2-SFS.

        :param other: Another 2-SFS or a scalar or array.
        :return: 2-SFS
        """
        if isinstance(other, TwoSFS):
            return self - other.data

        return TwoSFS(self.data - other)

    def __mul__(self, other) -> 'TwoSFS':
        """
        Multiply the 2-SFS.

        :param other: Another 2-SFS or a scalar or array.
        :return: 2-SFS
        """
        if isinstance(other, TwoSFS):
            return self * other.data

        return TwoSFS(self.data * other)

    def __floordiv__(self, other) -> 'TwoSFS':
        """
        Floor-divide the 2-SFS.

        :param other: Another 2-SFS or a scalar or array.
        :return: 2-SFS
        """
        if isinstance(other, TwoSFS):
            return self // other.data

        return TwoSFS(self.data // other)

    def __truediv__(self, other) -> 'TwoSFS':
        """
        Divide the 2-SFS.

        :param other: Another 2-SFS or a scalar or array.
        :return: 2-SFS
        """
        if isinstance(other, TwoSFS):
            return self / other.data

        return TwoSFS(self.data / other)

    def __pow__(self, power) -> 'TwoSFS':
        """
        Raise the 2-SFS to a power.

        :param power: Exponent
        :return: 2-SFS
        """
        return TwoSFS(self.data ** power)

    def fold(self) -> 'TwoSFS':
        """
        Fold the 2-SFS by adding up ``i`` and ``n - i`` for both axes.
        Note that this only makes sense for counts or frequencies.

        :return: Folded 2-SFS.
        """
        data = self.data.copy()

        for _ in range(2):
            # compute left and right half and merge them
            left = np.concatenate((data[:self.w], np.zeros((self.n - self.w, self.n))))
            right = np.concatenate((data[self.w:][::-1], np.zeros((self.w, self.n))))

            # add parts and rotate
            data = (left + right).T

        return TwoSFS(data)

    def symmetrize(self) -> 'TwoSFS':
        """
        Symmetrize the 2-SFS so that ``i, j`` and ``j, i`` are the same.

        :return: Symmetric 2-SFS.
        """
        return TwoSFS((self.data + self.data.T) / 2)

    def interior(self, normalize: bool = False) -> np.ndarray:
        """
        The interior (segregating) block of the 2-SFS: pairs for which both sites carry between ``1`` and ``n - 1``
        derived alleles, i.e. both are polymorphic. The two monomorphic bins (all-ancestral and all-derived) are
        excluded, matching the polymorphic block of the standard site-frequency spectrum.

        :param normalize: If ``True``, return the conditional joint distribution ``P(i, j | both polymorphic)``
            instead of the raw pair counts.
        :return: The ``(n - 1) x (n - 1)`` interior block, or its normalization.
        :raises ValueError: If ``normalize`` is requested but the interior is empty.
        """
        interior = self.data[1:-1, 1:-1].astype(float)

        if not normalize:
            return interior

        total = interior.sum()

        if total == 0:
            raise ValueError('The interior (segregating) block of the 2-SFS is empty; cannot normalize.')

        return interior / total

    def _embed(self, interior: np.ndarray) -> 'TwoSFS':
        """
        Embed an interior (segregating) matrix back into a full ``n x n`` :class:`TwoSFS`, with the two
        monomorphic rows and columns set to zero, so that its indexing stays aligned with this spectrum.

        :param interior: The ``(n - 1) x (n - 1)`` interior matrix.
        :return: The full-size :class:`TwoSFS`.
        """
        data = np.zeros_like(self.data, dtype=float)
        data[1:-1, 1:-1] = interior

        return TwoSFS(data)

    def _branch_length_covariance(self) -> np.ndarray:
        """
        The full-spectrum branch-length covariance ``Cov(L_i, L_j) = P(i, j) - P(i) P(j)``, where ``P`` is the
        two-SFS normalized over *all* pairs (monomorphic bins included) and ``P(i)`` is its marginal. For a two-SFS
        in which every site pairs with the same number of window partners, that marginal equals the one-dimensional
        site-frequency spectrum, so the interior block of this matrix is the class-resolved branch-length covariance
        between two linked sites, matching PhaseGen's ``sfs2.mean - outer(sfs.mean, sfs.mean)`` up to the (constant)
        mutational scale. The marginal only approximates the site-frequency spectrum where sites pair with unequal
        partner counts (within a pairing window of a contig boundary, or in sparse regions); the deviation is
        negligible when the pairing window is small relative to the region.

        This requires the monomorphic sites: they anchor the marginal to the per-site class distribution. Without
        them the marginal collapses to the polymorphic-only distribution and the interior deviation is the
        model-blind conditional joint, which does not recover the coalescent signal. The two-SFS is symmetrized here
        (it counts unordered site pairs), so the covariance is symmetric regardless of the input.

        :return: The full ``(n + 1) x (n + 1)`` covariance matrix.
        :raises ValueError: If the 2-SFS is empty, non-finite, or carries no monomorphic-involving pairs.
        """
        # a two-SFS counts unordered site pairs, so symmetrize; this makes the row and column marginals (the SFS)
        # coincide and the covariance symmetric even for a directly-constructed asymmetric matrix
        data = self.data + self.data.T
        total = data.sum()

        if not np.isfinite(total):
            raise ValueError('The 2-SFS contains non-finite values (NaN or inf).')

        if total == 0:
            raise ValueError('The 2-SFS is empty.')

        # the two monomorphic bins (first and last row/column) must carry mass: they anchor the marginal to the
        # site-frequency spectrum. Their absence (a polymorphic-only spectrum) leaves the covariance undefined.
        border = np.concatenate([data[[0, -1], :].ravel(), data[:, [0, -1]].ravel()])
        if not np.any(border):
            raise ValueError(
                'The 2-SFS carries no monomorphic-involving pairs, so the branch-length covariance/correlation is '
                'not defined: the marginal is not anchored to the site-frequency spectrum. Parse an all-sites input '
                'with the monomorphic (invariant) sites specified.'
            )

        p = data / total
        m = p.sum(axis=0)

        return p - np.outer(m, m)

    def cov(self) -> 'TwoSFS':
        """
        The class-resolved branch-length covariance ``Cov(L_i, L_j)`` between two linked sites, returned over the
        segregating interior of a full-size :class:`TwoSFS` (monomorphic bins zeroed). Entry ``(i, j)`` is the
        covariance of the branch lengths subtending ``i`` and ``j`` derived alleles at the two loci; a positive
        entry means the two classes co-vary positively across linked sites. It is the deviation of the full joint
        class distribution from independence, ``P(i, j) - P(i) P(j)``, normalized over *all* pairs so that ``P(i)``
        is the true site-frequency spectrum. It therefore matches PhaseGen's ``sfs2.mean - outer(sfs.mean)`` (up to
        the constant mutational scale) and reproduces the multiple-merger signal, unlike a polymorphic-only
        normalization.

        This requires the monomorphic-site counts: parse an all-sites input with the monomorphic (invariant) sites
        specified. With only a :class:`~sfsutils.parser.TargetSiteCounter` (extrapolated, not real, monomorphic
        counts) the result is approximate and can be unreliable, as the interior residual is hypersensitive to the
        target-site count; use a real all-sites input for an accurate covariance, or :meth:`fpmi` for a
        monomorphic-free statistic on polymorphic-only data. For a valid two-SFS the interior diagonal is a
        nonnegative branch-length variance; an arbitrary (non-coalescent) input can give a non-positive-semidefinite
        result.

        :return: The branch-length covariance over the interior, embedded in a full-size :class:`TwoSFS`.
        :raises ValueError: If the 2-SFS is empty, non-finite, or carries no monomorphic-involving pairs.
        """
        return self._embed(self._branch_length_covariance()[1:-1, 1:-1])

    def corr(self) -> 'TwoSFS':
        """
        The class-resolved branch-length correlation corresponding to :meth:`cov`: ``R[i, j] = Cov(L_i, L_j) /
        sqrt(Var(L_i) Var(L_j))`` over the segregating interior, standardized by the branch-length variances (the
        interior diagonal of the full-spectrum covariance). For a valid (coalescent) two-SFS this is a proper
        correlation, matching PhaseGen's branch-length correlation, with entries in ``[-1, 1]`` and a unit diagonal.
        Like :meth:`cov`, it requires the monomorphic-site counts and is only approximate under a
        :class:`~sfsutils.parser.TargetSiteCounter` (see :meth:`cov`; :meth:`fpmi` needs no monomorphic sites).
        Classes with negligible branch-length variance are returned as zero, and because the underlying quantity is a
        cross-covariance between the two loci (not a within-locus covariance matrix), the entries are clipped to
        ``[-1, 1]`` as a safeguard for arbitrary input.

        :return: The branch-length correlation over the interior, embedded in a full-size :class:`TwoSFS`.
        :raises ValueError: If the 2-SFS is empty, non-finite, or carries no monomorphic-involving pairs.
        """
        c = self._branch_length_covariance()
        var = np.clip(np.diag(c), 0.0, None)

        # treat a class whose branch-length variance is negligible relative to the covariance scale as having no
        # variance (return zero for it) rather than dividing by ~0 and amplifying noise
        floor = 1e-10 * np.abs(c).max()
        sd = np.sqrt(np.where(var > floor, var, 0.0))
        denom = np.outer(sd, sd)

        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(denom > 0, c / denom, 0.0)

        # the standardized cross-covariance is not Cauchy-Schwarz bounded for a non-coalescent input, so clip
        return self._embed(np.clip(r, -1.0, 1.0)[1:-1, 1:-1])

    def fpmi(self) -> 'TwoSFS':
        """
        The frequency pointwise mutual information (fPMI) of two linked sites, over the segregating interior. For the
        normalized joint distribution ``p(i, j)`` of the derived-allele classes of paired polymorphic sites, entry
        ``(i, j)`` is ``log[p(i, j) / (p(i) p(j))]``, the log-ratio of the observed joint to the product of its
        marginals: positive where classes ``i`` and ``j`` co-occur more often than under independence, negative
        where they co-occur less, and zero for independent loci.

        Unlike :meth:`cov` / :meth:`corr`, fPMI is a ratio computed purely from the polymorphic interior, so it
        needs no monomorphic sites and is exactly invariant to them: a polymorphic-only (SNP) spectrum gives the
        same result as the all-sites spectrum. It is the statistic of Fenton, Rice, Novembre and Desai (2025,
        *Genetics* 229(4):iyaf023, https://doi.org/10.1093/genetics/iyaf023) for detecting departures from Kingman
        coalescence; its low-frequency associations shift toward positive under multiple-merger genealogies.

        :return: The fPMI over the interior, embedded in a full-size :class:`TwoSFS`; classes absent from the
            spectrum are returned as zero.
        :raises ValueError: If the 2-SFS is empty or non-finite, or has no polymorphic pairs.
        """
        # a two-SFS counts unordered site pairs, so symmetrize; take the polymorphic interior only (fPMI is a ratio
        # over the segregating classes and ignores the monomorphic bins, hence needs no all-sites input)
        interior = (self.data + self.data.T)[1:-1, 1:-1]

        if not np.isfinite(interior).all():
            raise ValueError('The 2-SFS contains non-finite values (NaN or inf).')

        total = interior.sum()
        if total == 0:
            raise ValueError('The 2-SFS has no polymorphic pairs, so the fPMI is undefined.')

        p = interior / total
        outer = np.outer(p.sum(axis=0), p.sum(axis=1))

        # classes that never occur (zero joint or zero marginal) are left at zero rather than log(0)
        pmi = np.zeros_like(p)
        mask = (p > 0) & (outer > 0)
        pmi[mask] = np.log(p[mask] / outer[mask])

        return self._embed(pmi)

    def fill_monomorphic(self, fill_value=np.nan) -> 'TwoSFS':
        """
        Fill the monomorphic entries (first and last row and column) of the 2-SFS.

        :param fill_value: Value to fill the monomorphic entries with.
        :return: 2-SFS
        """
        other = self.copy()

        other.data[:1, :] = fill_value
        other.data[-1:, :] = fill_value
        other.data[:, :1] = fill_value
        other.data[:, -1] = fill_value

        return other

    def plot(
            self,
            ax: 'plt.Axes' = None,
            title: str = None,
            max_abs: float = None,
            log_scale: bool = False,
            cbar_kws: Dict = None,
            show: bool = True,
    ) -> 'plt.Axes':
        """
        Plot the 2-SFS as a heatmap.

        :param ax: Axes to plot on.
        :param title: Title of the plot.
        :param max_abs: Maximum absolute value to plot.
        :param log_scale: Whether to use a logarithmic scale.
        :param cbar_kws: Keyword arguments for the color bar.
        :param show: Whether to show the plot.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm, LogNorm
        import seaborn as sns

        if self.n < 3:
            logger.warning('Nothing to plot.')
            return plt.gca()

        if cbar_kws is None:
            cbar_kws = dict(pad=0.05)

        # keep only the segregating interior (the monomorphic bins dominate a raw pair-count spectrum); this is
        # what is plotted, and the colour scale is based on it rather than on the whole matrix
        data = self.data[1:-1, 1:-1]

        # truncate data if folded
        if self.is_folded():
            data = data[:self.w - 1, :self.w - 1]

        # a raw pair-count spectrum carries mass in the monomorphic bins (row/column 0 and n); the class-resolved
        # results (cov / corr / fpmi) are embedded with those bins zeroed. Use that to choose the colour scale:
        # a sequential log scale for the heavy-tailed counts, a diverging symmetric-log scale centred at zero for
        # the (possibly signed) derived quantities.
        is_counts = np.nansum(np.abs(np.r_[self.data[[0, -1], :].ravel(), self.data[:, [0, -1]].ravel()])) > 0
        if is_counts:
            positive = data[np.isfinite(data) & (data > 0)]
            norm = LogNorm(vmin=positive.min() if positive.size else 1, vmax=np.nanmax(data) or 1)
            cmap = 'viridis'
        else:
            m = max_abs if max_abs is not None else (np.nanmax(np.abs(data)) or 1)
            norm, cmap = SymLogNorm(linthresh=m / 10, vmin=-m, vmax=m), 'PuOr_r'

        ax = sns.heatmap(data, norm=norm, cmap=cmap, cbar_kws=cbar_kws, ax=ax)

        # invert y-axis and remove ticks
        ax.invert_yaxis()
        ax.axis('square')

        if log_scale:
            ax.set_xscale('log', base=1.001)
            ax.set_yscale('log', base=1.001)

        ax.set_xticks(ax.get_yticks())
        ax.set_xticklabels([str(int(label + 1)) for label in ax.get_xticks()])
        ax.set_yticklabels([str(int(label + 1)) for label in ax.get_yticks()])

        # remove confusing color bar ticks
        ax.collections[0].colorbar.ax.tick_params(size=0)

        # add frame around plot
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_edgecolor('grey')

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return ax

    def plot_surface(
            self,
            ax: 'plt.Axes' = None,
            title: str = None,
            max_abs: float = None,
            vmin: float = None,
            vmax: float = None,
            show: bool = True,
    ) -> 'plt.Axes':
        """
        Plot the 2-SFS as a surface.

        :param ax: Axes to plot on.
        :param title: Title of the plot.
        :param max_abs: Maximum absolute value to plot.
        :param vmin: Minimum value to plot.
        :param vmax: Maximum value to plot.
        :param show: Whether to show the plot.
        :return: Axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm

        if self.n < 3:
            logger.warning('Nothing to plot.')
            return plt.gca()

        if max_abs is None:
            max_abs = self.get_max_abs() or 1

        # remove monomorphic sites
        data = self.data[1:-1, 1:-1]

        # truncate data if folded
        if self.is_folded():
            data = data[:self.w - 1, :self.w - 1]

        x = np.arange(1, data.shape[0] + 1)
        y = np.arange(1, data.shape[0] + 1)

        x_grid, y_grid = np.meshgrid(x, y)

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # vmin and vmax don't seem to work here
        ax.plot_surface(
            x_grid,
            y_grid,
            data,
            cmap='PuOr_r',
            vmin=vmin,
            vmax=vmax,
            norm=SymLogNorm(
                linthresh=max_abs / 10,
                vmin=-max_abs,
                vmax=max_abs
            )
        )

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return ax

    def mask_diagonal(self, fill_value=np.nan) -> 'TwoSFS':
        """
        Mask both the primary and secondary diagonal entries of the 2-SFS matrix.

        The primary diagonal runs from the top-left to the bottom-right,
        and the secondary diagonal runs from the top-right to the bottom-left.

        :param fill_value: The value to fill the diagonal entries with.
        :return: A new 2-SFS with both diagonals masked.
        """
        data = self.data.copy()
        np.fill_diagonal(data, fill_value)

        data = np.fliplr(data)
        np.fill_diagonal(data, fill_value)
        data = np.fliplr(data)

        return TwoSFS(data)

    def get_max_abs(self) -> float:
        """
        Get the maximum absolute entry of the 2-SFS matrix.

        :return: The maximum absolute entry.
        """
        return np.nanmax(np.abs(self.data))

    def mask_upper(self, fill_value=np.nan) -> 'TwoSFS':
        """
        Mask the upper triangular entries of the 2-SFS matrix.

        :param fill_value: The value to fill the upper triangular entries with.
        :return: A new 2-SFS with upper triangular entries masked.
        """
        data = self.copy().data

        data[np.triu(np.ones_like(data, dtype=bool), k=1)] = fill_value

        return TwoSFS(data)


class TwoLocusSFS(TwoSFS):
    """
    The two-locus site-frequency spectrum under recombination: a square matrix whose entry ``(i, j)`` is the
    expected product of the branch length subtending ``i`` samples at locus 0 and ``j`` samples at locus 1, for two
    loci separated by a given recombination rate ``r``. It interpolates between the within-tree cross-moment of the
    SFS at ``r = 0`` (fully linked) and the outer product of the marginal SFS as ``r`` tends to infinity
    (independent loci).
    """
    pass


class JointSFS(AbstractSpectrum):
    """
    A joint (multi-population) site-frequency spectrum.

    The data is a ``P``-dimensional array of shape ``(n_0 + 1, ..., n_{P-1} + 1)`` where ``P`` is the number of
    populations and entry ``(k_0, ..., k_{P-1})`` counts sites (or branch length) with ``k_p`` derived alleles in
    population ``p``. For two populations this is a 2-dimensional array (analogous to but generally rectangular,
    unlike the square :class:`TwoSFS`); for three populations it is a 3-dimensional array, and so on.
    """

    # class-level default so jsonpickle can restore spectra serialized without population names
    pop_names: List[str] = None

    def __init__(self, data: np.ndarray | list, pop_names: List[str] = None) -> None:
        """
        Construct from a data array.

        :param data: A ``P``-dimensional array.
        :param pop_names: Optional names of the ``P`` populations (one per axis), used for plot axis labels. Defaults
            to ``pop_0, ..., pop_{P-1}`` when not given.
        :raises ValueError: If the data is not at least 1-dimensional or if the number of population names does not
            match the number of axes.
        """
        data = np.asarray(data)

        if data.ndim < 1:
            raise ValueError('Data has to be at least 1-dimensional.')

        if pop_names is not None and len(pop_names) != data.ndim:
            raise ValueError(f'Expected {data.ndim} population names (one per axis), got {len(pop_names)}.')

        #: The joint SFS array.
        self.data: np.ndarray = data

        #: Names of the populations (one per axis); falls back to ``pop_0, ..., pop_{P-1}`` if not provided.
        self.pop_names: List[str] = list(pop_names) if pop_names is not None else [f'pop_{i}' for i in range(data.ndim)]

    def _names(self) -> List[str]:
        """
        The population names, resolving to ``pop_0, ..., pop_{P-1}`` for spectra restored (via jsonpickle) without
        an explicit :attr:`pop_names`.

        :return: One population name per axis.
        """
        return self.pop_names if self.pop_names is not None else [f'pop_{i}' for i in range(self.data.ndim)]

    @property
    def n_pops(self) -> int:
        """
        Number of populations (dimensions of the joint SFS).
        """
        return self.data.ndim

    def __getitem__(self, item) -> np.ndarray:
        """
        Index into the joint SFS array.

        :param item: Index.
        :return: Indexed value or sub-array.
        """
        return self.data[item]

    def __add__(self, other) -> 'JointSFS':
        """
        Add to the joint SFS.

        :param other: Another joint SFS or a scalar or array.
        :return: Joint SFS.
        """
        return JointSFS(self.data + (other.data if isinstance(other, JointSFS) else other), self.pop_names)

    def __sub__(self, other) -> 'JointSFS':
        """
        Subtract from the joint SFS.

        :param other: Another joint SFS or a scalar or array.
        :return: Joint SFS.
        """
        return JointSFS(self.data - (other.data if isinstance(other, JointSFS) else other), self.pop_names)

    def __mul__(self, other) -> 'JointSFS':
        """
        Multiply the joint SFS.

        :param other: Another joint SFS or a scalar or array.
        :return: Joint SFS.
        """
        return JointSFS(self.data * (other.data if isinstance(other, JointSFS) else other), self.pop_names)

    def __truediv__(self, other) -> 'JointSFS':
        """
        Divide the joint SFS.

        :param other: Another joint SFS or a scalar or array.
        :return: Joint SFS.
        """
        return JointSFS(self.data / (other.data if isinstance(other, JointSFS) else other), self.pop_names)

    def __pow__(self, power) -> 'JointSFS':
        """
        Raise the joint SFS to a power.

        :param power: Exponent.
        :return: Joint SFS.
        """
        return JointSFS(self.data ** power, self.pop_names)

    def marginalize(self, pops: Sequence[int]) -> 'JointSFS':
        """
        Marginalize the joint SFS onto a subset of populations by summing over the other populations. This is useful
        for example to obtain a 2-dimensional view of a higher-dimensional joint SFS.

        :param pops: The population indices to keep, in the desired axis order.
        :return: A joint SFS over the specified populations.
        :raises ValueError: If any population index is out of range.
        """
        keep = tuple(int(p) for p in pops)

        if any(p < 0 or p >= self.n_pops for p in keep):
            raise ValueError(f'Population indices must be in [0, {self.n_pops - 1}].')

        drop = tuple(i for i in range(self.n_pops) if i not in keep)

        data = self.data.sum(axis=drop) if drop else self.data

        # reorder the remaining axes (which are in ascending order) to match the requested order
        order = [sorted(keep).index(p) for p in keep]

        return JointSFS(np.transpose(data, order), [self._names()[p] for p in keep])

    def fold(self) -> 'JointSFS':
        """
        Fold the joint SFS by folding each population axis independently. Along every axis the entry with ``k``
        derived alleles is combined with its reflection at ``n_p - k``, summing the two halves into the minor
        (lower) half and zeroing the reflected upper half, generalising :meth:`Spectrum.fold` and
        :meth:`TwoSFS.fold` to an arbitrary number of populations. Note that this only makes sense for counts or
        frequencies. Folding an already folded joint SFS is a no-op.

        :return: Folded joint SFS.
        """
        data = self.data.copy()

        for axis in range(data.ndim):
            mid = data.shape[axis] // 2

            lower = tuple(slice(0, mid) if i == axis else slice(None) for i in range(data.ndim))
            upper = tuple(slice(data.shape[axis] - mid, None) if i == axis else slice(None) for i in range(data.ndim))

            # add the reflected upper half into the lower half and empty the upper half
            data[lower] += np.flip(data[upper], axis=axis)
            data[upper] = 0

        return JointSFS(data, self.pop_names)

    def is_folded(self) -> bool:
        """
        Check if the joint SFS is folded.

        :return: Whether the joint SFS is folded.
        """
        return bool(np.all(self.data == self.fold().data))

    def plot(
            self,
            pops: Tuple[int, int] = (0, 1),
            ax: 'plt.Axes' = None,
            title: str = None,
            log_scale: bool = True,
            mask_monomorphic: bool = True,
            cbar_kws: Dict = None,
            show: bool = True,
    ) -> 'plt.Axes':
        """
        Plot the joint SFS as a 2-dimensional heatmap. For more than two populations, the joint SFS is first
        marginalized onto the two requested populations (summing over the others). The colour scale is
        logarithmic by default, since the joint SFS is heavily skewed toward the low-frequency corner.

        :param pops: The two population indices to plot (y-axis, x-axis).
        :param ax: Axes to plot on.
        :param title: Title of the plot.
        :param log_scale: Whether to use a logarithmic color scale (default ``True``).
        :param mask_monomorphic: Whether to mask the monomorphic corners (all-zero and all-derived).
        :param cbar_kws: Keyword arguments for the color bar.
        :param show: Whether to show the plot.
        :return: Axes.
        :raises ValueError: If not exactly two populations are requested or the marginalized data is not 2-dimensional.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import seaborn as sns

        if len(pops) != 2:
            raise ValueError('Exactly two populations must be specified for a 2-dimensional plot.')

        # reduce to the two requested populations
        # marginalize onto the requested populations; for exactly two this still applies the requested
        # axis order (transposing when pops=(1, 0)), which drawing self untransposed would ignore
        data = self.marginalize(pops).data.astype(float).copy()

        if data.ndim != 2:
            raise ValueError('Plotting requires a 2-dimensional (marginalized) joint SFS.')

        if mask_monomorphic:
            data[0, 0] = np.nan
            data[-1, -1] = np.nan

        if cbar_kws is None:
            cbar_kws = dict(pad=0.05)

        # create a fresh 2-D axes if none is given (so we never draw onto a leftover 3-D axes from plot_surface)
        if ax is None:
            _, ax = plt.subplots()

        ax = sns.heatmap(
            data,
            norm=LogNorm() if log_scale else None,
            cmap='viridis',
            cbar_kws=cbar_kws,
            ax=ax
        )

        # put the origin at the bottom left
        ax.invert_yaxis()
        ax.set_xlabel(f'allele count {self._names()[pops[1]]}')
        ax.set_ylabel(f'allele count {self._names()[pops[0]]}')

        # square cells, a grey frame, and unobtrusive color bar ticks (as for the 2-SFS plot)
        ax.set_aspect('equal')
        ax.collections[0].colorbar.ax.tick_params(size=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('grey')

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return ax

    def plot_surface(
            self,
            pops: Tuple[int, int] = (0, 1),
            ax: 'plt.Axes' = None,
            title: str = None,
            log_scale: bool = False,
            mask_monomorphic: bool = True,
            cmap: str = 'viridis',
            show: bool = True,
    ) -> 'plt.Axes':
        """
        Plot the joint SFS of two populations as a surface, with the two allele-count axes on the horizontal
        plane and the number of sites as height. For more than two populations, the joint SFS is first
        marginalized onto the two requested populations (summing over the others).

        :param pops: The two population indices to plot (y-axis, x-axis).
        :param ax: Axes to plot on.
        :param title: Title of the plot.
        :param log_scale: Whether to use a logarithmic color scale.
        :param mask_monomorphic: Whether to mask the monomorphic corners (all-zero and all-derived).
        :param cmap: The colormap.
        :param show: Whether to show the plot.
        :return: Axes.
        :raises ValueError: If not exactly two populations are requested or the marginalized data is not 2-dimensional.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        if len(pops) != 2:
            raise ValueError('Exactly two populations must be specified for a surface plot.')

        # reduce to the two requested populations
        # marginalize onto the requested populations; for exactly two this still applies the requested
        # axis order (transposing when pops=(1, 0)), which drawing self untransposed would ignore
        data = self.marginalize(pops).data.astype(float).copy()

        if data.ndim != 2:
            raise ValueError('Plotting requires a 2-dimensional (marginalized) joint SFS.')

        if mask_monomorphic:
            data[0, 0] = np.nan
            data[-1, -1] = np.nan

        # allele-count grid (0..n_p) for each of the two populations
        x_grid, y_grid = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'})

        ax.plot_surface(x_grid, y_grid, data, cmap=cmap, norm=LogNorm() if log_scale else None)

        ax.set_xlabel(f'allele count {self._names()[pops[1]]}')
        ax.set_ylabel(f'allele count {self._names()[pops[0]]}')
        ax.set_zlabel('branch length')

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return ax


_D = TypeVar("_D", bound="_DictSpectraSerialization")


class _DictSpectraSerialization:
    """
    JSON serialization shared by the dict-backed spectra collections (:class:`JointSpectra` and :class:`TwoSpectra`),
    whose :attr:`data` maps each type to a spectrum object wrapping a numpy array. The arrays are converted to nested
    lists for encoding and back to arrays on decode. The CSV/DataFrame-backed :class:`~sfsutils.spectrum.Spectra` does not use this.
    """

    #: The spectra keyed by type.
    data: Dict[str, AbstractSpectrum]

    def to_file(self, file: str) -> None:
        """
        Save to file (in JSON format).

        :param file: File path.
        """
        with open(file, 'w') as f:
            f.write(self.to_json())

    def to_json(self) -> str:
        """
        Convert to a JSON string.

        :return: JSON string.
        """
        obj = copy.deepcopy(self)

        # convert each spectrum's array to a list
        for s in obj.data.values():
            s.data = s.data.tolist()

        return jsonpickle.encode(obj)

    @classmethod
    def from_file(cls: type[_D], file: str) -> _D:
        """
        Load from file.

        :param file: File path.
        :return: Spectra.
        """
        with open(file, 'r') as f:
            return cls.from_json(f.read())

    @classmethod
    def from_json(cls: type[_D], json: str) -> _D:
        """
        Load from a JSON string.

        :param json: JSON string.
        :return: Spectra.
        """
        obj = jsonpickle.decode(json)

        # convert each spectrum's list back to a numpy array
        for s in obj.data.values():
            s.data = np.array(s.data)

        return obj


class JointSpectra(_DictSpectraSerialization, AbstractSpectra):
    """
    A collection of joint (multi-population) site-frequency spectra keyed by type, the multi-population analogue of
    :class:`~sfsutils.spectrum.Spectra`. This is the return type of :meth:`~sfsutils.parser.Parser.parse` when the parser is given
    populations, with one :class:`JointSFS` per stratification type (or a single ``all`` type when no stratifications
    are used).
    """

    def __init__(self, data: Dict[str, Union['JointSFS', np.ndarray]], pop_names: List[str] = None) -> None:
        """
        Construct from a dictionary of joint spectra.

        :param data: Dictionary of joint spectra (or plain arrays) keyed by type.
        :param pop_names: Optional names of the populations (one per axis). Only used for entries given as plain
            arrays; entries already given as :class:`JointSFS` keep their own names.
        """
        #: The joint spectra keyed by type.
        self.data: Dict[str, JointSFS] = {
            t: v if isinstance(v, JointSFS) else JointSFS(np.asarray(v), pop_names) for t, v in data.items()
        }

    @property
    def types(self) -> List[str]:
        """
        The types.
        """
        return list(self.data.keys())

    @property
    def pop_names(self) -> List[str]:
        """
        Names of the populations (one per axis), taken from the first type.

        :raises ValueError: If the collection is empty.
        """
        if not self.data:
            raise ValueError('Empty JointSpectra has no population names.')

        return next(iter(self.data.values()))._names()

    @property
    def n_pops(self) -> int:
        """
        Number of populations (dimensions of each joint SFS).

        :raises ValueError: If the collection is empty.
        """
        if not self.data:
            raise ValueError('Empty JointSpectra has no populations.')

        return next(iter(self.data.values())).n_pops

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of each joint SFS.

        :raises ValueError: If the collection is empty.
        """
        if not self.data:
            raise ValueError('Empty JointSpectra has no shape.')

        return next(iter(self.data.values())).shape

    @property
    def all(self) -> 'JointSFS':
        """
        The ``all`` type, equal to the sum of all joint spectra.

        :raises ValueError: If the collection is empty.
        """
        if not self.data:
            raise ValueError('Empty JointSpectra has no joint spectra to sum.')

        spectra = list(self.data.values())

        return JointSFS(sum(s.data for s in spectra), spectra[0].pop_names)

    def marginalize(self, pops: Sequence[int]) -> 'JointSpectra':
        """
        Marginalize every joint SFS onto a subset of populations (see :meth:`JointSFS.marginalize`).

        :param pops: The population indices to keep, in the desired axis order.
        :return: Marginalized joint spectra.
        """
        return JointSpectra({t: s.marginalize(pops) for t, s in self.data.items()})

    def __getitem__(self, key: str) -> 'JointSFS':
        """
        Get the joint SFS for a given type.

        :param key: Type.
        :return: Joint SFS.
        """
        return self.data[key]

    def __iter__(self) -> Iterator:
        """
        Iterate over the types.

        :return: Iterator.
        """
        return iter(self.data)

    def __len__(self) -> int:
        """
        The number of types.

        :return: Number of types.
        """
        return len(self.data)

    def to_dict(self) -> Dict[str, JointSFS]:
        """
        Get the joint spectra as a dictionary keyed by type.

        :return: Dictionary of joint spectra keyed by type.
        """
        return dict(self.data)


class TwoSpectra(_DictSpectraSerialization, AbstractSpectra):
    """
    A collection of two-site (two-locus) site-frequency spectra keyed by type, the two-dimensional analogue of
    :class:`~sfsutils.spectrum.Spectra`. This is the return type of :meth:`~sfsutils.parser.Parser.parse` when the two-SFS is parsed
    with stratifications, holding one :class:`TwoSFS` per stratification type. Because the two-SFS pairs sites,
    stratified parsing counts only within-stratum pairs, so summing the per-type spectra does not in general recover
    the unstratified two-SFS (cross-stratum pairs are not counted).
    """

    def __init__(self, data: Dict[str, Union['TwoSFS', np.ndarray]]) -> None:
        """
        Construct from a dictionary of two-site spectra.

        :param data: Dictionary of two-site spectra (or plain square arrays) keyed by type.
        """
        #: The two-site spectra keyed by type.
        self.data: Dict[str, TwoSFS] = {
            t: v if isinstance(v, TwoSFS) else TwoSFS(np.asarray(v)) for t, v in data.items()
        }

    @property
    def types(self) -> List[str]:
        """
        The types.
        """
        return list(self.data.keys())

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of each two-site SFS.

        :raises ValueError: If the collection is empty.
        """
        if not self.data:
            raise ValueError('Empty TwoSpectra has no shape.')

        return next(iter(self.data.values())).shape

    @property
    def all(self) -> 'TwoSFS':
        """
        The ``all`` type, equal to the sum of the per-type two-site spectra (the within-stratum pairs pooled over
        strata; cross-stratum pairs remain uncounted).

        :raises ValueError: If the collection is empty.
        """
        if not self.data:
            raise ValueError('Empty TwoSpectra has no spectra to sum.')

        return TwoSFS(sum(s.data for s in self.data.values()))

    def __getitem__(self, key: str) -> 'TwoSFS':
        """
        Get the two-site SFS for a given type.

        :param key: Type.
        :return: Two-site SFS.
        """
        return self.data[key]

    def __iter__(self) -> Iterator:
        """
        Iterate over the types.

        :return: Iterator.
        """
        return iter(self.data)

    def __len__(self) -> int:
        """
        The number of types.

        :return: Number of types.
        """
        return len(self.data)

    def to_dict(self) -> Dict[str, 'TwoSFS']:
        """
        Get the two-site spectra as a dictionary keyed by type.

        :return: Dictionary of two-site spectra keyed by type.
        """
        return dict(self.data)


def parse_polydfe_sfs_config(file: str) -> dict:
    """
    Parse frequency spectra and mutational target site from
    polyDFE configuration file.

    :param file: File name
    :return: Dictionary
    """
    df = pd.read_csv(file, header=None, comment='#')

    # parse number of spectra and sample size
    n_neut, n_sel, n = np.array(df.iloc[0][0].split()).astype(int)

    # issue notice about number of spectra and sample size
    logger.info(f'Parsing {n_neut} neutral and {n_sel} selected SFS with '
                f'a sample size of {n}.')

    # issue notice that variable mutation rates are not modelled
    if n_neut > 1 or n_sel > 1:
        logger.info('Note that variable mutation rates are not modelled here. '
                    'The parsed spectra are thus merged together.')

    def to_spectrum(data: np.ndarray) -> Spectrum:
        """
        Parse spectrum, number of mutational target sites, and the optional
        divergence counts together with their separate mutational target size.

        :param data: Spectrum data
        :return: Spectrum object
        """
        # iterate over spectra and merge them as we do not
        # support variable mutation rates
        data_merged = data.sum(axis=0)

        # polymorphic counts
        polymorphic = list(data_merged[:n - 1])

        # parse number mutational target sites for ingroup
        n_sites = float(data_merged[n - 1])

        # parse optional divergence counts
        n_div = float(data_merged[n]) if n < data_merged.shape[0] else 0

        # parse optional number of mutational target sites for divergence counts
        n_sites_div = float(data_merged[n + 1]) if n + 1 < data_merged.shape[0] else None

        return Spectrum.from_polydfe(polymorphic, n_sites=n_sites, n_div=n_div), n_sites_div

    # iterate over spectra and merge them as we do not
    # support variable mutation rates
    data_neut = np.array([df.iloc[i][0].split() for i in range(1, n_neut + 1)], dtype=float)
    sfs_neut, n_sites_div_neut = to_spectrum(data_neut)

    # iterate over spectra and merge them as we do not
    # support variable mutation rates
    data_sel = np.array([df.iloc[i][0].split() for i in range(n_neut + 1, n_neut + n_sel + 1)], dtype=float)
    sfs_sel, n_sites_div_sel = to_spectrum(data_sel)

    return dict(
        sfs_neut=sfs_neut,
        sfs_sel=sfs_sel,
        n_sites_div_neut=n_sites_div_neut,
        n_sites_div_sel=n_sites_div_sel
    )
