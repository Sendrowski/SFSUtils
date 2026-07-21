"""
JSON handlers for SFS objects.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import logging

import numpy as np
import pandas as pd
from jsonpickle.handlers import BaseHandler

from .spectrum import Spectrum, Spectra

# configure logger
logger = logging.getLogger('sfsutils')


class NumpyArrayHandler(BaseHandler):
    """
    Handler for numpy arrays.
    """

    def flatten(self, x: np.ndarray, data: dict) -> dict:
        """
        Convert Spectrum to dict.

        :param x: Numpy array
        :param data: Dictionary
        :return: Simplified dictionary
        """
        return data | dict(data=x.tolist(), dtype=str(x.dtype))

    def restore(self, data: dict) -> np.ndarray:
        """
        Restore Spectrum.

        :param data: Dictionary
        :return: Numpy array
        """
        # fall back to inferred dtype for old-style payloads lacking the dtype field
        return np.array(data['data'], dtype=data.get('dtype'))


class SpectrumHandler(BaseHandler):
    """
    Handler for spectrum objects.
    """

    def flatten(self, sfs: Spectrum, data: dict) -> dict:
        """
        Convert Spectrum to dict.

        :param sfs: Spectrum object
        :param data: Dictionary
        :return: Simplified dictionary
        """
        return data | dict(data=sfs.to_list())

    def restore(self, data: dict) -> Spectrum:
        """
        Restore Spectrum.

        :param data: Dictionary
        :return: Spectrum object
        """
        return Spectrum.from_list(data['data'])


class SpectraHandler(BaseHandler):
    """
    Handler for spectra objects.
    """

    def flatten(self, sfs: Spectra, data: dict) -> dict:
        """
        Convert Spectra to dict.

        :param sfs: Spectra object
        :param data: Dictionary
        :return: Simplified dictionary
        """
        return data | dict(data=sfs.to_dict())

    def restore(self, data: dict) -> Spectra:
        """
        Restore Spectra.

        :param data: Dictionary
        :return: Spectra object
        """
        return Spectra.from_dict(data['data'])


class DataframeHandler(BaseHandler):
    """
    There were also problems with dataframes, hence the custom handler.
    """

    def flatten(self, df: pd.DataFrame, data: dict) -> dict:
        """
        Convert dataframe to dict.

        :param df: Dataframe
        :param data: Dictionary
        :return: Simplified dictionary
        """
        # the 'split' orient stores the index as a JSON list, so an integer index survives the round-trip
        # (the default column->index->value mapping turns integer index labels into strings)
        return data | dict(data=df.to_dict(orient='split'))

    def restore(self, data: dict) -> pd.DataFrame:
        """
        Restore dataframe.

        :param data: Dictionary
        :return: Dataframe
        """
        payload = data['data']
        # 'split' payloads carry index/columns/data; fall back to the legacy column->index mapping for
        # dataframes serialized before the switch to 'split'
        if isinstance(payload, dict) and {'index', 'columns', 'data'} <= set(payload):
            return pd.DataFrame(
                data=payload['data'],
                index=self._restore_axis(payload['index']),
                columns=self._restore_axis(payload['columns'])
            )
        return pd.DataFrame(payload)

    @staticmethod
    def _restore_axis(labels: list) -> pd.Index:
        """
        Rebuild an axis from its serialized labels. JSON turns the tuples of a ``MultiIndex`` into lists,
        which a plain index would keep as unhashable list labels.

        :param labels: Serialized axis labels.
        :return: The restored index.
        """
        if labels and all(isinstance(label, list) for label in labels):
            return pd.MultiIndex.from_tuples([tuple(label) for label in labels])

        return pd.Index(labels)
