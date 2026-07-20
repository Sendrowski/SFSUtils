"""
Package-wide settings
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-09-12"


class Settings:
    """
    Class that holds package-wide settings
    """
    #: Whether to disable the progress bar.
    disable_pbar: bool = False

    #: Global override for parallel processing. ``False`` disables it everywhere, overriding a
    #: per-call request; ``None`` defers to the caller's own ``parallelize`` argument.
    parallelize: bool = None
