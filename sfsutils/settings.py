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

    #: Whether to use parallel processing. ``False`` disables it package-wide, including for calls
    #: that request it; ``None`` and ``True`` defer to the caller's own ``parallelize`` argument.
    parallelize: bool = None
