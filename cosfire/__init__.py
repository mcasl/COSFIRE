from importlib.metadata import PackageNotFoundError, version

from .base import (Cosfire,
                   )
from . import base

try:
    __version__ = version('cosfire')
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = '0+unknown'

__all__ = ['Cosfire', 'base', '__version__']
