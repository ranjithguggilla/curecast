"""curecast: strength prediction and low-clinker mix optimisation for concrete."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("curecast")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.1.0"

__all__ = ["__version__"]
