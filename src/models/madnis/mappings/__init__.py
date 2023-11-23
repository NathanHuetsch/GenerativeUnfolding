""" Mappings for the MadNIS framework """

# Load the subfolders
from . import functional

# Import the base class first
from .base import *

# Then all inheriting modules
from .identity import *
from .linear import *
from .nonlinearities import *
from .permutation import *
from .split import *

__all__ = [
    "Mapping",
    "InverseMapping",
    "ChainedMapping",
    "Identity",
    "LinearMapping",
    "Sigmoid",
    "Logit",
    "CauchyCDF",
    "CauchyCDFInverse",
    "NormalCDF",
    "NormalCDFInverse",
    "Permutation",
    "PermuteExchange",
    "PermuteRandom",
    "PermuteSoft",
    "PermuteSoftLearn",
    "ConditionalSplit",
    "DimensionalSplit",
]
