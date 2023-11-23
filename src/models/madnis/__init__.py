r"""
==========================================
                   _     __ _____  __    
   /\/\   __ _  __| | /\ \ \\_   \/ _\    
  /    \ / _` |/ _` |/  \/ / / /\/\ \    
 / /\/\ \ (_| | (_| / /\  /\/ /_  _\ \   
 \/    \/\__,_|\__,_\_\ \/\____/  \__/ 
 
==========================================

Machine Learning for neural multi-channel 
importance sampling in MadGraph.
Modules to construct machine-learning based
Monte Carlo integrator using PyTorch.

"""
from . import distributions
from . import mappings
from . import models

__all__ = ["distributions", "mappings", "models"]
