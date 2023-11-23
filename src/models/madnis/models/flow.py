from typing import Dict, Callable, List, Optional, Union
from functools import partial

import numpy as np
import torch

from ..mappings.base import Mapping, ChainedMapping
from ..mappings.split import ConditionalSplit, ConditionalPseudoSplit
from ..mappings.coupling.base import CouplingBlock
from ..mappings.coupling.linear import AffineCoupling
from ..mappings.nonlinearities import Sigmoid, Logit
from ..mappings.identity import Identity
from ..mappings import permutation as perm
from ..distributions.base import MappedDistribution
from ..distributions.uniform import StandardUniform
from ..distributions.normal import StandardNormal
from ..models.mlp import StackedMLP

class FlowMapping(ChainedMapping):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        n_blocks: int,
        subnet_meta: Dict,
        subnet_constructor: Callable = None,
        coupling_block: CouplingBlock = AffineCoupling,
        coupling_kwargs: Dict = {},
        permutations: Optional[str] = "soft",
        hypercube_latent: bool = False,
        hypercube_couplings: bool = False,
        hypercube_permutations: bool = False,
        hypercube_data: bool = False
    ):
        mappings = []
        def map_space(is_hypercube, target_hypercube):
            if not is_hypercube and target_hypercube:
                mappings.append(Sigmoid(dims_in, dims_c))
            elif is_hypercube and not target_hypercube:
                mappings.append(Logit(dims_in, dims_c))
            return target_hypercube

        is_hypercube = hypercube_data

        if permutations == "log":
            n_perms = int(np.ceil(np.log2(dims_in)))
            # use at least n_perms blocks
            n_blocks = int(2 * n_perms)
            splitting_masks = torch.tensor([
                [int(i) for i in np.binary_repr(i, n_perms)] for i in range(dims_in)
            ]).flip(dims=(1,)).bool().t().repeat_interleave(2, dim=0)
            splitting_masks[1::2, :] ^= True
        elif permutations == "exchange":
            splitting_masks = torch.cat((
                torch.ones(dims_in // 2, dtype=torch.bool),
                torch.zeros(dims_in - dims_in // 2, dtype=torch.bool)
            ))[None,:].repeat((n_blocks, 1))
            splitting_masks[1::2, :] ^= True
        else:
            splitting_masks = [None] * n_blocks

        first_block = True

        perm_class = {
            "random": perm.PermuteRandom,
            "soft": perm.PermuteSoft,
            "softlearn": perm.PermuteSoftLearn
        }.get(permutations)
        for i in range(n_blocks):
            if perm_class is not None and not first_block:
                is_hypercube = map_space(is_hypercube, hypercube_permutations)
                mappings.append(perm_class(dims_in, dims_c))

            is_hypercube = map_space(is_hypercube, hypercube_couplings)
            mappings.append(coupling_block(
                dims_in,
                dims_c,
                subnet_meta = subnet_meta,
                subnet_constructor = subnet_constructor,
                splitting_mask = splitting_masks[i],
                **coupling_kwargs
            ))
            first_block = False
        map_space(is_hypercube, hypercube_latent)

        super().__init__(dims_in, dims_c, mappings)
