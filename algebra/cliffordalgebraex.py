# This file extends the functionality of the original CliffordAlgebra class
# Original code from: https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks
# Modifications by katyafilimoshina on 05.2024

import math
from .cliffordalgebra import CliffordAlgebra

class CliffordAlgebraQT(CliffordAlgebra):
    def __init__(self):
        super().__init__()

        # Dimensions of the subspaces of quaternion types 0, 1, 2, 3
        self.dim_1 = int(2**(self.dim-2) + 2**(self.dim/2 -1) * math.sin(math.pi * self.dim / 4))
        self.dim_2 = int(2**(self.dim-2) - 2**(self.dim/2 -1) * math.cos(math.pi * self.dim / 4))
        self.dim_0 = int(2**(self.dim-1) - self.dim_2)
        self.dim_3 = int(2**(self.dim-1) - self.dim_1)