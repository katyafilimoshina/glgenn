# This file extends the functionality of the original CliffordAlgebra class
# Original code from: https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks
# Modifications by katyafilimoshina on 05.2024

import math
import functools
import numpy as np

from .cliffordalgebra import CliffordAlgebra

class CliffordAlgebraQT(CliffordAlgebra):
    def __init__(self, metric):
        super().__init__(metric)

        # Dimensions of the subspaces of quaternion types 0, 1, 2, 3
        self.dim_1 = int(2**(self.dim-2) + 2**(self.dim/2 -1) * math.sin(math.pi * self.dim / 4))
        self.dim_2 = int(2**(self.dim-2) - 2**(self.dim/2 -1) * math.cos(math.pi * self.dim / 4))
        self.dim_0 = int(2**(self.dim-1) - self.dim_2)
        self.dim_3 = int(2**(self.dim-1) - self.dim_1)

    
    @functools.cached_property
    def weights_permutation(self):
        qt_0, qt_1, qt_2, qt_3 = 0, 0, 0, 0
        arange = np.arange(2 ** self.dim)
        permutation = []

        for grade in range(self.n_subspaces):
            for elem in range(self.subspaces[grade]):
                if grade % 4 == 0:
                    permutation.append(arange[qt_0])
                    qt_0 += 1
                elif grade % 4 == 1:
                    permutation.append(self.dim_0 + arange[qt_1])
                    qt_1 += 1
                elif grade % 4 == 2:
                    permutation.append(self.dim_0 + self.dim_1 + arange[qt_2])
                    qt_2 += 1
                elif grade % 4 == 3:
                    permutation.append(self.dim_0 + self.dim_1 + self.dim_2 + arange[qt_3])
                    qt_3 += 1
        
        return permutation