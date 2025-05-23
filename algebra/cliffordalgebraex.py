# This file extends the functionality of the original CliffordAlgebra class
# Original code from: https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks

import math
import functools
import numpy as np
import torch

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
    

    @functools.cached_property
    def qt_geometric_product_paths(self):
        # Sum up the results of multiplications (mod 4): [4, n+1, n+1]
        qt_results = torch.zeros((4, self.dim + 1, self.dim + 1), dtype=bool)
        for grade in range(self.dim + 1):
            qt_results[grade % 4, :, :] += self.geometric_product_paths[grade, :, :]

        # Sum up the rows in multiplication table (mod 4): [4, 4, n+1]
        qt_sum_rows = torch.zeros((4, 4, self.dim + 1), dtype=bool)
        for grade in range(self.dim + 1):
            qt_sum_rows[:, grade % 4, :] += qt_results[:, grade, :]

        # Sum up the columns in multiplication table (mod 4): [4, 4, 4]
        qt_sum_cols = torch.zeros((4, 4, 4), dtype=bool)
        for grade in range(self.dim + 1):
            qt_sum_cols[:, :, grade % 4] += qt_sum_rows[:, :, grade]

        return qt_sum_cols
    

    @functools.cached_property
    def qt_to_list(self):
        """
        Get list of 4 lists with ids of basis elements for 4 qt in slices
        """
        return [self.grade_to_slice[::4], self.grade_to_slice[1::4], self.grade_to_slice[2::4], self.grade_to_slice[3::4]]


    def get_qt(self, mv: torch.Tensor, qt: int) -> torch.Tensor:
        """
        Project a multivector onto a subspaces of fixed qt
        """
        qt_list = self.qt_to_list[qt]
        indices = [(s.start, s.stop) for s in qt_list]
        new_slices = [slice(start.item(), stop.item()) for start, stop in indices]
        projection = []
        for slice_ in new_slices:
            projection.append(mv[..., slice_])
        return torch.cat(projection, dim=2)


    @functools.cached_property
    def qt_to_index(self):
        """
        Get list of tensors with ids of basis elements for 4 qt 
        """
        result = []
        for slices in self.qt_to_list:
            indices = []
            for s in slices:
                start = s.start.item()
                stop = s.stop.item()
                indices.extend(range(start, stop))
            result.append(torch.tensor(indices))
        return result


    def norms_qt(self, mv):
        return [
                self.norm(self.get_qt(mv, qt), blades=self.qt_to_index[qt])
                for qt in range(4)
            ]