from typing import TYPE_CHECKING, Optional, Union, cast, Sequence
from typing_extensions import TypeAlias

import numpy as cp
#import cupy as cp

StateVectorType: TypeAlias = "npt.NDArray[cp.cfloat]"

class QuantumState:
    def __init__(
        self,
        n_qubits: int,
        batch_size: int,
    ) -> None:
        self._dim = 2**n_qubits
        self._batch_size = batch_size
        data = cast(StateVectorType, cp.zeros(self._dim))
        data[0] = 1.0
        datas = []
        for i in range(batch_size):
            datas.append(data.copy())
        self._vector = cp.asarray(datas)

    # @property
    # def vector(self) -> StateVectorType:
    #     return self._vector

    # TODO: just simple only single qubit and target only!
    def apply(self, targets: int, matrix):
        # only 1 qubit
        qubits = []
        qubits.append(targets)
        masks = self.mask_vec(qubits)
        qsize = 1
        for i in range(self._dim >> qsize):
            indices = self.indices_vec(i, qubits, masks)
            #print("indices:", indices)
            values = []
            matrixs = []
            for bi in range(self._batch_size):
                v = [self._vector[bi][j] for j in indices]
                values.append(v)
                matrixs.append(matrix.copy())

            values = cp.asarray(values)
            #print("values.shape:", values.shape)
            matrixs = cp.asarray(matrixs)
            #print("matrixs.shape:", matrixs.shape)

            new_values = cp.einsum('ijk, ij->ik', matrixs, values)
            #print("new_values.shape:", new_values.shape)
            for bi in range(self._batch_size):
                for (j, nv) in zip(indices, new_values[bi]):
                    self._vector[bi][j] = nv

    def mask_vec(self, qubits: Sequence[int]):
        # only 1 qubit
        min_qubit_mask = 1 << qubits[0]
        max_qubit_mask = 1 << qubits[0]
        mask_low = min_qubit_mask - 1
        mask_high = ~(max_qubit_mask - 1)
        return [max_qubit_mask, mask_low, mask_high]

    def indices_vec(self, index: int, qubits: Sequence[int], masks: Sequence[int]):
        # only 1 qubit
        mask, mask_low, mask_high = masks[0], masks[1], masks[2]
        basis_0 = (index & mask_low) + ((index & mask_high) << len(qubits))
        basis_1 = basis_0 + mask
        return [basis_0, basis_1]
