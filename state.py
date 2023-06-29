from typing import TYPE_CHECKING, Optional, Union, cast, Sequence
from typing_extensions import TypeAlias

#import numpy as np
import cupy as cp

StateVectorType: TypeAlias = "npt.NDArray[cp.cfloat]"

class QuantumState:
    def __init__(
        self,
        n_qubits: int,
        vector: Optional[Union[StateVectorType, "npt.ArrayLike"]] = None,
    ) -> None:
        self._dim = 2**n_qubits
        self._vector: StateVectorType
        if vector is None:
            self._vector = cast(StateVectorType, cp.zeros(self._dim))
            self._vector[0] = 1.0
        else:
            vector = cp.asarray(vector, dtype=cp.cfloat)
            if len(vector) != self._dim:
                raise ValueError(f"The dimension of vector must be {self._dim}.")
            self._vector = vector

    @property
    def vector(self) -> StateVectorType:
        return self._vector

    # TODO: just simple only single qubit and target only!
    def apply(self, targets: int, matrix):
        # only 1 qubit
        qubits = []
        qubits.append(targets)
        masks = self.mask_vec(qubits)
        qsize = 1
        for i in range(self._dim >> qsize):
            indices = self.indices_vec(i, qubits, masks)
            # TODO: GPUメモリ上で処理したい
            values = [self.vector[i] for i in indices]
            values = cp.asarray(values)
            new_values = matrix.dot(values)
            for (i, nv) in zip(indices, new_values):
                self._vector[i] = nv

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