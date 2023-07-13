from typing import TYPE_CHECKING, Optional, Union, cast, Sequence
from typing_extensions import TypeAlias

import numpy as np

StateVectorType: TypeAlias = "npt.NDArray[np.cfloat]"

class QuantumState:
    def __init__(
        self,
        n_qubits: int,
        vector: Optional[Union[StateVectorType, "npt.ArrayLike"]] = None,
    ) -> None:
        self._dim = 2**n_qubits
        self._vector: StateVectorType
        self.n_qubits = n_qubits
        if vector is None:
            self._vector = cast(StateVectorType, np.zeros(self._dim))
            self._vector[0] = 1.0
            self._vector[1] = 2.0
            self._vector[2] = 3.0
            self._vector[3] = 4.0
            self._vector[4] = 5.0
            self._vector[5] = 6.0
            self._vector[6] = 7.0
            self._vector[7] = 8.0
        else:
            vector = np.asarray(vector, dtype=np.cfloat)
            if len(vector) != self._dim:
                raise ValueError(f"The dimension of vector must be {self._dim}.")
            self._vector = vector

    @property
    def vector(self) -> StateVectorType:
        return self._vector

    # TODO: just simple only single qubit and target only!
    def apply(self, targets: Sequence[int], matrix):
        qubits = []
        qubits.append(targets)
        masks = self.mask_vec(qubits)
        qsize = 1
        for i in range(self._dim >> qsize):
            indices = self.indices_vec(i, qubits, masks)
            values = [self.vector[i] for i in indices]
            new_values = matrix.dot(values)
            for (i, nv) in zip(indices, new_values):
                self._vector[i] = nv

    def apply_cnot_gate(self, n, control: int, target: int):
        if target == control:
            return
        higher_mask = 0
        middle_mask = 0
        lower_mask = 0
        if target > control:
            higher_mask = (1 << (n-2)) - (1 << (target-1))
            middle_mask = (1 << (target-1)) - (1 << control)
            lower_mask = (1 << control) - 1
        else:
            higher_mask = (1 << (n-2)) - (1 << (control-1))
            middle_mask = (1 << (control-1)) - (1 << target)
            lower_mask = (1 << target) - 1
        for it in range(1 << (n-2)):
            i = ((it & higher_mask) << 2) | ((it & middle_mask) << 1) | (it & lower_mask) | (1 << control)
            j = i | (1 << target)
            # print(f'n:{n}, it: {it}, i: {i}, j: {j}')
            self._vector[i], self._vector[j] = self._vector[j], self._vector[i]

    def apply_swap_gate(self, target1: int, target2, matrix):
        if target1 == target2:
            return
        qubits = []
        qubits.append(target1)
        qubits.append(target2)
        qubits.sort()
        masks = self.mask_vec(qubits)
        qsize = 2
        for i in range(self._dim >> qsize):
            indices = self.indices_vec(i, qubits, masks)
            values = [self.vector[i] for i in indices]
            new_values = matrix.dot(values)
            for (i, nv) in zip(indices, new_values):
                self._vector[i] = nv

    def mask_vec(self, qubits: Sequence[int]):
        if 1 <= len(qubits) <= 2:
            min_qubit_mask = 1 << qubits[0]
            max_qubit_mask = 1 << qubits[0]
            mask_low = min_qubit_mask - 1
            mask_high = ~(max_qubit_mask - 1)
        return [max_qubit_mask, mask_low, mask_high]

    def indices_vec(self, index: int, qubits_tgt: Sequence[int], masks: Sequence[int], qubits_ctrl: Optional[Sequence[int]] = None):
        mask, mask_low, mask_high = masks[0], masks[1], masks[2]
        res = []
        qubits = qubits_tgt
        qubits.sort()
        if len(qubits) == 1:
            if qubits_ctrl is None:
                basis_0 = (index & mask_low) + ((index & mask_high) << len(qubits))
                basis_1 = basis_0 + mask
                res = [basis_0, basis_1]
            else:
                controle_mask = 1 << qubits_ctrl[0]
                qsize = len(qubits_ctrl) + len(qubits_tgt)
                basis_0 = (index & mask_low) + ((index & mask_high) << qsize) + controle_mask
                target_mask = 1 << qubits_tgt[0]
                basis_1 = basis_0 + target_mask
                res = [basis_0, basis_1]
        elif len(qubits) == 2:
            basis_0 = (index & mask_low) + ((index & mask_high) << len(qubits))
            target_mask1 = 1 << qubits[1]
            target_mask2 = 1 << qubits[0]
            basis_1 = basis_0 + target_mask1
            basis_2 = basis_0 + target_mask2
            basis_3 = basis_1 + target_mask2
            res = [basis_0, basis_1, basis_2, basis_3]
        return res