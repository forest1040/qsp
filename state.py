from typing import TYPE_CHECKING, Optional, Union, cast, Sequence
from typing_extensions import TypeAlias

import numpy as np
import random
import cmath

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
        else:
            vector = np.asarray(vector, dtype=np.cfloat)
            if len(vector) != self._dim:
                raise ValueError(f"The dimension of vector must be {self._dim}.")
            self._vector = vector

    @property
    def vector(self) -> StateVectorType:
        return self._vector
    
    def copy(self) -> "QuantumState":
        return QuantumState(self.n_qubits, self._vector.copy())

    def set_computatuional_basis(self, basis: int):
        for i in range(self._dim):
            self._vector[i] = 0.0
        self._vector[basis] = 1.0
    
    def set_zero_state(self):
        self._vector = np.zeros(self._dim, dtype=complex)

    def set_random_state(self):
        state = np.zeros(self._dim, dtype=complex)
        for i in range(self._dim):
            state[i] = random.random() + random.random() * 1j
        norm = np.linalg.norm(state)
        self._vector = state / norm

    def set_state(self, vector: StateVectorType):
        self._vector = vector

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

    def apply_cnot_gate(self, control: int, target: int):
        if target == control:
            return
        n = self.n_qubits
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
            self._vector[i], self._vector[j] = self._vector[j], self._vector[i]

    def apply_swap_gate(self, target1: int, target2: int):
        if target1 == target2:
            return
        if target1 > target2:
            target1, target2 = target2, target1
        mask0 = 1 << target1
        mask1 = 1 << target2
        upper_bit_range = 1 << (self.n_qubits - target2 - 1)
        middle_bit_range = 1 << (target2 - target1 - 1)
        lower_bit_range = 1 << target1
        for ub in range(upper_bit_range):
            for mb in range(middle_bit_range):
                for lb in range(lower_bit_range):
                    i = (ub << (target2+1)) | (mb << (target1+1)) | lb
                    self._vector[i | mask0], self._vector[i | mask1] = self._vector[i | mask1], self._vector[i | mask0]

    def apply_densematrix(self, target_list: int, matrix):
        num_target = len(target_list)
        num_outer = self.n_qubits - num_target
        target_mask = 0
        for idx in target_list: target_mask |= (1 << idx)
        state_idx = [[0 for j in range(1 << num_target)] for i in range(1 << num_outer)]
        state_updated = [[0 for j in range(1 << num_target)] for i in range(1 << num_outer)]
        for it0 in range(1 << num_outer):
            for it1 in range(1 << num_target):
                idx = it0
                target_idx = 0
                while target_idx < num_target:
                    target = target_list[target_idx]
                    value = it1 >> target_idx
                    target_idx += 1
                    upper_mask = ((1 << (self.n_qubits-target)) - 1) << target
                    lower_mask = (1 << target) - 1
                    idx = ((idx & upper_mask) << 1) | (value << target) | (idx & lower_mask)
                state_idx[it0][it1] = idx
        for it0 in range(1 << num_outer):
            for it1 in range(1 << num_target):
                for it2 in range(1 << num_target):
                    state_updated[it0][it1] += matrix[it1][it2] * self._vector[state_idx[it0][it2]]
        for it0 in range(1 << num_outer):
            for it1 in range(1 << num_target):
                self._vector[state_idx[it0][it1]] = state_updated[it0][it1]

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