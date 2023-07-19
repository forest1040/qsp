from typing import TYPE_CHECKING, Optional, Union, cast
from typing_extensions import TypeAlias

import random
from math import pi

import numpy as np

from state import QuantumState
from gate import *
from circuit import QuantumCircuit

def main():
    # ============= 量子状態 =================
    n = 3
    state0 = QuantumState(n)
    state1 = QuantumState(n)
    state2 = QuantumState(n)
    print(f'init: {state0._vector}')
    state0.set_state(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=complex))
    state1 = state0.copy()

    # ============= 量子ゲート ===============
    rx_gate = RXGate(0, pi)
    h_gate = HGate(0)
    cnot_gate = CNOTGate(0, 2)
    swap_gate = SWAPGate(1, 2)
    dense_matrix = DenseMatrix([2], np.array([[2, 1], [3, 4]], dtype=complex))

    # ============= 量子回路 =================
    circuit1 = QuantumCircuit(n)
    circuit1.add_gate(dense_matrix)
    circuit1.update_quantum_state(state1)

    circuit2 = QuantumCircuit(n)
    circuit2.add_gate(swap_gate)
    circuit2.update_quantum_state(state2)
    print(circuit1)
    # print(circuit2)
    print(f'state0: {state0.vector}')
    print(f'state1: {state1.vector}')
    # print(f'state2: {state2.vector}')

    # ============= 量子状態 =================
    state0.set_zero_state()
    print(f'zerostate: {state0.vector}')
    state0.set_random_state()
    print(f'randomstate: {state0.vector}')
    for i in range(8):
        state0.set_computatuional_basis(i)
        print(f'computationalbasis{i}: {state0.vector}')

main()
