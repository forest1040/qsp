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
    print(state0.vector)

    # ============= 量子ゲート ===============
    rx_gate = RXGate(0, pi)
    cnot_gate = CNOTGate(0, 2)

    # ============= 量子回路 =================
    circuit = QuantumCircuit(n)
    # circuit.add_gate(rx_gate)
    # circuit.update_quantum_state(state1)
    circuit.add_gate(cnot_gate)
    circuit.update_quantum_state(state2)
    print(circuit)
    print(f'state0: {state0.vector}')
    # print(f'state1: {state1.vector}')
    print(f'state2: {state2.vector}')

main()
