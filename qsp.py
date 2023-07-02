from typing import TYPE_CHECKING, Optional, Union, cast
from typing_extensions import TypeAlias

import random
from math import pi

from state import QuantumState
from gate import RXGate
from circuit import QuantumCircuit

def main():
    BATCH_SIZE = 10
    # ============= 量子状態 =================
    n = 4
    state = QuantumState(n, BATCH_SIZE)
    print(state._vector)

    # ============= 量子ゲート ===============
    t = random.random() * pi
    rx_gate = RXGate(0, t)
    print(rx_gate)

    # ============= 量子回路 =================
    circuit = QuantumCircuit(n)
    circuit.add_gate(rx_gate)
    circuit.update_quantum_state(state)
    print(circuit)
    print(state._vector)

main()

# TODO:
# QuantumStateは、QuantumCircuit内に持たせた方がよいかも。

