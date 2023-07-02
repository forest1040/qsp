from typing import TYPE_CHECKING, Optional, Union, cast
from typing_extensions import TypeAlias

import random
from math import pi

import time

from state import QuantumState
from gate import RXGate
from circuit import QuantumCircuit

MAX_COUNT = 1000
DEPTH = 10
N = 4

def execute(n: int, batch_size: int):
    state = QuantumState(n, batch_size)
    circuit = QuantumCircuit(n)
    for _ in range(DEPTH):
        for i in range(n):
            t = random.random() * pi
            rx_gate = RXGate(i, t)
            circuit.add_gate(rx_gate)
    circuit.update_quantum_state(state)
    #print(circuit)
    #print(state._vector)

def main():
    #batch_size = [1, 10, 100, 1000]
    #batch_size = [100, 1000, 10000]
    batch_size = [100, 1000]
    for b in batch_size:
        #print(f"batch_size: {b}")
        start = time.time()
        cnt = MAX_COUNT // b
        for c in range(cnt):
            #print("counter:", c)
            execute(N, b)
        print(f"batch_size: {b}, {time.time() - start}")
        #print(time.time() - start)


main()

# TODO:
# QuantumStateは、QuantumCircuit内に持たせた方がよいかも。

