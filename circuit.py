import typing
from typing import cast, Any, Callable, Dict, Optional, Tuple, Type

from gate import Gate
from state import QuantumState

class QuantumCircuit:
    def __init__(self, n_qubits: int):
        self.gates = []
        self.n_qubits = n_qubits

    def add_gate(self, gate: Gate):
        self.gates.append(gate)

    def update_quantum_state(self, state: QuantumState):
        for gate in self.gates:
            gate.update_quantum_state(state)
    
