import cmath
import math
from typing import Any, cast, Callable, Iterable, Iterator, List, NoReturn, Optional, Tuple, Type, TypeVar, Union

import numpy as np

from state import QuantumState

class Gate:
    @property
    def n_qargs(self) -> int:
        raise NotImplementedError()

    def dagger(self) -> 'Gate':
        raise NotImplementedError()

    def matrix(self) -> np.ndarray:
        raise NotImplementedError()
    
    def update_quantum_state(self, state: QuantumState):
        raise NotImplementedError()

class OneQubitGate(Gate):
    u_params: Optional[Tuple[float, float, float, float]]

    @property
    def n_qargs(self) -> int:
        return 1
    
class TwoQubitGate(Gate):
    u_params: Optional[Tuple[float, float, float, float, float, float]]

    @property
    def n_qargs(self) -> int:
        return 2

class RXGate(OneQubitGate):
    """Rotate-X gate"""
    lowername = "rx"

    def __init__(self, targets: int, theta: float):
        self.targets = targets
        # TODO: control
        self.theta = theta
        if isinstance(theta, float):
            self.u_params = (theta, -math.pi / 2.0, math.pi / 2.0, 0.0)
        else:
            self.u_params = None

    @classmethod
    def create(cls,
               targets: int,
               params: tuple,
               options: Optional[dict] = None) -> 'RXGate':
        if options:
            raise ValueError(f"{cls.__name__} doesn't take options")
        return cls(targets, params[0])

    def dagger(self):
        return RXGate(self.targets, -self.theta)

    def matrix(self):
        t = self.theta * 0.5
        a = math.cos(t)
        b = -1j * math.sin(t)
        return np.array([[a, b], [b, a]], dtype=complex)

    def update_quantum_state(self, state: QuantumState):
        state.apply(self.targets, self.matrix())

class CNOTGate(TwoQubitGate):
    """Controlled-NOT gate"""
    lowername = "cnot"

    def __init__(self, controls: int, targets: int):
        self.controls = controls
        self.targets = targets
        self.u_params = None

    @classmethod
    def create(cls,
               targets: int,
               params: tuple,
               options: Optional[dict] = None) -> 'CNOTGate':
        if options:
            raise ValueError(f"{cls.__name__} doesn't take options")
        return cls(targets[0], targets[1])

    def dagger(self):
        return CNOTGate(self.controls, self.targets)

    def matrix(self):
        return np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0],
                         [0.0, 0.0, 1.0, 0.0]], dtype=complex)

    def update_quantum_state(self, state: QuantumState):
        state.apply_controle_gate(self.targets, self.controls, self.matrix())

class SWAPGate(TwoQubitGate):
        """SWAP gate"""
        lowername = "swap"

        def __init__(self, targets: Tuple[int, int]):
            self.targets = targets
            self.u_params = None

        @classmethod
        def create(cls,
                   targets: int,
                   params: tuple,
                   options: Optional[dict] = None) -> 'SWAPGate':
            if options:
                raise ValueError(f"{cls.__name__} doesn't take options")
            return cls(targets)

        def dagger(self):
            return SWAPGate(self.targets)

        def matrix(self):
            return np.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]], dtype=complex)

        def update_quantum_state(self, state: QuantumState):
            state.apply(self.targets[0], self.targets[1], self.matrix())
