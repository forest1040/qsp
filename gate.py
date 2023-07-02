import cmath
import math
from typing import Any, cast, Callable, Iterable, Iterator, List, NoReturn, Optional, Tuple, Type, TypeVar, Union

#import numpy as cp
import cupy as cp

from state import QuantumState

class Gate:
    @property
    def n_qargs(self) -> int:
        raise NotImplementedError()

    def dagger(self) -> 'Gate':
        raise NotImplementedError()

    def matrix(self) -> cp.ndarray:
        raise NotImplementedError()
    
    def update_quantum_state(self, state: QuantumState):
        raise NotImplementedError()

class OneQubitGate(Gate):
    u_params: Optional[Tuple[float, float, float, float]]

    @property
    def n_qargs(self) -> int:
        return 1

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
        return cp.asarray([[a, b], [b, a]], dtype=complex)

    def update_quantum_state(self, state: QuantumState):
        state.apply(self.targets, self.matrix())
