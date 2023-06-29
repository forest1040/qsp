from typing import TYPE_CHECKING, Optional, Union, cast, Sequence
from typing_extensions import TypeAlias

StateVectorType: TypeAlias = "npt.NDArray[cp.cfloat]"

import cupy as cp

batch_size = 10
n_qubits = 2
_dim = 2**n_qubits
datas = []
data = cast(StateVectorType, cp.zeros(_dim))
for _ in range(batch_size):
    datas.append(data.copy())
_vector = cp.asarray(datas)
print(type(_vector))
print(_vector.shape)
print(_vector)

