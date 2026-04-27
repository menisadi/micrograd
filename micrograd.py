from collections.abc import Callable, Iterable
from typing import override
from math import exp


class Value:
    data: int | float
    grad: int | float
    _label: str | None
    _children: set[Value]
    _backward: Callable[[], None]

    def __init__(
        self, n: int | float, _children: Iterable[Value] = (), _label: str | None = ""
    ) -> None:
        self.data = n
        self.grad = 0

        self._label = _label
        self._children = set(_children)
        self._backward = lambda: None

    def __add__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(n=self.data + other.data, _children=(self, other), _label="+")

    def __mul__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other, _label="")
        return Value(n=self.data * other.data, _children=(self, other), _label="*")

    def tanh(self) -> Value:
        t = (exp(self.data) - exp(-self.data)) / (exp(self.data) + exp(-self.data))
        return Value(n=t, _children=(self,), _label="tanh")

    @override
    def __repr__(self) -> str:
        return f"Value({self.data}, {self._label})"
