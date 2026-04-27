from collections.abc import Callable
from typing import override
from math import exp


class Value:
    data: int | float
    grad: int | float
    label: str | None
    _backward: Callable[[], None]

    def __init__(self, n: int | float, label: str | None = "") -> None:
        self.data = n
        self.grad = 0
        self.label = label
        self._backward = lambda: None

    def __add__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other, "")
        return Value(self.data + other.data, "+")

    def __mul__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other, "")
        return Value(self.data * other.data, "*")

    def tanh(self) -> Value:
        t = (exp(self.data) - exp(-self.data)) / (exp(self.data) + exp(-self.data))
        return Value(t, "tanh")

    @override
    def __repr__(self) -> str:
        return f"Value({self.data}, {self.label})"
