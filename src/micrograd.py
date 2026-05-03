from collections.abc import Callable, Iterable
from typing import override
from math import exp, log


class Value:
    data: int | float
    grad: int | float
    label: str
    _op: str
    _children: set[Value]
    _backward: Callable[[], None]

    def __init__(
        self,
        n: int | float,
        _children: Iterable[Value] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = n
        self.grad = 0

        self.label = label
        self._op = _op
        self._children = set(_children)
        self._backward = lambda: None

    def __add__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(n=self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other: Value | int | float) -> Value:
        return self.__add__(other)

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | int | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: Value | int | float) -> Value:
        return (-self) + other

    def __mul__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(n=self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other: Value | int | float) -> Value:
        return self.__mul__(other)

    def __pow__(self, expo: int | float) -> Value:
        out = Value(n=self.data**expo, _children=(self,), _op="^")

        def _backward():
            self.grad += expo * self.data ** (expo - 1) * out.grad

        out._backward = _backward

        return out

    def log(self) -> Value:
        if self.data <= 0:
            raise ValueError("expected a positive input")
        out = Value(n=log(self.data), _children=(self,), _op="log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out

    def tanh(self) -> Value:
        t = (exp(self.data) - exp(-self.data)) / (exp(self.data) + exp(-self.data))
        out = Value(n=t, _children=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward

        return out

    def relu(self) -> Value:
        r = max(0, self.data)
        out = Value(n=r, _children=(self,), _op="relu")

        def _backward():
            relu_grad = 0 if out.data == 0 else 1
            self.grad += relu_grad * out.grad

        out._backward = _backward

        return out

    def _topo_sort(self) -> list[Value]:
        visited: set[Value] = set()
        topo_result: list[Value] = []

        def dfs(node: Value):
            if node in visited:
                return
            visited.add(node)
            for prev in node._children:
                dfs(prev)
            topo_result.append(node)

        dfs(self)
        return list(reversed(topo_result))

    def backward(self) -> None:
        grad_order = self._topo_sort()
        self.grad = 1
        for node in grad_order:
            node._backward()

    def print_graph(self, indent: int = 0) -> None:
        name = self.label or self._op or "val"
        # If case a compound value is also given a label:
        op_str = f" ({self._op})" if self.label and self._op else ""
        print(f"{'│ ' * indent}{name}{op_str}: {self.data:.4f} (grad={self.grad:.4f})")
        for child in self._children:
            child.print_graph(indent + 1)

    @override
    def __repr__(self) -> str:
        name = self.label or self._op or "val"
        return f"Value({self.data}, name={name})"
