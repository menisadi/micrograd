from src.micrograd import Value
from random import uniform


class Neuron:
    _weights: list[Value]
    _bias: Value
    _activation: str | None

    def __init__(self, size: int, activation: str | None = "tanh") -> None:
        self._weights = [Value(uniform(-1, 1)) for _ in range(size)]
        self._bias = Value(uniform(-1, 1))
        self._activation = activation

    def __call__(self, inputs: list[Value]) -> Value:
        dot_prod = sum([x * w for x, w in zip(self._weights, inputs)], self._bias)
        if self._activation == "relu":
            activated = dot_prod.relu()
        elif self._activation == "tanh":
            activated = dot_prod.tanh()
        elif self._activation == "sigmoid":
            activated = dot_prod.sigmoid()
        else:
            activated = dot_prod
        return activated

    def parameters(self) -> list[Value]:
        return self._weights + [self._bias]


class Layer:
    _nodes: list[Neuron]

    def __init__(
        self, input_size: int, output_size: int, activation: str | None = "tanh"
    ) -> None:
        self._nodes = [
            Neuron(input_size, activation=activation) for _ in range(output_size)
        ]

    def __call__(self, inputs: list[Value]) -> list[Value]:
        out = [n(inputs) for n in self._nodes]
        return out

    def parameters(self) -> list[Value]:
        return [p for n in self._nodes for p in n.parameters()]


class MLP:
    _layers: list[Layer]

    def __init__(
        self, layers_sizes: list[int], activation: str | None = "tanh"
    ) -> None:
        self._layers = [
            Layer(nin, nout, activation if i < len(layers_sizes) - 2 else None)
            for i, (nin, nout) in enumerate(zip(layers_sizes, layers_sizes[1:]))
        ]

    def __call__(self, inputs: list[Value]) -> list[Value]:
        for lay in self._layers:
            inputs = lay(inputs)
        return inputs

    def parameters(self) -> list[Value]:
        return [p for lay in self._layers for p in lay.parameters()]


def quad_loss(ytrue: list[Value], ypred: list[Value]) -> Value:
    return sum([(yt - yp) ** 2 for yt, yp in zip(ytrue, ypred)], Value(0))


def binary_cross_entropy(ytrue: list[Value], ypred: list[Value]) -> Value:
    return sum(
        [
            -yt * yp.sigmoid().log() - (1 - yt) * (1 - yp).sigmoid().log()
            for yt, yp in zip(ytrue, ypred)
        ],
        Value(0),
    )
