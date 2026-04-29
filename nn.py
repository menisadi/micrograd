from micrograd import Value
from random import uniform

class Neuron:
    _weights: list[Value]
    _bias: Value

    def __init__(self, size: int) -> None:
        self._weights = [Value(uniform(-1, 1)) for _ in range(size)]
        self._bias = Value(uniform(-1, 1))
        

    def __call__(self, inputs: list[Value]) -> Value:
        dot_prod = sum([x * w for x, w  in zip(self._weights, inputs)], self._bias)
        return dot_prod.tanh()

class Layer:
    def __init__(self) -> None:
        pass

class MLP:
    def __init__(self) -> None:
        pass

if __name__=="__main__":
    # sanity check
    xs = [Value(i) for i in range(3)]
    n = Neuron(3)
    print(n._weights)
    print(n._bias)
    print(n(xs))
