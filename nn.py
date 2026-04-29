from micrograd import Value
from random import random

class Node:
    _weights: list[Value]
    def __init__(self, size: int) -> None:
        self._weights = [Value(random()) for _ in range(size)]
        

    def __call__(self, inputs: list[Value]) -> Value:
        prod = sum(x*w for x,w  in zip(self._weghts, inputs))
        return Value(0)

class Layer:
    def __init__(self) -> None:
        pass

class MLP:
    def __init__(self) -> None:
        pass
