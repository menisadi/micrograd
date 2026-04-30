import pytest
from collections.abc import Callable
from src.micrograd import Value

def finite_diff_numerical(func: Callable[[float], float], input: int | float, eps: float) -> float:
    return (func(input + eps) - func(input - eps)) / (2 * eps)

def test_value_init():
    v = Value(4)
    assert v.data == 4

def test_add_gradient():
    x = Value(2.0)
    out = x + Value(3.0)
    out.backward()
    numerical = finite_diff_numerical(lambda v: (Value(v) + Value(3.0)).data, 2.0, 1e-5)
    assert x.grad == pytest.approx(numerical, abs=1e-4)

def test_mul_gradient():
    x = Value(3.0)
    out = x * Value(4.0)
    out.backward()
    numerical = finite_diff_numerical(lambda v: (Value(v) * Value(4.0)).data, 3.0, 1e-5)
    assert x.grad == pytest.approx(numerical, abs=1e-4)

def test_tanh_gradient():
    x = Value(0.5)
    out = x.tanh()
    out.backward()
    numerical = finite_diff_numerical(lambda v: Value(v).tanh().data, 0.5, 1e-5)
    assert x.grad == pytest.approx(numerical, abs=1e-4)
