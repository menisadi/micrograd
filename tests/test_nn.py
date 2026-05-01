from src.nn import Neuron, Layer, MLP
from src.micrograd import Value

def test_neuron_call():
    xs = [Value(i) for i in range(3)]
    n = Neuron(3)
    result = n(xs)
    assert isinstance(result, Value)
    assert result.data >= -1 and result.data <= 1

def test_neuron_grad():
    xs = [Value(i) for i in range(1, 4)]
    n = Neuron(3)
    result = n(xs)
    result.backward()
    assert all(x.grad != 0 for x in xs)
    assert all(p.grad != 0 for p in n.parameters())

def test_layer_call():
    xs = [Value(i) for i in range(3)]
    inputes, outputs = 3, 4
    one_layer = Layer(inputes, outputs)
    out = one_layer(xs)
    assert len(out) == outputs
    assert all(r.data >= -1 and r.data <= 1 for r in out)

def test_layer_grad():
    xs = [Value(i) for i in range(1, 4)]
    inputes, outputs = 3, 4
    one_layer = Layer(inputes, outputs)
    out = one_layer(xs)
    result = sum(out, Value(0))
    result.backward()
    assert all(x.grad != 0 for x in xs)
    assert all(p.grad != 0 for p in one_layer.parameters())

def test_mlp_call():
    xs = [Value(i) for i in range(1, 4)]
    layers_sizes = [3, 4, 2]
    mlp = MLP(layers_sizes)
    out = mlp(xs)
    assert all(r.data >= -1 and r.data <= 1 for r in out)

def test_mlp_grad():
    xs = [Value(i) for i in range(1, 4)]
    layers_sizes = [3, 4, 2]
    mlp = MLP(layers_sizes)
    out = mlp(xs)
    result = sum(out, Value(0))
    result.backward()
    assert all(x.grad != 0 for x in xs)
    assert all(p.grad != 0 for p in mlp.parameters())
