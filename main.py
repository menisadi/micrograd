from src.micrograd import Value
from src.nn import MLP, quad_loss


def value_example():
    x1 = Value(0.3, label="x1")
    x2 = Value(0.2, label="x2")
    w1 = Value(0.5, label="w1")
    w2 = Value(-0.4, label="w2")
    b = 1

    x1w1 = x1 * w1
    x1w1.label = "x1w1"
    x2w2 = x2 * w2
    x2w2.label = "x2w2"

    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1w1x2w2"
    r = x1w1x2w2 + b
    r.label = "r"
    o = r.tanh()
    o.label = "o"

    o.backward()
    o.print_graph()


def full_example():
    xs = [
        [Value(0.0), Value(0.0)], 
        [Value(1.0), Value(0.0)], 
        [Value(0.0), Value(1.0)], 
        [Value(1.0), Value(1.0)], 
    ]
    ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]
    mlp = MLP([2, 4, 1])
    step = 0.1
    iterations = 1000
    for _ in range(iterations):
        ypred = [mlp(inpt)[0] for inpt in xs]
        loss = quad_loss(ys, ypred)
        loss.backward()
        for p in  mlp.parameters():
            p.data -= p.grad * step
            p.grad = 0
    ypred = [mlp(inpt)[0] for inpt in xs]
    loss = quad_loss(ys, ypred)
    print(loss.data)
    print(f"True: {ys}")
    print(f"Predictions: {ypred}")

if __name__ == "__main__":
    full_example()
