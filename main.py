from src.micrograd import Value
from src.nn import MLP, binary_cross_entropy, quad_loss
from src.examples import xor, or_gate
from collections.abc import Callable
from tqdm import tqdm


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


def train(
    xs: list[list[Value]],
    ys: list[Value],
    mlp: MLP,
    step: float,
    loss: Callable[[list[Value], list[Value]], Value],
    iterations: int,
    decay: bool = False,
) -> tuple[list[Value], list[float | int]]:
    losses: list[float | int] = []
    learning_rate = step
    for i in tqdm(range(iterations)):
        if decay:
            learning_rate = step * (1 - i / iterations)
        ypred = [mlp(inpt)[0] for inpt in xs]
        current_loss = loss(ys, ypred)
        current_loss.backward()
        # tqdm.write(f"Loss: {current_loss.data:.4f} | Rate: {learning_rate:.4f}")
        losses.append(current_loss.data)
        for p in mlp.parameters():
            p.data -= p.grad * learning_rate
            p.grad = 0
    ypred = [mlp(inpt)[0] for inpt in xs]
    current_loss = loss(ys, ypred)
    return ypred, losses


def full_example():
    xs, ys = or_gate()
    activ = "sigmoid"
    mlp = MLP([2, 4, 1], activation=activ)

    ypred1, losses = train(
        xs,
        ys,
        mlp=mlp,
        step=0.1,
        loss=binary_cross_entropy,
        iterations=2000,
        decay=True,
    )
    print(f"Activation: {activ}")
    print(losses[-1])

    xs, ys = xor()
    activ = "tanh"
    mlp = MLP([2, 4, 1], activation=activ)

    ypred2, losses = train(
        xs, ys, mlp=mlp, step=0.1, loss=quad_loss, iterations=2000, decay=True
    )
    print(f"Activation: {activ}")
    print(losses[-1])

    print("------")
    print(", ".join([f"{y.sigmoid().data:.4f}" for y in ypred1]))
    print(", ".join([f"{y.data:.4f}" for y in ypred2]))


if __name__ == "__main__":
    full_example()
