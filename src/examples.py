from src.micrograd import Value

Dataset = tuple[list[list[Value]], list[Value]]


def xor() -> Dataset:
    xs = [
        [Value(0.0), Value(0.0)],
        [Value(1.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(1.0)],
    ]
    ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]
    return xs, ys


def or_gate() -> Dataset:
    xs = [
        [Value(0.0), Value(0.0)],
        [Value(1.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(1.0)],
    ]
    ys = [Value(0.0), Value(1.0), Value(1.0), Value(1.0)]
    return xs, ys
