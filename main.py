from micrograd import Value


def main():
    x1 = Value(0.3, label="x1")
    x2 = Value(0.2, label="x2")
    w1 = Value(0.5, label="w1")
    w2 = Value(-0.4, label="w2")
    b = 1

    x1w1 = x1 * w1
    x2w2 = x2 * w2

    x1w1x2w2 = x1w1 + x2w2
    r = x1w1x2w2 + b
    o = r.tanh()

    print(o)

    o.grad = 1
    o._backward()
    r._backward()
    x1w1x2w2._backward()
    x2w2._backward()
    x1w1._backward()

    o.print_graph()


if __name__ == "__main__":
    main()
