from micrograd import Value


def main():
    x1 = Value(0.3)
    x2 = Value(0.2)
    w1 = Value(0.5)
    w2 = Value(0.4)
    b = 1

    x1w1 = x1 * w1
    x2w2 = x2 * w2

    r = x1w1 + x2w2 + b
    o = r.tanh()

    print(o)


if __name__ == "__main__":
    main()
