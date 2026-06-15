import matplotlib.pyplot as plt
from typing import List, Tuple

if __name__=="__main__":
    plt.title("Problem: XOR, Algorithmus: Parameter-Shift - QNN, Durchschnitt über 250 Iterationen; Differenzenfunktion")
    plt.ylabel("Fehler[MSE]")
    plt.xlabel("Iteration[1]")
    plt.grid(True)
    plt.yscale("symlog", linthresh=1e-10)

    with open("qcomputer.txt") as f:
        dump: str = f.read()
        lines: List[str] = dump.split("\n")
        lines = lines[1:]
        lines = list(filter(lambda x: x, lines))
        lines2: List[Tuple[int, float, float]] = list(map(lambda x: (int(x[0]), float(x[1]), float(x[2])), [x.split(" ") for x in lines]))
        X: List[int] = [x[0] for x in lines2]
        Y: List[float] = [x[1] for x in lines2]
        Y2: List[float] = [x[2] for x in lines2]
        Y3 = [y - y2 for y, y2 in zip(Y, Y2)]
        plt.plot(X, Y3, label="Differenz Y - Y2")

    plt.legend()
    plt.show()

"""
    with open("ga_sbx_cauchy.txt") as f:
        dump: str = f.read()
        lines: List[str] = dump.split("\n")
        lines: List[str] = list(filter(lambda x: x, lines))
        Y: List[float] = list(map(lambda x: float(x), lines))
        X: List[int] = [x for x in range(len(Y))]
        plt.plot(X, Y, label="Cauchy-Verteilung, SBX")

    with open("ga_sbx_normal.txt") as f:
        dump: str = f.read()
        lines: List[str] = dump.split("\n")
        lines: List[str] = list(filter(lambda x: x, lines))
        Y: List[float] = list(map(lambda x: float(x), lines))
        X: List[int] = [x for x in range(len(Y))]
        plt.plot(X, Y, label="Normalverteilung, SBX")


    with open("ga_blx_alpha_normal.txt") as f:
        dump: str = f.read()
        lines: List[str] = dump.split("\n")
        lines: List[str] = list(filter(lambda x: x, lines))
        Y: List[float] = list(map(lambda x: float(x), lines))
        X: List[int] = [x for x in range(len(Y))]
        plt.plot(X, Y, label="Normalverteilung, BLX-alpha")


    with open("ep_normal.txt") as f:
        dump: str = f.read()
        lines: List[str] = dump.split("\n")
        lines: List[str] = list(filter(lambda x: x, lines))
        Y: List[float] = list(map(lambda x: float(x), lines))
        X: List[int] = [x for x in range(len(Y))]
        plt.plot(X, Y, label="Mutationsoperator: Normalverteilung")

    plt.hlines(y=0.00659819, xmin = 40, xmax=100, color='red', linestyle="--")

"""

