import matplotlib.pyplot as plt
from typing import List

if __name__=="__main__":
    #plt.title("Evolutionary Programming Algorithmen im Vergleich, Optimierung XOR, 4 Parameter, Fehlerkurve gemittelt über 100 Durchläufe")
    plt.ylabel("Fehler[MSE]")
    plt.xlabel("Iteration[1]")
    plt.grid(True)
    plt.yscale("log")

    with open("ga_blx_alpha_cauchy.txt") as f:
        dump: str = f.read()
        lines: List[str] = dump.split("\n")
        lines: List[str] = list(filter(lambda x: x, lines))
        Y: List[float] = list(map(lambda x: float(x), lines))
        X: List[int] = [x for x in range(len(Y))]
        plt.plot(X, Y, label="Genetischer Algorithmus: Cauchy-Verteilung, BLX-alpha")

    with open("ps.txt") as f:
        dump: str = f.read()
        lines: List[str] = dump.split("\n")
        lines: List[str] = list(filter(lambda x: x, lines))
        Y: List[float] = list(map(lambda x: float(x), lines))
        X: List[int] = [x for x in range(len(Y))]
        plt.plot(X, Y, label="Parameter-Shift Gradientenabstieg, eta=1")

    with open("ep_cauchy.txt") as f:
        dump: str = f.read()
        lines: List[str] = dump.split("\n")
        lines: List[str] = list(filter(lambda x: x, lines))
        Y: List[float] = list(map(lambda x: float(x), lines))
        X: List[int] = [x for x in range(len(Y))]
        plt.plot(X, Y, label="Evolutionary Programming: Cauchy-Verteilung")

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

