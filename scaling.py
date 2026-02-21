import matplotlib.pyplot as plt 

if __name__ == "__main__":
    with open("scaling_test2.txt") as f:
        scaling_values = list(filter(lambda x: len(x) == 2, (map(lambda x: x.split(" "), filter(lambda x: x, f.read())))))
        X = scaling_values[:][0]
        Y = scaling_values[:][1]
