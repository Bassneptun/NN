from typing import cast
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.utils import Bunch
import numpy as np

digits: Bunch = cast(Bunch, load_digits(n_class=2))

n: int = 10
pca = PCA(n_components=n)

targets = digits.target
data = pca.fit_transform(digits.data)

with open(f"pca_data{n}", "w") as file:
  for target, data in zip(targets, data):
      file.write(str(target) + "\n" + "".join(filter(lambda x: x != "\n", np.array2string(data))) + "\n"*2)
