import numpy as np
from algorithms import detect_known, detect_unknown
from testing import gen_graph
from deletion import to_const


if __name__ == '__main__':
    np.random.seed(2)

    X = np.array([10, 20.2, 500.5, 4100, 50, 60, 70.5, 80, 90, 3000])

    X_clean, Y_pred = detect_unknown(X, 0.5, 0.01)
    gen_graph(X, Y_pred, 0.5, 0.01)