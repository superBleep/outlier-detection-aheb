import numpy as np
from algorithms import detect_unknown
from testing import gen_graph, gen_data, gen_features


def test_figure_8():
    X = [
        np.array([10, 20.2, 500.5, 4100, 50, 60, 70.5, 80, 90, 3000]),
        np.array([1000, 900, 799.4, 700, -6000, 500, 450, 300, -2200, -10000]),
        np.array([100, 100.1, 100, 100, 6000, 100, 350, 10000, 2200, 100]),
        np.array([10, 19.9, 500.5, -4100, 50, 60, 72.5, 80, 90, 4000]),
        np.array([1000, 900, 799.5, 700, -6000, 650, 400, 300, -2200, 10000]),
        np.array([100, 100, 100, 99.9, -6000, 350, 100, 100, -2200, 10000]),
    ]

    for i in range(len(X)):
        _, Y_pred = detect_unknown(X[i].astype(float), 0.5, 0.01)
        gen_graph(X[i], Y_pred, 0.5, 0.01)


def test_figure_9():
    X = [
        np.array([10.1, 20.2, 30, 4100, 50, 60, 70.5, 80, 90, 3000]),
        np.array([5010, 900, 808, 800, 6000, 500, 400, 300, 2200, 100]),
        np.array([100.1, 100, 99.9, 100, 6000, 100, 100, 10000, 2200, 100]),
        np.array([1000, 19.9, 30, -5100, 50, 60, 72.5, 80, 90, 4000]),
        np.array([9000, 907, 800, 700, -6000, 605, 400, 300, 6200, 100]),
        np.array([101, 99.9, -6000, 100, 100, 350, 100, 100, 6000, 100])
    ]

    for i in range(len(X)):
        _, Y_pred = detect_unknown(X[i].astype(float), 0.5, 0.01)
        gen_graph(X[i], Y_pred, 0.5, 0.01)


def test_synthetic_data(ref_outlier, n):
    features = gen_features()
    ref_nonout = [f for f in features if f.get('ref_outlier') == ref_outlier]
    rand_samples = np.random.choice(ref_nonout, n).astype(list)

    print(f'{n} random samples; reference is outlier = {ref_outlier}')

    tn, fn, tp, fp = 0, 0, 0, 0

    for i, f in enumerate(rand_samples):
        print(f'\rProcessing sample nr. {i}', end='')

        X, Y = gen_data(f)
        _, Y_pred = detect_unknown(X, 0.5, 0.01)

        for i in range(len(Y)):
            if Y[i] == -1:  # Missing (considered nonoutlier)
                if Y_pred[i] == 0: tn += 1    
            if Y[i] == 0:  # Nonoutlier
                if Y_pred[i] == 0: tn += 1
                if Y_pred[i] != 0: fp += 1
            if Y[i] == 1:  # Outlier
                if Y_pred[i] == 0: fn += 1
                if Y_pred[i] != 0: tp += 1
    print()

    ptn = round(tn / (tn + fp) * 100, 2)
    print(f'Correctly detected non-outliers: {ptn}%')

    pfn = round(fn / (tn + fp) * 100, 2)
    print(f'Falsely detected non-outliers: {pfn}%')

    ptp = round(tp / (tp + fn) * 100, 2)
    print(f'Correctly detected outliers: {ptp}%')

    pfp = round(fp / (tp + fn) * 100, 2)
    print(f'Falsely detected outliers: {pfp}%')

if __name__ == '__main__':
    #test_figure_8()
    #test_figure_9()
    #test_synthetic_data(False, 10)
    #test_synthetic_data(True, 10)

    pass
