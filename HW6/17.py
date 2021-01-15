from pathlib import Path
import pandas as pd
import numpy as np
from decision_tree import *
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

T = 2000

def one_iter(X_train, Y_train, X_test, N):
    np.random.seed()
    idx = np.random.randint(0, N, int(N/2))
    x = X_train[idx]
    y = Y_train[idx]

    root = build(x, y, 0)
    y_predict = np.apply_along_axis(predict, 1, X_test, root=root)

    return y_predict


def main():
    data_dir = Path('../Data/hw6/')

    df_train = pd.read_csv(data_dir / 'hw6_train.dat', header=None, delimiter=' ')
    df_test = pd.read_csv(data_dir / 'hw6_test.dat', header=None, delimiter=' ')

    X_train, Y_train = df_train.iloc[:, :10].to_numpy(), df_train.iloc[:, 10].to_numpy()
    X_test, Y_test = df_test.iloc[:, :10].to_numpy(), df_test.iloc[:, 10].to_numpy()

    N_train, = Y_train.shape
    N_test, = Y_test.shape

    Y_predict = np.zeros_like(Y_train)
    Ein = 0

    with ProcessPoolExecutor() as executor:
        for y_predict in tqdm(executor.map(one_iter, [X_train]*T, [Y_train]*T, [X_test]*T, [N_train]*T), total=T):
            Y_predict += y_predict

    Y_predict[Y_predict >= 0] = 1
    Y_predict[Y_predict < 0] = -1

    print(f'Eout: {np.sum(Y_predict != Y_test) / N_test}')


if __name__ == '__main__':
    main()
