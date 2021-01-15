from pathlib import Path
import pandas as pd
import numpy as np
from decision_tree import *
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

T = 2000

def one_iter(X_train, Y_train, N):
    np.random.seed()
    idx = np.random.randint(0, N, int(N/2))
    oob_idx = np.array([i for i in np.arange(N) if i not in np.unique(idx)])
    N_oob,  = oob_idx.shape
    x = X_train[idx]
    y = Y_train[idx]

    root = build(x, y, 0)
    y_predict = np.apply_along_axis(predict, 1, X_train[oob_idx], root=root)
    eoob = np.sum(y_predict != Y_train[oob_idx]) / N_oob

    return eoob


def main():
    data_dir = Path('../Data/hw6/')

    df_train = pd.read_csv(data_dir / 'hw6_train.dat', header=None, delimiter=' ')

    X_train, Y_train = df_train.iloc[:, :10].to_numpy(), df_train.iloc[:, 10].to_numpy()

    N_train, = Y_train.shape

    Y_predict = np.zeros_like(Y_train)
    Eoob = 0

    with ProcessPoolExecutor() as executor:
        for eoob in tqdm(executor.map(one_iter, [X_train]*T, [Y_train]*T, [N_train]*T), total=T):
            Eoob += eoob

    print(f'Eoob: {Eoob / T}')


if __name__ == '__main__':
    main()
