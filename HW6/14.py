from pathlib import Path
import pandas as pd
import numpy as np
from decision_tree import *


def main():
    data_dir = Path('../Data/hw6/')

    df_train = pd.read_csv(data_dir / 'hw6_train.dat', header=None, delimiter=' ')
    df_test = pd.read_csv(data_dir / 'hw6_test.dat', header=None, delimiter=' ')

    X_train, Y_train = df_train.iloc[:, :10].to_numpy(), df_train.iloc[:, 10].to_numpy()
    X_test, Y_test = df_test.iloc[:, :10].to_numpy(), df_test.iloc[:, 10].to_numpy()

    N, = Y_test.shape

    root = build(X_train, Y_train, 0)
    Y_predict = np.apply_along_axis(predict, 1, X_test, root=root)
    print(f'Eout: {np.sum(Y_predict != Y_test) / N}')


if __name__ == '__main__':
    main()
