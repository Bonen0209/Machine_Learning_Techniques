import numpy as np
from libsvm.svmutil import *
from libsvm.commonutil import svm_read_problem


def main():
    y_train, x_train = svm_read_problem('../Data/hw5/satimage.scale')

    y_train = [temp == 6.0 for temp in y_train]

    y_test, x_test = svm_read_problem('../Data/hw5/satimage.scale.t')

    y_test = [temp == 6.0 for temp in y_test]

    for g in [0.1, 1, 10, 100, 1000]:

        print(f'---------')
        print(f'Gamma {g}')

        model = svm_train(y_train, x_train, f'-t 2 -g {g} -c 0.1')

        p_label, p_acc, p_val = svm_predict(y_test, x_test, model)

        print(f'Eout = {p_acc[1]}')


if __name__ == '__main__':
    main()
