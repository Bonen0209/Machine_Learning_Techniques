import numpy as np
from libsvm.svmutil import *
from libsvm.commonutil import svm_read_problem


def main():
    y_train, x_train = svm_read_problem('../Data/hw5/satimage.scale')

    y_train = [temp == 3.0 for temp in y_train]

    model = svm_train(y_train, x_train, '-t 0 -c 10')

    sv_coef = np.array(model.get_sv_coef())
    sv_indices = [index-1 for index in model.get_sv_indices()]

    _, x_support = svm_read_problem('../Data/hw5/satimage.scale', return_scipy=True)
    
    x_support = x_support.toarray()[sv_indices]

    print(f'||w|| is {np.linalg.norm(x_support.T @ sv_coef)}')


if __name__ == '__main__':
    main()
