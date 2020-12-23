import numpy as np
from libsvm.svmutil import *
from libsvm.commonutil import svm_read_problem


def main():
    y_train, x_train = svm_read_problem('../Data/hw5/satimage.scale')

    for i in range(5):

        print(f'Label {i+1}')

        temp_y_train = [temp == i+1 for temp in y_train]

        model = svm_train(temp_y_train, x_train, '-t 1 -d 2 -g 1 -r 1 -c 10')

        p_label, p_acc, p_val = svm_predict(temp_y_train, x_train, model)


if __name__ == '__main__':
    main()
