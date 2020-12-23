import numpy as np
from tqdm import tqdm
from libsvm.svmutil import *
from libsvm.commonutil import svm_read_problem
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

ITER = 1000

def one_iter(y, x):
    np.random.seed()
    shuffle = np.random.permutation(len(y))

    Evals = list()
    for g in [0.1, 1, 10, 100, 1000]:
        model = svm_train(y[shuffle[200:]], x[shuffle[200:]], f'-t 2 -g {g} -c 0.1 -q')

        p_label, p_acc, p_val = svm_predict(y[shuffle[:200]], x[shuffle[:200]], model, '-q')

        Evals.append((p_acc[1], g))

    return min(Evals)

def main():
    y_train, x_train = svm_read_problem('../Data/hw5/satimage.scale', return_scipy=True)

    y_train = y_train == 6.0

    gammas = {g:0 for g in [0.1, 1, 10, 100 ,1000]}


    with ProcessPoolExecutor() as executor:
        for _, idx in tqdm(executor.map(one_iter, [y_train]*ITER, [x_train]*ITER), total=ITER):
            gammas[idx] += 1

    print(gammas)

if __name__ == '__main__':
    main()
