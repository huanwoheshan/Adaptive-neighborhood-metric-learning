import argparse
import pdb

from functools import partial

from matplotlib import pyplot as plt

import numpy as np

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

from sklearn.datasets import (
    load_digits,
    load_iris,
    load_wine,
)

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils import (
    Bunch,
    check_random_state,
    shuffle as util_shuffle,
)

from mann import MANN, find_SD,find_SD1, data_wash

SEED = 1337
TEST_SIZE = 0.7
N_NEIGHBORS = 1

DATA_LOADERS = {
    'wine': load_wine,
    'iris': load_iris,
}

def main5(file_name='G://songkun//matlab_database//time_uci_test'):
    parser = argparse.ArgumentParser(
        description='Apply the kNN classifier using different metrics.',
    )

    parser.add_argument(
        '-d', '--data',
        choices=DATA_LOADERS,
        default='wine',
        help='on which data to run the model',
    )
    parser.add_argument(
        '--to-plot',
        action='store_true',
        help='plot the projected data',
    )
    parser.add_argument(
        '--seed',
        default=SEED,
        type=int,
        help='seed to fix the randomness',
    )
    parser.add_argument(
        '-v', '--verbose',
        default=0,
        action='count',
        help='how much information to output',
    )
    from sklearn.decomposition import PCA

    args = parser.parse_args()
    np.random.seed(args.seed)

    from scipy.io import loadmat, savemat
    import os
    fileList = os.listdir(file_name)
    t1 = []
    t2 = []

    #al = np.linspace(0.01,2,20)
    #b1 = np.linspace(0.1,2,20)
    al = np.power(2,np.linspace(-10,5,15))
    b1 = np.power(2,np.linspace(-10,5,15))
    #  accuracy = np.zeros((len(al),len(b1),n_iter))
    re_file = np.zeros((len(fileList),))
    #for file_i1 in range(len(fileList)-3):
    # file_i = file_i1+6
    for i_iter in range(5):
        file_i = 12
        m = loadmat(file_name + '//' + fileList[file_i])
        X = m['X']
        Y = m['Y']
        X,y=data_wash(X, Y)
# m = loadmat('C:\\Users\\songkun\\Downloads\\Benchmark Datasets\\MSRA25_uni.mat')
  #  m = loadmat('G:\\songkun\\matlab_database\\Benchmark Datasets\\UCI\\cars_uni')
  #  m = loadmat('G:\\songkun\\matlab_database\\Benchmark Datasets\\UCI\\Segment_uni')
  #  L1 = loadmat('G:\\songkun\\matlab_project\\mlcircus-lmnn-5b49cafaeb9a\\lmnn2\\L')
        n_iter = 1

        accuracy = np.zeros((len(al), len(b1), 150))
        ac_knn = np.zeros((len(al) + 1, 150))
        for kk in range(n_iter):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE)
            pca = PCA(n_components=X_tr.shape[1])
            pca.fit(X_tr)
            A = pca.components_
        #        A = L1['L']+np.eye()
            K1 = 10
            K2 = 10
        # Apply metric model
            #S_m1, D_m1 = find_SD1(y_tr)
            S_m2, D_m2 = find_SD(X_tr, y_tr, K1, K2*4)
            S_m = S_m2
            D_m = D_m2
            project_L = np.zeros((X_tr.shape[1], X_tr.shape[1], len(al), len(b1)))
            m1 = 1
            for i in range(len(al)):
                for j in range(len(b1)):
                    model = MANN(m = b1[j], alph = -al[i], reg = 0, mu = 1/np.sqrt(b1[j]), S_m = S_m, D_m = D_m, Y = y_tr, s_type = 1)
                    X_tr1 = model.fit_transform(X_tr, b1[j]*A)
                    X_te1 = model.transform(X_te)
                    A1 = model.A_return()
                    print(A1[0, 0:5])
                    print(al[i])
                    print(b1[j])
                    for kk1 in range(40):
                        knn = KNeighborsClassifier(n_neighbors = kk1 + 1)
                        knn.fit(X_tr1, y_tr.reshape(y_tr.shape[0],))
                        y_pr = knn.predict(X_te1)
                        accuracy[i, j, kk1] = 100 * accuracy_score(y_te, y_pr)
                    print(np.max(accuracy))
                    project_L[0:A1.shape[0], 0:A1.shape[1], i, j] = A1

            for kk1 in range(45):
                knn = KNeighborsClassifier(n_neighbors = kk1 + 1)
                knn.fit(X_tr,y_tr.reshape(y_tr.shape[0],))
                y_pr1 = knn.predict(X_te)
                ac_knn[0, kk1] = 100 * accuracy_score(y_te, y_pr1)
            savemat('C://Users//songkun//Desktop//10_26_tiaocan//de_ann20200313' + '//' + 'accuracy_' + fileList[file_i]+ str(i_iter), {'acc': accuracy, 'knn':ac_knn, 'L':project_L})
        re_file[file_i] = np.max(accuracy)
        savemat('C://Users//songkun//Desktop//10_26_tiaocan//de_ann20200313' + '//' + 'accuracy_all', {'acc_all_set': re_file})

if __name__ == '__main__':
    main5()
