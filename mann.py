import pdb

import numpy as np
from numpy import linalg as la
from scipy.optimize import (
    check_grad,
    fmin_cg,
    fmin_ncg,
    fmin_bfgs,
)

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

from sklearn.preprocessing import (
    StandardScaler,
)


def square_dist(x1, x2=None):
    """If x1 is NxD and x2 is MxD (default x1), return NxM square distances."""

    if x2 is None:
        x2 = x1

    return (
        np.sum(x1 * x1, 1)[:, np.newaxis] +
        np.sum(x2 * x2, 1)[np.newaxis, :] -
        np.dot(x1, (2 * x2.T))
    )

def data_wash(X,Y):
    N,d = X.shape[0],X.shape[1]
    label = np.unique(Y)
    temp = np.zeros((N,1)) == 1
    for i in range(len(label)):
        if np.sum([label[i] == Y])< 3:
            temp = temp |(label[i]==Y)
    return X[~temp[:,0],:],Y[~temp[:,0]]

def find_SD(X, Y, K1 = 15, K2 = 30):
    N, d = X.shape[0], X.shape[1]
    label = np.unique(Y)
    ind_x = np.array([i for i in range(N)])
    S = []
    D = []
    for i in range(label.shape[0]):
        ind = Y == label[i]
        X_s = X[ind[:,0],:]
        X_d = X[~ind[:,0],:]
        print(ind.shape)
        ind_xs = ind_x[ind[:,0]]
        ind_xd = ind_x[~ind[:,0]]
        dist_s = square_dist(X_s)
        dist_d = square_dist(X_s,X_d)

        ind_s = dist_s.argsort()
        ind_d = dist_d.argsort()

        S.extend(ind_xs[ind_s[:, 1:K1+1]].tolist())
        D.extend(ind_xd[ind_d[:, 1:K2+1]].tolist())
        print(i)
    return S, D

def find_SD1(Y):
    label = np.unique(Y)
    N = len(Y)
    ind_x = np.array([i for i in range(N)])
    S = []
    D = []
    for i in range(Y.shape[0]):
        ind = (Y == Y[i])
        S.append(ind_x[(ind & (ind_x != i))[0,:]])
        D.append(ind_x[~ind[:,0]])
    return S, D

''' 
from scipy.io import loadmat
from scipy.io import savemat
m = loadmat('G:\\songkun\\matlab_database\\Benchmark Datasets\\UCI\\chess_uni')
X = m['X']
Y = m['Y']
S, D = find_SD(X,Y,7,14)
print(X.shape)
'''




#
#
#
def mann_cost(A, xx, S_m, D_m, label, m, alph, reg, mu = 1,s_type = 1):
    """Neighbourhood Components Analysis: cost function and gradients

        ff, gg = mann_cost_log(A, xx, yy)

    Evaluate a linear projection from a D-dim space to a K-dim space (K<=D).
    See Goldberger et al. (2004).

    Inputs:
        A  KxD Current linear transformation.
        xx NxD Input data
        yy Nx1 Corresponding labels, taken from any discrete set

    Outputs:
        ff 1x1 MANN cost function
        gg KxD partial derivatives of ff wrt elements of A

    Motivation: gradients in existing implementations, and as written in the
    paper, have the wrong scaling with D. This implementation should scale
    correctly for problems with many input dimensions.

    Note: this function should be passed to a MINIMIZER.

    """

    N, D = xx.shape
    assert(A.shape[1] == D)


    # projection function:
    zz = np.dot(A, xx.T).T  # KxN

    # TODO Subsample part of data to compute loss on.
    # kk = np.exp(-square_dist(zz.T, zz.T[idxs]))  # Nxn
    # kk[idxs, np.arange(len(idxs))] = 0
    gg = np.zeros((D, D))
    yy = 0
    yy1 = 0
    gg1 = np.zeros((D, D))
    s_yy1 = 1
    for i in range(N):
        S_m_i = S_m[i]
        D_m_i = D_m[i]
        if len(S_m_i) <1:
            continue
        if len(D_m_i) <1:
            continue
        dist_s1 = square_dist(zz[i:i+1, :], zz[S_m_i, :])
        dist_d1 = square_dist(zz[i:i+1, :], zz[D_m_i, :])
        dist_s1 = dist_s1.reshape((dist_s1.shape[1],))
        dist_d1 = dist_d1.reshape((dist_d1.shape[1],))
        ind_d_hard = dist_d1 < np.max(dist_s1)+10
        ind_s_hard = dist_s1 > np.min(dist_d1)-10

        dist_s1 = dist_s1[ind_s_hard]

        dist_d1 = dist_d1[ind_d_hard]

        if alph < 0:
            ave_s = np.max(dist_s1)
        else:
            ave_s = np.min(dist_s1)
        dist_s = dist_s1 - ave_s
        #ave_d = np.sum(dist_d1)/dist_d1.shape[0]
        ave_d = np.min(dist_d1)
        dist_d = dist_d1 - ave_d # to avoid the value too large to let the exp(dist_d1) be nan
        dist_s_exp = np.exp(-alph * dist_s) + 1e-8
        dist_d_exp = np.exp(-m * dist_d) + 1e-8
        dist_s_exp[np.isnan(dist_s_exp)] = np.max(dist_s_exp[~np.isnan(dist_s_exp)])
        dist_d_exp[np.isnan(dist_d_exp)] = np.max(dist_d_exp[~np.isnan(dist_d_exp)])
        if s_type == 1:
            t_s = -np.log(np.sum(dist_s_exp)/dist_s_exp.shape[0])/alph + ave_s + 1
            t_d = -np.log(np.sum(dist_d_exp)/dist_d_exp.shape[0])/m + ave_d
        else:
            t_s = -np.log(np.sum(dist_s_exp)) / alph + 1
            t_d = -np.log(np.sum(dist_d_exp))/m

        t = t_s - t_d
        if t > 0:
            yy = yy + t
            t_s_rate = dist_s_exp / np.sum(dist_s_exp)
            t_d_rate = dist_d_exp / np.sum(dist_d_exp)
            s_temp = np.dot((t_s_rate[np.newaxis,:].T * xx[np.array(S_m_i)[ind_s_hard], :]).T, xx[np.array(S_m_i)[ind_s_hard], :])
            d_temp = np.dot((t_d_rate[np.newaxis,:].T * xx[np.array(D_m_i)[ind_d_hard], :]).T, xx[np.array(D_m_i)[ind_d_hard], :])
            gg = gg + s_temp - d_temp
            if np.isnan(gg.sum()):
                print('nan')
        gg1 = gg1 + np.dot(xx[(label[i] == label)[:, 0], :].T, xx[(label[i] == label)[:, 0], :])
        yy1 = yy1 + np.sum(zz[(label[i] == label)[:, 0], :] ** 2)
        s_yy1 = s_yy1 + np.sum(label[i] == label)
    gg = gg / zz.shape[1] + mu * gg1 / s_yy1 + reg*np.eye(D)

    yy = yy / zz.shape[1] + mu * yy1 / s_yy1 + reg * np.dot(A.ravel(), A.ravel())
    #gg = np.dot(gg,A.T).T + 2 * mu * A
    #gg = gg / np.sqrt(np.trace(np.dot(gg , gg.T)))
    return yy, gg


def mann_cost_ratio(A, xx, S_m, D_m, label, m, alph, reg, mu = 1,s_type = 1):
    """Neighbourhood Components Analysis: cost function and gradients

        ff, gg = mann_cost_log(A, xx, yy)

    Evaluate a linear projection from a D-dim space to a K-dim space (K<=D).
    See Goldberger et al. (2004).

    Inputs:
        A  KxD Current linear transformation.
        xx NxD Input data
        yy Nx1 Corresponding labels, taken from any discrete set

    Outputs:
        ff 1x1 MANN cost function
        gg KxD partial derivatives of ff wrt elements of A

    Motivation: gradients in existing implementations, and as written in the
    paper, have the wrong scaling with D. This implementation should scale
    correctly for problems with many input dimensions.

    Note: this function should be passed to a MINIMIZER.

    """

    N, D = xx.shape
    assert(A.shape[1] == D)


    # projection function:
    zz = np.dot(A, xx.T).T  # KxN

    # TODO Subsample part of data to compute loss on.
    # kk = np.exp(-square_dist(zz.T, zz.T[idxs]))  # Nxn
    # kk[idxs, np.arange(len(idxs))] = 0
    gg = np.zeros((D, D))
    yy = 0
    yy1 = 0
    gg1 = np.zeros((D, D))
    s_yy1 = 1
    for i in range(N):
        S_m_i = S_m[i]
        D_m_i = D_m[i]
        if len(S_m_i) <1:
            continue
        if len(D_m_i) <1:
            continue
        dist_s1 = square_dist(zz[i:i+1, :], zz[S_m_i, :])
        dist_d1 = square_dist(zz[i:i+1, :], zz[D_m_i, :])
        dist_s1 = dist_s1.reshape((dist_s1.shape[1],))
        dist_d1 = dist_d1.reshape((dist_d1.shape[1],))
        ind_d_hard = dist_d1 < np.max(dist_s1)+10
        ind_s_hard = dist_s1 > np.min(dist_d1)-10
        if np.sum(ind_d_hard) < 1:
            continue
        if np.sum(ind_s_hard) < 1:
            continue

        dist_s1 = dist_s1[ind_s_hard]

        dist_d1 = dist_d1[ind_d_hard]

        if alph < 0:
            ave_s = np.max(dist_s1)
        else:
            ave_s = np.min(dist_s1)
        dist_s = dist_s1 - ave_s
        #ave_d = np.sum(dist_d1)/dist_d1.shape[0]
        ave_d = np.min(dist_d1)
        dist_d = dist_d1 - ave_d # to avoid the value too large to let the exp(dist_d1) be nan
        dist_s_exp = np.exp(-alph * dist_s) + 1e-8
        dist_d_exp = np.exp(- dist_d) + 1e-8
        dist_s_exp[np.isnan(dist_s_exp)] = np.max(dist_s_exp[~np.isnan(dist_s_exp)])
        dist_d_exp[np.isnan(dist_d_exp)] = np.max(dist_d_exp[~np.isnan(dist_d_exp)])
        if s_type == 1:
            t_s = -np.log(np.sum(dist_s_exp)/dist_s_exp.shape[0])/alph + ave_s + m
            t_d = -np.log(np.sum(dist_d_exp)/dist_d_exp.shape[0]) + ave_d
        else:
            t_s = -np.log(np.sum(dist_s_exp)) / alph + ave_s + m
            t_d = -np.log(np.sum(dist_d_exp))+ ave_d

        t = t_s - t_d
        if t > 0:
            yy = yy + t
            t_s_rate = dist_s_exp / np.sum(dist_s_exp)
            t_d_rate = dist_d_exp / np.sum(dist_d_exp)
            s_temp = np.dot((t_s_rate[np.newaxis,:].T * xx[np.array(S_m_i)[ind_s_hard], :]).T, xx[np.array(S_m_i)[ind_s_hard], :])
            d_temp = np.dot((t_d_rate[np.newaxis,:].T * xx[np.array(D_m_i)[ind_d_hard], :]).T, xx[np.array(D_m_i)[ind_d_hard], :])
            gg = gg + s_temp - d_temp
            if np.isnan(gg.sum()):
                print('nan')

        if m != 0:
            gg1 = gg1 + np.dot(xx[(label[i] == label)[:, 0], :].T, xx[(label[i] == label)[:, 0], :])
            yy1 = yy1 + np.sum(zz[(label[i] == label)[:, 0], :] ** 2)
            s_yy1 = s_yy1 + np.sum(label[i] == label)
    gg = gg / zz.shape[1] + mu * gg1 / s_yy1 + reg*np.eye(D)

    yy = yy / zz.shape[1] + mu * yy1 / s_yy1 + reg * np.dot(A.ravel(), A.ravel())
    #gg = np.dot(gg,A.T).T + 2 * mu * A
    #gg = gg / np.sqrt(np.trace(np.dot(gg , gg.T)))
    return yy, gg



class MANN(BaseEstimator, TransformerMixin):
    def __init__(self, S_m, D_m,Y, m=1, alph=-1, lamd = 0.1,reg=0, mu=1, s_type=1, dim=None, optimizer='gd'):
        self.reg = reg
        self.K = dim
        self.m = m
        self.alph = alph
        self.lamd = lamd
        self.standard_scaler = StandardScaler()
        self.S_m = S_m
        self.D_m = D_m
        self.Y = Y
        self.s_type = s_type

        if optimizer in ('cg', 'conjugate_gradients'):
            self._fit = self._fit_conjugate_gradients
        elif optimizer in ('gd', 'gradient_descent'):
            self._fit = self._fit_gradient_descent
        else:
            raise ValueError("Unknown optimizer {:s}".format(optimizer))
    def A_return(self):
        return self.A
    def fit(self, X,A):

        N, D = X.shape

        if self.K is None:
            self.K = D
        #np.random.seed(1337)
        self.A = A
        self.M = np.dot(A.T,A)
        #self.A = np.random.randn(self.K, D)/np.sqrt(D)
        #print(self.A)

        X = self.standard_scaler.fit_transform(X)
        return self._fit(X)

    def _fit_gradient_descent(self, X):
        # Gradient descent.
        self.learning_rate = 1
        self.error_tol = 0.001
        self.max_iter = 1000

        prev_error = np.inf


        current_error = np.inf
        prev_error, g= mann_cost_ratio(self.A, X, self.S_m, self.D_m,self.Y, self.m, self.alph, self.reg, self.s_type)
        f1 = []
        for it in range(self.max_iter):
            g = g/np.sqrt(np.trace(g*g.T))
            An = self.A - self.learning_rate * g.dot(self.A)
            if np.isnan(An.sum()):
                print('nan')
            f, g = mann_cost_ratio(An, X, self.S_m, self.D_m, self.Y, self.m, self.alph, self.reg, self.s_type)
            g = g / np.sqrt(np.trace(g * g.T))
            err = f - prev_error
            if f < prev_error:
                self.A = An
                self.M = self.A.dot(self.A.T)
                self.learning_rate = 1.03 * self.learning_rate
                prev_error = f
                f1.append(f)
            else:
                self.learning_rate = 0.9 * self.learning_rate


#            print('{:4d} {:+.6f}'.format(it, f))
#            print('{:4d} {:+.6f}'.format(it, err))

            if self.learning_rate <1e-7 and np.abs(err) < self.error_tol:
                break
        #print('the loss at each iteration:')
        print(f1)

        return self

    def _fit_conjugate_gradients(self, X):
        N, D = X.shape

        def costf(A):
            f, _ = mann_cost_ratio(A.reshape([self.K, D]), X, self.S_m, self.D_m,self.Y, self.m, self.alph, self.reg)
            return f

        def costg(A):
            _, g = mann_cost_ratio(A.reshape([self.K, D]), X, self.S_m, self.D_m,self.Y, self.m, self.alph, self.reg)
            return g.ravel()

        print(check_grad(costf, costg, 0.1 * np.random.randn(self.K * D)))
        self.A = fmin_cg(costf, self.A.ravel(), costg, maxiter=4000)
        self.A = self.A.reshape([self.K, D])
        return self

    def fit_transform(self, X,A):
        #print(A[0,0])
        self.fit(X,A)
        return self.transform(X)

    def transform(self, X):
        return np.dot(self.standard_scaler.transform(X), self.A.T)

    def loss():
        f, _ = mann_cost_ratio(self.A.reshape([self.K, D]), X, self.S_m, self.D_m, self.m, self.alph, self.reg)
        return f
