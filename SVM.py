"""
    Author: Yinbin Ma
"""

import numpy as np
try:
    from fastcache import clru_cache as lru_cache
except ImportError:
    from functools import lru_cache

def with_epsilon(epsilon):
    def func(a, b):
        return np.abs(a - b) < epsilon
    return func


def rand(m, x):
    r = x
    while r == x:
        r = np.random.randint(m)
    return r

def lru_kernel(X, kernel_func, maxsize=1024):
    n, _ = X.shape
    cache_size = int(min(n*(n+1)/2, maxsize))
    @lru_cache(maxsize=cache_size)
    def func(i, j):
        xi, xj = X[i], X[j]
        return kernel_func(xi, xj)
    return func

def inner(xi, xj):
    return np.dot(xi, xj)

def gaussian(sigma):
    def func(xi, xj):
        x = xi - xj
        return np.exp(-1 * x.dot(x) / 2 / sigma / sigma)
    return func

class SVM():
    def __init__(self, max_iter=10000, C=1.0, epsilon=0.001, kernel='linear', sigma=1):
        self.max_iter = max_iter
        self.C = C
        if kernel == 'linear':
            self.kernel = inner
        elif kernel == 'rbf':
            self.kernel = gaussian(sigma)
        else:
            self.kernel = inner
        self.with_epsilon = with_epsilon(epsilon)
        self.b = None

    def fit(self, X, y):
        self.n, _ = X.shape
        self.alpha = np.zeros((self.n))
        self.X = X
        self.u = np.zeros((self.n))
        self.y = y
        self.cache_kernel = lru_kernel(X, self.kernel)
        self.b = 0
        self.update()

        for i in range(self.max_iter):
            kkt_1 = self.check_kkt()
            kkt_2 = np.sum(y * self.alpha)
            if np.all(kkt_1) and self.with_epsilon(kkt_2, 0) == True:
                break
            first_alphas = np.arange(self.n)[kkt_1 == False]
            for j in first_alphas:
                alpha_j = self.alpha[j]
                i = rand(self.n, j)
                alpha_i = self.alpha[i]
                L, H = self.compute_L_H(alpha_j, alpha_i, y[j], y[i])
                eta = self.cache_kernel(i, i) + self.cache_kernel(j, j) - 2 * self.cache_kernel(min(i, j), max(i, j))
                if eta == 0:
                    continue
                Ei, Ej = self.u[i] - y[i], self.u[j] - y[j]
                alpha_j_new = alpha_j + y[j] * (Ei - Ej) / eta
                alpha_j_new = max(min(alpha_j_new, H), L)
                alpha_i_new = alpha_i + y[i]*y[j]* (alpha_j - alpha_j_new)

                b_1 = self.b + Ei + y[i] * (alpha_i_new - alpha_i) * self.cache_kernel(i, i) \
                        + y[j] * (alpha_j_new - alpha_j) * self.cache_kernel(min(i, j), max(i, j))
                b_2 = self.b + Ej + y[i] * (alpha_i_new - alpha_i) * self.cache_kernel(min(i, j), max(i, j)) \
                        + y[j] * (alpha_j_new - alpha_j) * self.cache_kernel(j, j)
                
                if 0 <= alpha_j_new <= self.C:
                    self.b = b_2
                elif 0 <= alpha_i_new <= self.C:
                    self.b = b_1
                else:
                    self.b = (b_1 + b_2)/2
                
                self.alpha[i] = alpha_i_new
                self.alpha[j] = alpha_j_new
                self.update()

        support_vector = self.alpha > 0       
        self.alpha = self.alpha[support_vector]
        self.X = self.X[support_vector]
        self.y = self.y[support_vector]
    
    def update(self):
        for i in range(self.n):
            k = np.array([self.cache_kernel(min(i, j), max(i, j)) for j in range(self.n)])
            self.u[i] = np.sum(self.alpha * self.y * k) - self.b

    def check_kkt(self):
        result = np.zeros((self.n), dtype=np.bool)
        v = self.y * self.u
        for i in range(self.n):
            if self.alpha[i] == 0:
                result[i] = v[i] >= 1
            elif self.alpha[i] == self.C:
                result[i] = v[i] <= 1
            else:
                result[i] = v[i] == 1
        return result

    def __predict(self, x):
        if self.b is None:
            return 1
        k = np.array([self.kernel(x, self.X[i]) for i in range(self.X.shape[0])])
        return np.sign(np.sum(self.alpha * self.y * k) - self.b)

    def predict(self, X):
        return np.array([self.__predict(X[i]) for i in range(X.shape[0])])

    def compute_L_H(self, alpha_prime_j, alpha_prime_i, y_j, y_i):
        C = self.C
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))
