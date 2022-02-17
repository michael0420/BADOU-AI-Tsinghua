# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 11:45:15 2021

@author: Administrator
"""

import cv2
import numpy as np
import scipy as sp
import scipy.linalg as sl


class RANSAC():
    def __init__(self, data, model):
        self.data = data
        self.model = model

    def fit(self, n, k, t, d, debug=False, return_all=False):
        iteration = 0
        bestfit = None
        besterr = np.inf
        best_inliner_idx = None
        while iteration < k:
            inner_idxs, test_idxs = self.random_partition(n, self.data.shape[0])
            inner_data = self.data[inner_idxs, :]
            test_data = self.data[test_idxs, :]
            maybemodel = self.model.fit(inner_data)
            test_err = self.model.get_error(test_data, maybemodel)
            print('test_err = ', test_err < t)
            also_idxs = test_idxs[test_err < t]
            also_inner = self.data[also_idxs, :]
            if debug:
                print('test_err.min()', test_err.min())
                print('test_err.max()', test_err.max())
                print('numpy.mean(test_err)', np.mean(test_err))
                print('iteration %d:len(alsoinliers) = %d' % (iteration, len(also_inner)))
            if (len(also_inner) > d):
                betterdata = np.concatenate((inner_data, also_inner))
                bettermodel = self.model.fit(betterdata)
                better_err = self.model.get_error(betterdata, bettermodel)
                thiserr = np.mean(better_err)
                if thiserr < besterr:
                    bestfit = bettermodel
                    besterr = thiserr
                    best_inner_idxs = np.concatenate((inner_idxs, also_idxs))
            iteration += 1
        if bestfit is None:
            raise ValueError("did't meet fit acceptance criteria")
        if return_all:
            return bestfit, {'inliers': best_inner_idxs}
        else:
            return bestfit

    def random_partition(self, n, n_data):
        all_idxs = np.arange(n_data)
        np.random.shuffle(all_idxs)
        idx1 = all_idxs[:n]
        idx2 = all_idxs[n:]
        return idx1, idx2


class LinearLeastSquareModel():
    def __init__(self, input_columes, output_columes, debug=False):
        self.input_columes = input_columes
        self.output_columes = output_columes
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columes]).T
        B = np.vstack([data[:, i] for i in self.output_columes]).T
        x, resids, rank, s = sl.lstsq(A, B)
        return x

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columes]).T
        B = np.vstack([data[:, i] for i in self.output_columes]).T
        B_fit = sp.dot(A, model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def main():
    n_sample = 500
    A = 20 * np.random.random((n_sample, 1))
    perfect_fit = 60 * np.random.normal(size=(1, 1))
    B = A * perfect_fit
    A_noisy = A + np.random.normal(size=A.shape)
    B_noisy = B + np.random.normal(size=B.shape)
    if 1:
        n_outliers = 100
        all_indxs = np.arange(A_noisy.shape[0])
        np.random.shuffle(all_indxs)
        outliers_idxs = all_indxs[:n_outliers]
        A_noisy[outliers_idxs] = 20 * np.random.random((n_outliers, 1))
        B_noisy[outliers_idxs] = 50 * np.random.random((n_outliers, 1))
    data = np.hstack((A_noisy, B_noisy))
    input_columes = range(1)
    output_columes = [1 + i for i in range(1)]
    model = LinearLeastSquareModel(input_columes, output_columes, debug=False)
    linear_fit, resids, rank, s = sp.linalg.lstsq(data[:, input_columes], data[:, output_columes])
    ransac = RANSAC(data, model)
    ransac_fit, ransac_data = ransac.fit(50, 1000, 7e3, 300, debug=False, return_all=True)
    if 1:
        import pylab

        sort_idxs = np.argsort(A[:, 0])
        A_col0_sorted = A[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        # else:
        #     pylab.plot( A_noisy[non_outlier_idxs,0], B_noisy[non_outlier_idxs,0], 'k.', label='noisy data' )
        #     pylab.plot( A_noisy[outlier_idxs,0], B_noisy[outlier_idxs,0], 'r.', label='outlier data' )

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == "__main__":
    main()