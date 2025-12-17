import numpy as np
import math as m

import matplotlib.pyplot as plt
import matplotlib as mpl

def Least_sq_fit(ccf_indexes, ccf_list_np, rv_np):
    # perform the L-S fit --------------
    n_cols = len(ccf_indexes) + 1
    n_rows = len(ccf_list_np)  # len(ccf_list_np)<---- unsmoothed
    x = np.zeros((n_rows, n_cols))

    x[:, 0] = 1
    for i in np.arange(0, len(ccf_indexes)):
        x[:, i + 1] = ccf_list_np_transpose[ccf_indexes[i]]  # (for all observations)
    y = rv_np

    alpha = x.transpose().dot(x)
    beta = x.transpose().dot(y)

    # and finally we can write a_coeff = alpha^-1 * beta
    inv_alpha = np.linalg.inv(alpha)
    a_coeff = inv_alpha.dot(beta)

    CCF_matrix = np.zeros((n_rows, len(ccf_indexes)))
    for i in np.arange(0, len(ccf_indexes)):
        CCF_matrix[:, i] = ccf_list_np_transpose[ccf_indexes[i]]  # (for all observations)

    y_preds = CCF_matrix.dot(a_coeff[1:]) + a_coeff[0]

    # Compute the scatter metric
    og_rms = new_rms = np.std(y, ddof=1)
    new_rms = np.std(y - y_preds, ddof=1)
    print("original rms: " + str(round(og_rms, 3)) + ", new rms: " + str(round(new_rms, 3)))

    return a_coeff, y_preds, og_rms, new_rms

if __name__ == '__main__':
    Least_sq_fit()