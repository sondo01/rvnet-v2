import numpy as np
import math as m

import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy.io import fits
import pandas as pd
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.interpolate import interp1d



# A simple fwhm finder
# flip over ccf, find max, find half max locations, distance between those two locations, divide
def fwhm_sigma(ccf, axis):
    x = -ccf - np.min(-ccf)
    maxval = np.max(x)
    ind = np.argmax(x)
    # print("max: "+str(maxval)+" at index: "+str(ind))

    half1 = x[0:ind]
    half2 = x[ind:]

    h1 = np.abs(half1 -(maxval /2.))
    h2 = np.abs(half2 -(maxval /2.))

    min1 = np.min(h1)
    ind1 = np.where(h1 == min1)
    # print("half-max1: "+str(min1)+" at index: "+str(ind1))

    min2 = np.min(h2)
    ind2 = np.where(h2==min2)

    fwhm = ind2[0] +ind - ind1[0]

    # return fwhm, sigma
    return axis[fwhm[0]] - axis[0], (axis[fwhm[0]] - axis[0]) / 2.335

#returns fwhm and sigma


if __name__ == '__main__':
    fwhm_sigma()