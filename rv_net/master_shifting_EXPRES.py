import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
from astropy.io.fits import getheader
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import mpyfit
from astropy.io import fits
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import csv
import pickle
import math as m
import pandas as pd
import os
from fwhm_sigma import fwhm_sigma

# master shifting function that allows you to choose whether to shift by the best fit or provided rvs

def master_shifting_EXPRES(bjd, ccfBary, fwhm, wls, rvh, srvh,
                           ref_frame_shift,  # "off" or a specific value in km/s
                           removed_planet_rvs,  # array of rv values for planet signal in km/s OR "NULL"
                           zero_or_median,  # "zero" or "median"
                           shift_by_rv):  # "best_fit" or "provided rvs"
    number_of_ccfs = len(ccfBary)

    # HARPS direct data lists
    BJD_list = []
    og_ccf_list = []
    rv_from_HARPS_list = []
    rv_from_HARPS_error_list = []
    v_rad_raw_list = []

    # mpyfit lists
    mu_og_list = []
    mu_jup_list = []
    mu_planet_list = []
    mu_zero_list = []
    sigma_og_list = []

    # CCF lists
    compiled_ccf_list = []
    jup_shifted_CCF_data_list = []
    shifted_CCF_list = []
    final_ccf_list = []
    CCF_normalized_list = []

    spline_method = 'cubic'
    for i in range(0, number_of_ccfs):
        day_of_observation = bjd[i]
        BJD_list.append(day_of_observation)

        # extracts the CCF data and rv from fits
        CCF_data = ccfBary[i]
        og_ccf_list.append(CCF_data)
        rv_from_HARPS = rvh[i]
        rv_from_HARPS_list.append(rv_from_HARPS)
        rv_from_HARPS_error_list.append(srvh[i])
        rv_data = wls[i]

        # Finds the local minima using a Gaussian fit
        # Define the actual function where     A = p[0], mu = p[1], sigma = p[2], c = p[3]
        def gauss(x, p):
            return -p[0] * np.exp(-(x - p[1]) ** 2 / (2. * p[2] ** 2)) + p[3]

        # A simple minimization function:
        def least(p, args):
            x, y = args
            return gauss(x, p) - y

        m = np.median(CCF_data)

        parinfo = [{'fixed': False, 'step': 1e-4 * m},
                   {'fixed': False, 'step': 1e-6},
                   {'fixed': False, 'step': 1e-4},
                   {'fixed': False, 'step': 1e-3 * m}]

        # no_shift fit
        std = fwhm_sigma(m - CCF_data, rv_data)[1]
        sigma_og_list.append(std)
        index_mean = np.argmin(CCF_data)
        p0 = [(np.min(CCF_data) - np.max(CCF_data) / np.max(CCF_data)) * m,
              rv_data[index_mean],
              std,
              m]

        pfit_no_shift, results_no_shift = mpyfit.fit(least, p0, (rv_data, CCF_data), parinfo)
        mu_og = pfit_no_shift[1]
        mu_og_list.append(mu_og)
        compiled_ccf_list.append(CCF_data)

        # Add in reference frame shift

        if removed_planet_rvs[0] != "NULL":  # Remove known planet signals *a priori*
            jupiter_shift = removed_planet_rvs[i]
            v_rad_raw = rvh[i] + removed_planet_rvs[i]
            v_rad_raw_list.append(v_rad_raw)

            # planet removal shift
            rv_data_jupiter_shift = rv_data + jupiter_shift  # minus sign
            f_jup = interp1d(rv_data_jupiter_shift, CCF_data, kind=spline_method, fill_value='extrapolate')
            jupiter_shifted_CCF_data = f_jup(rv_data)
            jup_shifted_CCF_data_list.append(jupiter_shifted_CCF_data)
            compiled_ccf_list.append(jupiter_shifted_CCF_data)

            # fits the shifted by planets removed data
            p_shifted_jup = [(np.min(CCF_data) - np.max(CCF_data) / np.max(CCF_data)) * m,
                             rv_data[index_mean] + jupiter_shift,
                             std,
                             m]
            pfit_jup, results_jup = mpyfit.fit(least, p_shifted_jup, (rv_data, jupiter_shifted_CCF_data), parinfo)
            m_jup = pfit_jup[1]
            mu_jup_list.append(m_jup)

            if zero_or_median == "zero":
                # Shift to zero
                ccf_to_use = compiled_ccf_list[len(compiled_ccf_list) - 1]

                shift_to_zero = -(rv_from_HARPS)
                rv_data_shifted = rv_data + shift_to_zero

                f = interp1d(rv_data_shifted, ccf_to_use, kind=spline_method, fill_value='extrapolate')
                shifted_CCF_data = f(rv_data)
                shifted_CCF_list.append(shifted_CCF_data)
                compiled_ccf_list.append(shifted_CCF_data)

                # fits the shifted data
                p_shifted = [(np.min(CCF_data) - np.max(CCF_data) / np.max(CCF_data)) * m,
                             rv_data[index_mean] - shift_to_zero,
                             std,
                             m]
                pfit, results = mpyfit.fit(least, p_shifted, (rv_data, shifted_CCF_data), parinfo)
                m_zero = pfit[1]
                mu_zero_list.append(m_zero)  # -0.1)
            else:  # shifted to median instead
                ccf_to_use = compiled_ccf_list[len(compiled_ccf_list) - 1]

                if shift_by_rv == "best_fit":  # use the originally fitted values to shift by
                    shift_to_median = (np.mean(rvh) - m_jup)
                    rv_data_shifted = rv_data + shift_to_median

                    f = interp1d(rv_data_shifted, ccf_to_use, kind=spline_method, fill_value='extrapolate')
                    shifted_CCF_data = f(rv_data)
                    shifted_CCF_list.append(shifted_CCF_data)
                    compiled_ccf_list.append(shifted_CCF_data)

                    # fits the shifted data
                    p_shifted = [(np.min(CCF_data) - np.max(CCF_data) / np.max(CCF_data)) * m,
                                 rv_data[index_mean] - shift_to_median,
                                 std,
                                 m]
                    pfit, results = mpyfit.fit(least, p_shifted, (rv_data, shifted_CCF_data), parinfo)
                    m_zero = pfit[1]
                    mu_zero_list.append(m_zero)  # -0.1)

                else:  # shift by provided rvs
                    shift_to_median = (np.mean(rvh) - rv_from_HARPS)
                    rv_data_shifted = rv_data + shift_to_median

                    f = interp1d(rv_data_shifted, ccf_to_use, kind=spline_method, fill_value='extrapolate')
                    shifted_CCF_data = f(rv_data)
                    shifted_CCF_list.append(shifted_CCF_data)
                    compiled_ccf_list.append(shifted_CCF_data)

                    # fits the shifted data
                    p_shifted = [(np.min(CCF_data) - np.max(CCF_data) / np.max(CCF_data)) * m,
                                 rv_data[index_mean] - shift_to_median,
                                 std,
                                 m]
                    pfit, results = mpyfit.fit(least, p_shifted, (rv_data, shifted_CCF_data), parinfo)
                    m_zero = pfit[1]
                    mu_zero_list.append(m_zero)  # -0.1)
        else:  # Do not remove any planet signals *a priori*
            ccf_to_use = compiled_ccf_list[len(compiled_ccf_list) - 1]
            if zero_or_median == "zero":
                # Shift to zero

                shift_to_zero = -(rv_from_HARPS)
                rv_data_shifted = rv_data + shift_to_zero

                f = interp1d(rv_data_shifted, ccf_to_use, kind=spline_method, fill_value='extrapolate')
                shifted_CCF_data = f(rv_data)
                shifted_CCF_list.append(shifted_CCF_data)
                compiled_ccf_list.append(shifted_CCF_data)

                # fits the shifted data
                p_shifted = [(np.min(CCF_data) - np.max(CCF_data) / np.max(CCF_data)) * m,
                             rv_data[index_mean] - shift_to_zero,
                             std,
                             m]
                pfit, results = mpyfit.fit(least, p_shifted, (rv_data, shifted_CCF_data), parinfo)
                m_zero = pfit[1]
                mu_zero_list.append(m_zero)  # -0.1)
            else:  # shifted to median instead
                if shift_by_rv == "best_fit":  # use the originally fitted values to shift by
                    shift_to_median = (np.mean(rvh) - mu_og)
                    rv_data_shifted = rv_data + shift_to_median

                    f = interp1d(rv_data_shifted, ccf_to_use, kind=spline_method, fill_value='extrapolate')
                    shifted_CCF_data = f(rv_data)
                    shifted_CCF_list.append(shifted_CCF_data)
                    compiled_ccf_list.append(shifted_CCF_data)

                    # fits the shifted data
                    p_shifted = [(np.min(CCF_data) - np.max(CCF_data) / np.max(CCF_data)) * m,
                                 rv_data[index_mean] - shift_to_median,
                                 std,
                                 m]
                    pfit, results = mpyfit.fit(least, p_shifted, (rv_data, shifted_CCF_data), parinfo)
                    m_zero = pfit[1]
                    mu_zero_list.append(m_zero)  # -0.1)

                else:  # shift by provided rvs
                    shift_to_median = (np.mean(rvh) - rv_from_HARPS)
                    rv_data_shifted = rv_data + shift_to_median

                    f = interp1d(rv_data_shifted, ccf_to_use, kind=spline_method, fill_value='extrapolate')
                    shifted_CCF_data = f(rv_data)
                    shifted_CCF_list.append(shifted_CCF_data)
                    compiled_ccf_list.append(shifted_CCF_data)

                    # fits the shifted data
                    p_shifted = [(np.min(CCF_data) - np.max(CCF_data) / np.max(CCF_data)) * m,
                                 rv_data[index_mean] - shift_to_median,
                                 std,
                                 m]
                    pfit, results = mpyfit.fit(least, p_shifted, (rv_data, shifted_CCF_data), parinfo)
                    m_zero = pfit[1]
                    mu_zero_list.append(m_zero)  # -0.1)
        ccf_to_use = compiled_ccf_list[len(compiled_ccf_list) - 1]
        final_ccf_list.append(ccf_to_use)

        # normalize the CCFs
        x_left = ccf_to_use[0:150]
        x_right = ccf_to_use[573:723]
        x_norm_range = list(x_left) + list(x_right)
        CCF_normalized = ccf_to_use * (1 / np.mean(x_norm_range))
        CCF_normalized_list.append(CCF_normalized)

    # Create a dataframe
    d = {'BJD': BJD_list,
         'fwhm': fwhm,
         'wls': wls,
         'vrad_star': rvh,
         'svrad_star': srvh,
         'og_ccf_list': og_ccf_list,
         'zero_shifted_CCF_list': shifted_CCF_list,
         'CCF_normalized_list': CCF_normalized_list,
         'mu_og_list': mu_og_list,
         'mu_zero_list': mu_zero_list,
         'zero_or_median': zero_or_median,
         'shift_by_rv': shift_by_rv,
         }
    df = pd.DataFrame(data=d)

    return df