import os
from astropy.io import fits
import numpy as np
from master_shifting import master_shifting


def shifter_fits_maker(bjd, ccfBary, rvh,
                       ref_frame_shift,  # "off" or a specific value in km/s
                       removed_planet_rvs,  # array of rv values for planet signal in km/s
                       zero_or_median,
                       fwhm,
                       cont,
                       bis,
                       path):

    df_master = master_shifting(bjd, ccfBary, rvh,
                                ref_frame_shift=ref_frame_shift,  # "off" or a specific value in km/s
                                removed_planet_rvs= removed_planet_rvs,  # array of rv values for planet signal in km/s
                                zero_or_median = zero_or_median,
                                fwhm = fwhm,
                                cont = cont,
                                bis = bis)

    # write it all to one file
    # creates columns in CCF.fits files
    col1 = fits.Column(name='bjd', format='E', array=df_master["BJD"])

    # ccf params
    col2 = fits.Column(name='og_ccf_list', format='161E', dim='(732)',
                       array=np.array([df_master["og_ccf_list"]]).reshape((732 ,161)))
    col3 = fits.Column(name='jup_shifted_CCF_data_list' ,format='161E', dim='(732)',
                       array=np.array([df_master["jup_shifted_CCF_data_list"]]).reshape((732 ,161)))
    col4 = fits.Column(name='zero_shifted_CCF_list', format='161E',  dim='(732)',
                       array=np.array([df_master["zero_shifted_CCF_list"]]).reshape((732 ,161)))
    col5 = fits.Column(name='CCF_normalized_list', format='161E', dim='(161)',
                       array=np.array([df_master["CCF_normalized_list"]]).reshape((732 ,161)))
    col6 = fits.Column(name='mu_og_list', format='E', array=df_master["mu_og_list"])
    col7 = fits.Column(name='mu_jup_list', format='E', array=df_master["mu_jup_list"])
    col8 = fits.Column(name='mu_zero_list', format='E', array=df_master["mu_zero_list"])

    # rv and textfile params
    col9 = fits.Column(name='vrad_star', format='E', array=df_master["vrad_star"])
    # col10 = fits.Column(name='vrad_plan_star', format='E', array=df_master["vrad_plan_star"])
    col11 = fits.Column(name='fwhm', format='E', array=df_master["fwhm"])
    col12 = fits.Column(name='cont', format='E', array=df_master["cont"])
    col13 = fits.Column(name='bis_span', format='E', array=df_master["bis"])
    # col14 = fits.Column(name='noise', format='E', array=df_master["noise"])
    # col15 = fits.Column(name='s_mw', format='E', array=df_master["s_mw"])

    cols1 = fits.ColDefs([col2, col3, col4, col5])

    cols2 = fits.ColDefs([col1, col6, col7, col8,
                          col9])

    # hdu = fits.PrimaryHDU()
    primary_hdu = fits.PrimaryHDU()
    t = fits.BinTableHDU.from_columns([col2, col3, col4, col5])
    table_hdu2 = fits.BinTableHDU.from_columns([col1, col6, col7, col8,
                                                col9, col11, col12, col13])

    hdul = fits.HDUList([primary_hdu, t, table_hdu2])

    # Make directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    hdul.writeto(pat h +'/shifted_ccfs_combined.fits')
    hdul.close()


if __name__ == '__main__':
    shifter_fits_maker()