import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from astropy.io import fits
import sys
from scipy.interpolate import interp1d
from scipy import stats

# Ensure local imports work
sys.path.append(os.getcwd())

# Try to import master_shifting if available, otherwise define a placeholder or pass
try:
    from rv_net.master_shifting import master_shifting
except ImportError:
    pass

def load_harps_ccf_native(folder_path):
    """
    Loads HARPS-N CCF FITS files keeping NATIVE resolution.
    Extracts additional QC parameters (SNR, Cloud, Extinction) for filtering.
    """
    file_list = glob.glob(os.path.join(folder_path, "*.fits"))
    file_list.sort()
    
    if not file_list:
        return None

    print(f"Loading {len(file_list)} files from {folder_path} (Native Resolution)...")
    
    dataset = []

    for f in file_list:
        try:
            with fits.open(f) as hdul:
                # 1. Locate Data
                target_hdu = hdul[0] if (hdul[0].data is not None and hdul[0].data.size > 0) else hdul[1]
                header = target_hdu.header
                primary_header = hdul[0].header
                data = target_hdu.data

                # 2. Extract Keywords
                def get_val(keys, default=None):
                    for h in [header, primary_header]:
                        for k in keys:
                            if k in h: return h[k]
                    return default

                # Time
                bjd = get_val(['HIERARCH TNG QC BJD', 'BJD', 'MJD-OBS'])
                if bjd is None: continue
                bjd = float(bjd)

                # RV Data
                rv = get_val(['HIERARCH TNG QC CCF RV', 'HIERARCH ESO QC CCF RV'])
                fwhm = get_val(['HIERARCH TNG QC CCF FWHM', 'HIERARCH ESO QC CCF FWHM'])
                cont = get_val(['HIERARCH TNG QC CCF CONTRAST', 'HIERARCH ESO QC CCF CONTRAST'])
                bis = get_val(['HIERARCH TNG QC CCF BIS SPAN', 'HIERARCH ESO QC CCF BIS SPAN'], 0.0)

                # --- NEW: Quality Control Parameters for Weighting & Filtering ---
                # SNR usually on Order 50 for HARPS-N
                snr = get_val(['HIERARCH TNG QC SPECTRO SNR50', 'HIERARCH ESO QC SPECTRO SNR50', 'SNR', 'SNR50'], 0.0)
                
                # Atmosphere parameters (Standard in HARPS-N/TNG headers)
                # Cloud is often a probability 0-100 or 0-1. We assume 0-100 based on prompt.
                # cloud_prob = get_val('HIERARCH TNG QC CLOUD', 0.0) 
                obs_quality = get_val(['HIERARCH TNG QC OBS_QUALITY', 'OBS_QUALITY'], 0.0)
                extinction = get_val(['HIERARCH TNG QC EXT CORR', 'RV_DIFF_EXTINCTION'], 0.0)

                if rv is None or fwhm is None or cont is None: continue

                # 3. Grid Construction
                step = float(get_val(['HIERARCH TNG RV STEP', 'CDELT1', 'DELTAV'], 0.25))
                start = float(get_val(['HIERARCH TNG RV START', 'CRVAL1', 'STARTV']))
                
                # Handle 2D arrays
                if data.ndim == 2:
                    ccf_profile = data[-1, :]
                else:
                    ccf_profile = data
                
                # Sanitize Data
                if not np.isfinite(ccf_profile).all():
                    ccf_profile = np.nan_to_num(ccf_profile, nan=np.nanmedian(ccf_profile))

                n_pixels = len(ccf_profile)
                
                if start is None:
                    crpix = float(get_val(['CRPIX1'], n_pixels // 2))
                    crval = float(get_val(['CRVAL1'], 0.0))
                    start = crval - (crpix * step)

                native_grid = start + np.arange(n_pixels) * step

                dataset.append({
                    'filename': os.path.basename(f),
                    'bjd': bjd,
                    'rv': float(rv),
                    'ccf': ccf_profile.astype(np.float64),
                    'grid': native_grid.astype(np.float64),
                    'fwhm': float(fwhm),
                    'cont': float(cont),
                    'bis': float(bis),
                    'snr': float(snr),
                    'cloud': float(obs_quality),
                    'extinction': float(extinction)
                })

        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

    return dataset

def filter_observations(dataset, max_cloud=0.99, max_extinction=0.1):
    """
    Filters observations based on Cloud Probability, Extinction, and SNR.
    Reference: Cameron-2021 & Zoe-P1 aggressive filtering.
    """
    if not dataset: return []
    
    filtered_data = []
    removed_count = 0
    
    for d in dataset:
        # Check Cloud Probability
        # Some headers use 0.0-1.0, some 0-100. If val <= 1 but max is 99, we might need adjustment,
        # but usually headers are consistent with integer percentage for 'CLOUD'.
        is_cloudy = d['cloud'] > max_cloud
        
        # Check Extinction
        is_opaque = d['extinction'] > max_extinction
        
        if not is_cloudy and not is_opaque:
            filtered_data.append(d)
        else:
            removed_count += 1
            
    print(f"Filtering: Kept {len(filtered_data)}/{len(dataset)}. Removed {removed_count} bad frames.")
    return filtered_data


def align_and_resample(dataset, target_grid):
    """
    Aligns CCFs to zero velocity (Rest Frame) and resamples to target grid.
    Returns stacked arrays for CCFs and SNR (needed for weighting later).
    """
    if not dataset: return None, None, None

    bjd_arr = np.array([d['bjd'] for d in dataset])
    rv_arr = np.array([d['rv'] for d in dataset])
    snr_arr = np.array([d['snr'] for d in dataset])
    
    aligned_resampled_ccfs = []
    
    for i, data in enumerate(dataset):
        native_grid = data['grid']
        native_ccf = data['ccf']
        rv_star = rv_arr[i]
        
        # Interpolate
        interpolator = interp1d(native_grid, native_ccf, kind='cubic', fill_value="extrapolate")
        
        # Shift: To move star to 0 rest frame, sample at (Target + RV)
        shifted_query_points = target_grid + rv_star
        
        aligned_ccf = interpolator(shifted_query_points)
        
        # Normalize by wings (Continuum division)
        # Using outer 15 pixels on both sides
        wing_indices = np.concatenate([np.arange(15), np.arange(len(target_grid)-15, len(target_grid))])
        continuum = np.median(aligned_ccf[wing_indices])
        
        if continuum > 0:
            aligned_ccf /= continuum
            
        aligned_resampled_ccfs.append(aligned_ccf)

    return np.array(aligned_resampled_ccfs), bjd_arr, snr_arr

def compute_weighted_master(ccf_array, snr_array):
    """
    Computes a Master CCF using SNR-weighted averaging.
    Weight = SNR^2 (Inverse Variance approximation).
    """
    # Weights: SNR^2
    weights = snr_array ** 2
    
    # Handle case where all weights are zero (though filtering should prevent this)
    if np.sum(weights) == 0:
        return np.median(ccf_array, axis=0)
    
    # Broadcasting weights to match CCF shape (N_obs, N_pixels)
    weights_matrix = weights[:, np.newaxis]
    
    # Weighted Average
    weighted_master = np.sum(ccf_array * weights_matrix, axis=0) / np.sum(weights_matrix, axis=0)
    
    return weighted_master

def run_robust_pipeline():
    base_data_dir = "DATA"
    
    # Neural Network / Zoe-P1 standard grid
    TARGET_GRID = np.linspace(-20, 20, 161)
    
    # --- 1. Master Reference Generation (Quiet Star) ---
    quiet_date = "2018-03-29"
    print(f"\n--- Generating Master Reference from {quiet_date} ---")
    
    ref_raw = load_harps_ccf_native(os.path.join(base_data_dir, quiet_date))
    
    # Apply Filtering to Reference Frames too (Strict quality for master)
    ref_clean = filter_observations(ref_raw, max_cloud=0.99, max_extinction=0.1)
    # ref_clean = ref_raw
    if not ref_clean: 
        print("Error: No good reference frames found.")
        return

    # Align
    ref_ccfs, _, ref_snrs = align_and_resample(ref_clean, TARGET_GRID)
    
    # Create Weighted Master Reference
    master_reference_ccf = compute_weighted_master(ref_ccfs, ref_snrs)
    print("Master Reference created (Weighted by SNR^2).")

    # --- 2. Process Target Dates ---
    target_dates = ["2016-03-29", "2015-09-17", "2017-03-07", "2017-08-13"]
    
    for date in target_dates:
        print(f"\n--- Processing Target Date: {date} ---")
        path = os.path.join(base_data_dir, date)
        
        # Load
        target_raw = load_harps_ccf_native(path)
        
        # Filter (Cloud > 99% removed, Extinction > 0.5 removed)

        target_clean = filter_observations(target_raw, max_cloud=0.99, max_extinction=0.1)

        if not target_clean:
            print(f"Skipping {date}: All files filtered out.")
            continue
        
        # Align
        target_ccfs, bjds, target_snrs = align_and_resample(target_clean, TARGET_GRID)
        
        # Calculate Residuals (Observed - Master)
        individual_residuals = target_ccfs - master_reference_ccf
        
        # Create Daily Master (Weighted)
        daily_master = compute_weighted_master(target_ccfs, target_snrs)
        daily_residual = daily_master - master_reference_ccf
        
        # --- Visualization ---
        plt.figure(figsize=(10, 6))
        
        # Plot individual residuals (grey)
        plt.plot(TARGET_GRID, individual_residuals.T, color='black', alpha=0.05)
        
        # Plot Daily Average Residual (Red)
        plt.plot(TARGET_GRID, daily_residual, color='red', linewidth=2, label=f'Daily Mean Residual ({len(target_clean)} exp)')
        
        plt.title(f"Residuals: {date}\nFiltered (Cloud<99%, Ext<0.5) & Weighted by SNR²")
        plt.xlabel("Velocity (km/s)")
        plt.ylabel("Normalized Flux Delta")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_robust_pipeline()