import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
import os
import glob
import sys
from rv_net.master_shifting import master_shifting

# --- Constants for RV_NET ---
# The master_shifting function expects a specific velocity grid (161 points, -20 to 20 km/s).
# We must interpolate our data (which might vary, e.g., 49 bins) onto this grid.
RV_NET_START = -20.0
RV_NET_END = 20.0
RV_NET_BINS = 161
RV_TARGET_GRID = np.linspace(RV_NET_START, RV_NET_END, RV_NET_BINS) # The grid rv_net expects

def extract_data_from_fits(file_path):
    """
    Extracts metadata and CCF profile from a single HARPS-N CCF FITS file.
    Interpolates the CCF onto the standard RV_NET velocity grid.
    """
    try:
        with fits.open(file_path) as hdul:
            header = hdul[0].header

            # 1. Read Raw CCF Data
            # Shape is usually (Orders, Velocities). Last row is the combined average.
            ccf_data_2d = hdul[1].data
            ccf_raw = ccf_data_2d[-1, :]

            # 2. Reconstruct the Source Velocity Grid from Header
            # We need to know the velocity axis of the *current* file to interpolate correctly.
            # Using defaults based on your file if header keys are missing.
            vel_start = header.get('HIERARCH TNG RV START', -19.58)
            vel_step = header.get('HIERARCH TNG RV STEP', 0.82)
            n_bins_raw = ccf_raw.shape[0]

            # Construct the velocity axis for this specific file
            vel_source_grid = np.linspace(vel_start,
                                          vel_start + vel_step * (n_bins_raw - 1),
                                          n_bins_raw)

            # 3. Interpolate onto the RV_NET Target Grid (161 bins)
            # 'nearest' extrapolation handles edges if the new grid is slightly wider than source
            # Converting to float64 is crucial for mpyfit stability
            interpolator = interp1d(vel_source_grid, ccf_raw, kind='cubic',
                                    fill_value="extrapolate", bounds_error=False)
            ccf_interpolated = interpolator(RV_TARGET_GRID).astype(np.float64)

            # 4. Extract Header Values
            bjd_bary = header.get('HIERARCH TNG QC BJD', 0.0)
            berv = header.get('HIERARCH TNG QC BERV', 0.0)
            rv_bary_drs = header.get('HIERARCH TNG QC CCF RV', 0.0)

            # SNR (Signal to Noise Ratio) calculation (section 3.2.1-1)
            snr_vals = []
            for i in range(1, 70):
                key = f'HIERARCH TNG QC ORDER{i} SNR'
                if key in header:
                    snr_vals.append(header[key])
            snr_ccf = np.mean(snr_vals) if snr_vals else 100.0

            # Activity Indicators (section 3.2.1-2: exclude observations with cloud contamination)
            fwhm = header.get('HIERARCH TNG QC CCF FWHM', 0.0)
            cont = header.get('HIERARCH TNG QC CCF CONTRAST', 0.0)
            bis = header.get('HIERARCH TNG QC CCF BIS SPAN', 0.0)
            # Placeholders for QC. This dummy values need to be reviewed
            # is it related to OBS_QUALITY??????
            cloud_prob = 0.0
            diff_ext_corr = 15.0

            return {
                'bjd': float(bjd_bary),
                'ccf': ccf_interpolated, # Now shape (161,)
                'rv_drs': float(rv_bary_drs),
                'berv': float(berv),
                'snr_ccf': float(snr_ccf),
                'fwhm': float(fwhm),
                'cont': float(cont),
                'bis': float(bis),
                'cloud_prob': cloud_prob,
                'diff_ext_corr': diff_ext_corr
            }
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

def calculate_daily_weighted_average(daily_observations):
    """
    Step 1: Weighted average of observations for a single day.
    """
    # --- 1. Filter Observations ---
    filtered_obs = []
    EXCLUSION_THRESHOLD_CLOUD = 99.0
    EXCLUSION_THRESHOLD_EXT = 10.0

    for obs in daily_observations:
        if not (obs['cloud_prob'] > EXCLUSION_THRESHOLD_CLOUD and obs['diff_ext_corr'] < EXCLUSION_THRESHOLD_EXT):
            # With dummy values of cloud_prob and diff_ext_corr, all observations are kept
            # NEED TO BE REVIEWED
            filtered_obs.append(obs)

    if not filtered_obs:
        return None

    # --- 2. Weighted Sums ---
    # The 'ccf' here is already interpolated to (161,)
    ccf_shape = filtered_obs[0]['ccf'].shape

    w_ccf_sum = np.zeros(ccf_shape, dtype=np.float64)
    w_rv_sum = 0.0
    w_fwhm_sum = 0.0
    w_cont_sum = 0.0
    w_bis_sum = 0.0
    w_bjd_sum = 0.0
    total_weight = 0.0
    berv_sum = 0.0

    for obs in filtered_obs:
        w = obs['snr_ccf'] ** 2
        total_weight += w

        w_ccf_sum += obs['ccf'] * w
        w_rv_sum += obs['rv_drs'] * w
        w_fwhm_sum += obs['fwhm'] * w
        w_cont_sum += obs['cont'] * w
        w_bis_sum += obs['bis'] * w
        w_bjd_sum += obs['bjd'] * w
        berv_sum += obs['berv']

    if total_weight == 0: return None

    # --- 3. Compute Final Averages ---
    avg_results = {
        'bjd': w_bjd_sum / total_weight,
        'ccf': w_ccf_sum / total_weight,
        'rv': w_rv_sum / total_weight,
        'berv': berv_sum / len(filtered_obs),
        'fwhm': w_fwhm_sum / total_weight,
        'cont': w_cont_sum / total_weight,
        'bis': w_bis_sum / total_weight
    }
    return avg_results

def prepare_data_for_master_shifting(file_list):
    """
    Main pipeline driver.
    """
    all_obs = []
    print(f"Processing {len(file_list)} files...")

    for f in file_list:
        data = extract_data_from_fits(f)
        if data: all_obs.append(data)

    if not all_obs:
        return {"error": "No valid data extracted"}

    # Group by integer BJD (Day)
    daily_groups = {}
    for obs in all_obs:
        day = int(obs['bjd'])
        if day not in daily_groups: daily_groups[day] = []
        daily_groups[day].append(obs)

    # Process Days
    final_data = {
        'bjd': [], 'ccfBary': [], 'rvh': [],
        'fwhm': [], 'cont': [], 'bis': []
    }

    for day in sorted(daily_groups.keys()):
        avg = calculate_daily_weighted_average(daily_groups[day])
        if avg:
            final_data['bjd'].append(avg['bjd'])

            # ccfBary now has shape (161,)
            final_data['ccfBary'].append(avg['ccf'])
            final_data['rvh'].append(avg['rv'])

            final_data['fwhm'].append(avg['fwhm'])
            final_data['cont'].append(avg['cont'])
            final_data['bis'].append(avg['bis'])

    # Convert lists to numpy arrays
    # Ensure float64 for mpyfit stability
    return {k: np.array(v, dtype=np.float64) for k, v in final_data.items()}

# --- Execution ---

# 1. Get Files
folder_path = "./2015-07-29-30-31/" # Make sure this matches your actual folder
fits_files = glob.glob(os.path.join(folder_path, "*_CCF_A.fits"))

# 2. Process
if fits_files:
    processed = prepare_data_for_master_shifting(fits_files)

    if "error" not in processed:
        print(f"Data ready. Shapes: CCF {processed['ccfBary'].shape}, RV {processed['rvh'].shape}")

        # 3. Run master_shifting
        try:
            # Create array of zeros instead of passing "NULL" string.
            # master_shifting attempts to add this array to the RVs, so it must be numeric.
            removed_planets = np.zeros_like(processed['rvh'])

            df = master_shifting(
                processed['bjd'],
                processed['ccfBary'],
                processed['rvh'],
                ref_frame_shift="off",
                removed_planet_rvs=removed_planets,
                zero_or_median="zero",
                fwhm=processed['fwhm'],
                cont=processed['cont'],
                bis=processed['bis']
            )
            print("Master shifting complete.")
            # print(df.columns)
            print(df['CCF_normalized_list'].head().to_string(float_format="{:.6f}".format))



        except ImportError:
            print("rv_net module not found. Check your imports.")
        except Exception as e:
            print(f"Error during master_shifting: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(processed["error"])
else:
    print("No FITS files found.")
