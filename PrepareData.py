import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from astropy.io import fits
import sys
from scipy.interpolate import interp1d
from scipy import stats
import pandas as pd
sys.path.append(os.getcwd())
# Assuming rv_net package structure exists as per your setup
try:
    from rv_net.master_shifting import master_shifting
except ImportError:
    # Fallback if master_shifting.py is in the same directory
    try:
        from master_shifting import master_shifting
    except ImportError:
         print("Warning: master_shifting module not found.")

# Add the current directory to path to ensure local imports work


# Constants
TARGET_GRID_POINTS = 49
TARGET_RV_GRID = np.linspace(-20, 20, TARGET_GRID_POINTS)
DEFAULT_STEP_SIZE = 0.82

def get_header_val(header, keys, default=None):
    """Helper to safely extract value from header using multiple possible keys."""
    for k in keys:
        if k in header:
            return header[k]
    return default

def calculate_weight(header):
    """Calculates weight based on SNR or RV Error."""
    # Try to get Spectral S/N (Order 69 is standard reference)
    snr = get_header_val(header, ['HIERARCH TNG QC SPECTRO SN69', 'HIERARCH TNG QC ORDER69 SNR', 'SNR'], None)
    
    # Try to get RV Error
    rv_err = get_header_val(header, ['HIERARCH TNG QC CCF RV ERROR', 'HIERARCH ESO QC CCF RV ERROR'], None)
    
    if snr is not None and float(snr) > 0:
        return float(snr) ** 2
    elif rv_err is not None and float(rv_err) > 0:
        return 1.0 / (float(rv_err) ** 2)
    return 1.0

def process_single_fits(filepath, public_release_values):
    """
    Reads and processes a single FITS file.
    Returns a dictionary of extracted parameters or None if invalid.
    """
    
    fname = os.path.basename(filepath)
    # release = bary_to_helio_list[fname.replace( "_CCF_A.fits", ".fits")]
    release_name = fname.replace( "_CCF_A.fits", ".fits")
    # public_release_values.set_index('filename')
    try:
        with fits.open(filepath) as hdul:
            header = hdul[0].header
            data = hdul[1].data
            
            # Extract Critical Params
            bjd = get_header_val(header, ['MJD-OBS', 'BJD', 'HIERARCH TNG QC BJD'])
            # rv = get_header_val(header, ['HIERARCH TNG QC CCF RV', 'HIERARCH ESO QC CCF RV'])

            rv = (public_release_values.loc[release_name, 'rv'])/1000

            # print(f"BRV: {rv}, verse FITS rv: {rv_fits}")
            fwhm = public_release_values.loc[release_name, 'fwhm']/1000
            cont = public_release_values.loc[release_name, 'contrast']/1000
            bis = public_release_values.loc[release_name, 'bis_span']/1000

            # Validation
            if None in [bjd, rv, fwhm, cont]:
                return None
            
            if float(fwhm) == 0 or float(cont) == 0:
                return None

            # Grid Parameters
            step = float(get_header_val(header, ['HIERARCH TNG RV STEP', 'CDELT1'], DEFAULT_STEP_SIZE))
            rv_start = get_header_val(header, ['HIERARCH TNG RV START'])

            rv_start = float(rv_start)

            # CCF Processing
            # For 2D arrays (Orders, Pixels), take the last row (combined CCF)
            ccf_profile = data[-1, :] if data.ndim == 2 else data

            native_n = len(ccf_profile)
        
            ccf_profile = ccf_profile.astype(np.float64)

            # Resampling
            native_grid = rv_start + np.arange(native_n) * step

            interpolator = interp1d(native_grid, ccf_profile, kind='cubic', fill_value='extrapolate')
            ccf_resampled = interpolator(TARGET_RV_GRID)

            # NaNs Check
            if not np.isfinite(ccf_resampled).all():
                median_val = np.nanmedian(ccf_resampled)
                ccf_resampled[~np.isfinite(ccf_resampled)] = median_val
                ccf_resampled = np.nan_to_num(ccf_resampled, nan=median_val)

            weight = calculate_weight(header)

            return {
                'bjd': float(bjd),
                'rv': float(rv),
                'ccf': ccf_resampled,
                # 'ccf': ccf_profile,
                'fwhm': float(fwhm),
                'cont': float(cont),
                'bis': float(bis),
                'weight': weight,
                'step': step,
                'native_n': native_n,
                'rv_start': rv_start
            }
    except Exception as e:
        print(f"Error reading {os.path.basename(filepath)}: {e}")
        return None

def compute_nightly_average(nightly_data):
    """
    Computes weighted average for a list of observations.
    """
    # Convert list of dicts to dict of lists/arrays
    keys = nightly_data[0].keys()
    # Be explicit about what to stack to avoid jagged arrays if schemas differ slightly
    stack_keys = ['bjd', 'rv', 'fwhm', 'cont', 'bis', 'weight', 'ccf']
    arrays = {k: np.array([d[k] for d in nightly_data]) for k in stack_keys}
    
    weights = arrays['weight']
    sum_weights = np.sum(weights)
    
    if sum_weights == 0:
        return None

    # Weighted Averages
    avg_data = {}
    for k in ['bjd', 'rv', 'fwhm', 'cont', 'bis']:
        avg_data[k] = np.sum(arrays[k] * weights) / sum_weights
    
    # CCF Weighted Average (Broadcasting)
    # Shape: (N_obs, N_pixels) * (N_obs, 1)
    avg_data['ccf'] = np.sum(arrays['ccf'] * weights[:, np.newaxis], axis=0) / sum_weights

    return avg_data

def load_harps_ccf_data(folder_path, public_release_values):
    """
    Loads and bins HARPS-N CCF data by night.
    """
    file_list = glob.glob(os.path.join(folder_path, "*.fits"))
    if not file_list:
        print(f"Warning: No FITS files found in {folder_path}")
        return None

    file_list.sort()
    print(f"Loading {len(file_list)} files from {folder_path}...")

    # Group by Night
    night_groups = {}
    step_list = []
    native_n_list = []

    valid_count = 0
    for f in file_list:
        obs = process_single_fits(f, public_release_values)
        if obs:
            # night_id = int(obs['bjd']) # Noon-to-noon grouping
            night_id = folder_path.replace("DATA/", "") # DATA/2015-08-16
            if night_id not in night_groups:
                night_groups[night_id] = []
            night_groups[night_id].append(obs)
            
            step_list.append(obs['step'])
            native_n_list.append(obs['native_n'])
            valid_count += 1
        else:
            print("No obs")

    if valid_count == 0:
        return None

    print(f"Binning {valid_count} exposures into {len(night_groups)} nights...")

    # Compute Averages
    final_data = {k: [] for k in ['bjd', 'rvh', 'ccfBary', 'fwhm', 'cont', 'bis']}
    
    for night in sorted(night_groups.keys()):        
        avg = compute_nightly_average(night_groups[night])
        
        if avg:
            final_data['bjd'].append(avg['bjd'])
            final_data['rvh'].append(avg['rv'])
            final_data['ccfBary'].append(avg['ccf'])
            final_data['fwhm'].append(avg['fwhm'])
            final_data['cont'].append(avg['cont'])
            final_data['bis'].append(avg['bis'])

    # Format Output

    return {
        'bjd': np.array(final_data['bjd'], dtype=np.float64),
        'rvh': np.array(final_data['rvh'], dtype=np.float64),
        'ccfBary': final_data['ccfBary'], # List of arrays
        'fwhm': np.array(final_data['fwhm'], dtype=np.float64),
        'cont': np.array(final_data['cont'], dtype=np.float64),
        'bis': np.array(final_data['bis'], dtype=np.float64),
        'step': 0.82, # Resampled Grid Step
        'native_step': np.median(step_list) if step_list else DEFAULT_STEP_SIZE,
        'native_n': int(stats.mode(native_n_list, keepdims=True)[0][0]) if native_n_list else 49
    }

def process_date_residuals(data_dict, master_reference_ccf=None, is_reference_generation=False):
    """
    Runs the master_shifting pipeline on a dataset.
    """
    # 1. Master Shifting (Align to Zero)
    removed_planet_rvs = np.zeros_like(data_dict['rvh'])
    
    output = master_shifting(
        bjd=data_dict['bjd'],
        ccfBary=data_dict['ccfBary'],
        rvh=data_dict['rvh'],
        ref_frame_shift="off", 
        removed_planet_rvs=removed_planet_rvs, 
        zero_or_median="zero", 
        fwhm=data_dict['fwhm'],
        cont=data_dict['cont'],
        bis=data_dict['bis']
    )
    
    # Extract Shifted CCFs
    aligned_normalized_ccfs = output['CCF_normalized_list'].tolist()
    aligned_ccfs_arr = np.array(aligned_normalized_ccfs, dtype=np.float64)
    
    # 2. Daily Stacking (Median of aligned exposures)
    # Note: If len(data_dict['bjd']) > 1 (multiple nights in one folder), this stacks them all.
    daily_master_ccf = np.median(aligned_ccfs_arr, axis=0)

    if is_reference_generation:
        return aligned_normalized_ccfs[0]
    else:
        return output
    
    # elif master_reference_ccf is not None:
    #     # Return first night result (assuming 1 night per folder)
    #     daily_ccf = aligned_normalized_ccfs[0]
    #     cff_residual_list = daily_ccf - master_reference_ccf
    #     return cff_residual_list, daily_ccf, output
    
    # return daily_master_ccf, output

def plot_combined_residuals(velocity_axis, collected_residuals, opt, title_suffix=""):
    """
    Plots all collected residuals in a single figure.
    """
    # if not collected_residuals:
    #     print("No residuals to plot.")
    #     return
    
    if opt == "test":
        
        for date, resid, ccf, color in collected_residuals:    
            if date in ["2015-07-29","2016-03-29", "2015-09-17", "2017-03-07", "2017-08-13"]:
                plt.figure(figsize=(8, 8))   
                plt.ylim([-0.0006, 0.0025])    
                # plt.ylim([0.4, 1.05])    
                plt.plot(velocity_axis, resid, color=color, linewidth=1.5, alpha=0.7, label=date)
                # plt.plot(velocity_axis, ccf, color='black', linewidth=1.5, alpha=0.7, label=date)
                plt.title(f"Combined Residual CCFs {date}\nRed = Redshifted, Blue = Blueshifted")
                plt.xlabel("Radial Velocity (km/s) [Centered]")
                plt.ylabel("Normalized Flux Residuals")
                plt.xlim(min(velocity_axis), max(velocity_axis))
                plt.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                plt.grid(True, linestyle=':', alpha=0.6)
                # plt.legend(loc='best') # Optional if too many dates
                plt.tight_layout()
                plt.show()
    else :
        plt.figure(figsize=(8, 4)) 
        plt.ylim([-0.0006, 0.0035])
        for date, resid, ccf, color in collected_residuals:          
            plt.plot(velocity_axis, resid, color=color, linewidth=1.5, alpha=0.7, label=date)
            # plt.plot(velocity_axis, ccf, color='black', linewidth=1.5, alpha=0.7, label=date)

        plt.title(f"Combined Residual CCFs\nRed = Redshifted, Blue = Blueshifted")
        plt.xlabel("Radial Velocity (km/s) [Centered]")
        plt.ylabel("Normalized Flux Residuals")
        plt.xlim(min(velocity_axis), max(velocity_axis))
        plt.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.grid(True, linestyle=':', alpha=0.6)
        # plt.legend(loc='best') # Optional if too many dates
        plt.tight_layout()
        plt.show()

def createNpzFiles(dataframe_list, master_reference_ccf, outfile_name, cutoff_width=23):
    """
    Aggregates a list of Pandas DataFrames, computes residuals and cutoffs, 
    and saves to an .npz file.

    Args:
        dataframe_list (list): List of pd.DataFrames (the 'output' from process_date_residuals).
        master_reference_ccf (np.array): The master reference used to calculate residuals.
        outfile_name (str): Path to save the .npz file.
        cutoff_width (int): Half-width of the cutoff window (default 23 yields ~46 pixels).
    """
    
    if not dataframe_list:
        print("No data collected to save.")
        return

    # 1. Concatenate all daily DataFrames into one master DataFrame
    full_df = pd.concat(dataframe_list, ignore_index=True)

    print(f"Aggregating data... Total observations: {len(full_df)}")

    # 2. Extract Data Helper
    def stack_column(col_name):
        # Extract, flatten to 1D (fixes (N,1) vs (N,) mismatches), and trim
        # This prevents "setting an array element with a sequence" errors
        raw_values = [np.array(x).flatten() for x in full_df[col_name]]
        
        # Determine the minimum length among all rows to handle ragged arrays
        lengths = [len(x) for x in raw_values]
        min_len = min(lengths)
        
        # If lengths vary, warn the user
        if len(set(lengths)) > 1:
            print(f"Warning: Inhomogeneous lengths detected in column '{col_name}'. "
                  f"Range: {min(lengths)} - {max(lengths)}. Trimming to {min_len}.")
            
        # Trim all arrays to the minimum length
        trimmed_values = [x[:min_len] for x in raw_values]
            
        return np.vstack(trimmed_values)

    # 3. Basic Columns
    bjd = full_df['BJD'].to_numpy(dtype=np.float64)
    vrad_star = full_df['vrad_star'].to_numpy(dtype=np.float64) # This is 'rvh'
    fwhm = full_df['fwhm'].to_numpy(dtype=np.float64)
    cont = full_df['cont'].to_numpy(dtype=np.float64)
    bis = full_df['bis'].to_numpy(dtype=np.float64)

    # 4. CCF Lists (Full Size)
    og_ccf_list = stack_column('og_ccf_list')
    jup_shifted_CCF_data_list = stack_column('jup_shifted_CCF_data_list')
    zero_shifted_CCF_list = stack_column('zero_shifted_CCF_list')
    CCF_normalized_list = stack_column('CCF_normalized_list')

    # 5. Shift Parameters
    mu_og_list = full_df['mu_og_list'].to_numpy(dtype=np.float64)
    mu_jup_list = full_df['mu_jup_list'].to_numpy(dtype=np.float64)
    mu_zero_list = full_df['mu_zero_list'].to_numpy(dtype=np.float64)
    
    # 6. Shift by RV (Expected shape is scalar (), based on user requirements)
    shift_by_rv = np.array(0.0) 

    # 7. Compute Residuals
    
    # FIX: Handle case where user passes a tuple (CCF, metadata) instead of just CCF
    # This often happens if the output of process_date_residuals is passed directly
    if isinstance(master_reference_ccf, (tuple, list)) and len(master_reference_ccf) == 2:
        # If the first element looks like a CCF (length > 2), assume it's the data
        if hasattr(master_reference_ccf[0], '__len__') and len(master_reference_ccf[0]) > 2:
            print("Warning: master_reference_ccf appears to be a tuple (len=2). using master_reference_ccf[0] as the reference array.")
            master_reference_ccf = master_reference_ccf[0]

    # Force to numpy array to ensure shape properties work
    master_reference_ccf = np.array(master_reference_ccf)

    # Verify master_reference_ccf compatibility with the (potentially trimmed) CCF list
    current_ccf_len = CCF_normalized_list.shape[1]
    
    if master_reference_ccf.shape[0] != current_ccf_len:
        print(f"Warning: master_reference_ccf length ({master_reference_ccf.shape[0]}) "
              f"does not match data length ({current_ccf_len}). Trimming master ref.")
        master_reference_ccf = master_reference_ccf[:current_ccf_len]

    # Broadcast subtraction: (N_samples, N_pixels) - (N_pixels,)
    cff_residual_list = CCF_normalized_list - master_reference_ccf
    CCF_normalized_list_cutoff = CCF_normalized_list[:, 1:-2]
    CCF_residual_list_cutoff = cff_residual_list[:, 1:-2]
    # Calculate median and standard deviation across the entire dataset (axis 0)
    # shapes: (N_pixels,)
    # Calculate median and standard deviation across the entire dataset (axis 0)
    def normalize_arr(arr):
        med = np.median(arr, axis=0)
        std = np.std(arr, axis=0)
        # Handle zero std
        if np.isscalar(std):
            if std == 0: std = 1.0
        else:
            std[std == 0] = 1.0
        return (arr - med) / std

    ccf_residual_rescaled = normalize_arr(cff_residual_list)
    vrad_star_rescaled = normalize_arr(vrad_star)
    og_ccf_list_rescaled = normalize_arr(og_ccf_list)
    jup_shifted_CCF_data_list_rescaled = normalize_arr(jup_shifted_CCF_data_list)
    zero_shifted_CCF_list_rescaled = normalize_arr(zero_shifted_CCF_list)
    mu_og_list_rescaled = normalize_arr(mu_og_list)
    mu_zero_list_rescaled = normalize_arr(mu_zero_list)
    cont_rescaled = normalize_arr(cont)
    bis_rescaled = normalize_arr(bis)
    
    # Normalize cutoff version similarly (re-slicing the normalized full array is safer/consistent)
    ccf_residual_rescaled_cutoff = ccf_residual_rescaled[:, 1:-2]

    # 8. Save to NPZ
    np.savez(
        outfile_name,
        BJD=bjd,    
        vrad_star=vrad_star,
        og_ccf_list=og_ccf_list,
        jup_shifted_CCF_data_list=jup_shifted_CCF_data_list,
        zero_shifted_CCF_list=zero_shifted_CCF_list,
        CCF_normalized_list=CCF_normalized_list,
        cff_residual_list=cff_residual_list,
        CCF_normalized_list_cutoff=CCF_normalized_list_cutoff,
        CCF_residual_list_cutoff=CCF_residual_list_cutoff,
        ccf_residual_rescaled = ccf_residual_rescaled,
        ccf_residual_rescaled_cutoff = ccf_residual_rescaled_cutoff,
        mu_og_list=mu_og_list,
        mu_jup_list=mu_jup_list,
        mu_zero_list=mu_zero_list,
        fwhm=fwhm,
        cont=cont,
        bis=bis,
        shift_by_rv=shift_by_rv,
    )
    # np.savez(
    #     outfile_name,
    #     BJD=bjd,    
    #     vrad_star=vrad_star,
    #     og_ccf_list=og_ccf_list_rescaled,
    #     jup_shifted_CCF_data_list=jup_shifted_CCF_data_list_rescaled,
    #     zero_shifted_CCF_list=zero_shifted_CCF_list_rescaled,
    #     CCF_normalized_list=CCF_normalized_list,
    #     cff_residual_list=cff_residual_list,
    #     CCF_normalized_list_cutoff=CCF_normalized_list_cutoff,
    #     CCF_residual_list_cutoff=CCF_residual_list_cutoff,
    #     ccf_residual_rescaled = ccf_residual_rescaled,
    #     ccf_residual_rescaled_cutoff = ccf_residual_rescaled_cutoff,
    #     mu_og_list=mu_og_list_rescaled,
    #     mu_jup_list=mu_jup_list,
    #     mu_zero_list=mu_zero_list_rescaled,
    #     fwhm=fwhm,
    #     cont=cont_rescaled,
    #     bis=bis_rescaled,
    #     shift_by_rv=shift_by_rv,
    # )

    # plot_combined_residuals(velocity_axis, cff_residual_list, "test", title_suffix=f"(Ref: 2018-03-29)")
    print(f"Successfully saved {outfile_name}")



def run_pipeline(opt="test"):
    base_data_dir = "DATA"
    quiet_date = "2018-03-29"
    
    if not os.path.exists(base_data_dir):
        print(f"Error: Directory '{base_data_dir}' not found.")
        return

    # Manage option to select target dates
    if opt == "full":
        all_subdirs = sorted([d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))])
        target_dates = [d for d in all_subdirs if d != quiet_date]
    else:
        target_dates = ["2015-07-29","2016-03-29", "2015-09-17", "2017-03-07", "2017-08-13"]
  
    
    # Get a list of processed data from public_release_timeseries.csv
    public_release_values = pd.read_csv('public_release_timeseries.csv', index_col='filename')
    # Remove any leading/trailing spaces from column names
    public_release_values.columns = public_release_values.columns.str.strip()

    print(f"Found {len(target_dates)} target dates.")
    print("--- Starting Pipeline ---")
    
    # 1. Generate Master Reference (quiet date on 2018-03-29). We will subtract this from all other dates to obtain the residuals CCF.
    print(f"\nProcessing Reference: {quiet_date}")
    ref_data = load_harps_ccf_data(os.path.join(base_data_dir, quiet_date), public_release_values)
    # print(f"ref_data : {ref_data}")
    if ref_data is None:
        return

    master_reference = process_date_residuals(ref_data, is_reference_generation=True)
    ref_mean_rv = np.mean(ref_data['rvh'])

    
    # Grid Setup for plotting
    # ccf_step = ref_data.get('step', 0.25)
    ccf_step = 0.82
    n_points = len(master_reference)
    velocity_axis = (np.arange(n_points) - (n_points // 2)) * ccf_step 

    # 2. Process Targets
    collected_residuals = []
    collected_dataframes = []
    color_count = 0
    for date in target_dates:
        print(f"\nProcessing Target: {date}")
        target_path = os.path.join(base_data_dir, date)
        target_data = load_harps_ccf_data(target_path, public_release_values)
        
        if target_data is None:
            continue
            
        output = process_date_residuals(target_data, master_reference_ccf=master_reference)
        collected_dataframes.append(output)
        collected_residuals.append((date, output))

    createNpzFiles(
        collected_dataframes, 
        master_reference, 
        'New_HARPS_ready_for_TF_records.npz'
    )
    print(f"Number of dates: {len(collected_residuals)}")
    # 3. Final Plot
    # plot_combined_residuals(velocity_axis, collected_residuals, opt, title_suffix=f"(Ref: {quiet_date})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # sys.argv[0] is the script name, sys.argv[1] is the first argument
        input_arg = sys.argv[1]
        if input_arg == "help":
            print("Option:\n - help: This help. \n - test: (default) plot residual CCFs of specific dates ['2016-03-29', '2015-09-17', '2017-03-07', '2017-08-13'] to compare to the article. \n - full: plot residual CCFs of the full data set.")
        else:
            run_pipeline(input_arg)
    else:
        run_pipeline()

        
# list_ccf_resid = np.random.rand(len(list_ccf_resid), 49)
# new_list_ccf_resid = list_ccf_resid[:, 1:-2]