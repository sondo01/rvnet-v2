import numpy as np
import os
import glob
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy import units as u
from scipy.interpolate import interp1d
import pandas as pd

# --- 1. PROPER REFLEX MOTION CALCULATION ---
def calculate_solar_reflex_velocity(bjd_tdb):
    """
    Calculates the Sun's radial velocity relative to the Solar System Barycenter (SSB)
    as observed from Earth. This is the 'Reflex Motion' caused by Jupiter/Saturn.
    
    Zoe-P1 Method:
    "We transform... from the barycentric to the heliocentric reference frame...
    subtracting the Sun's barycentric motion."
    """
    try:
        # Use built-in ephemeris (sufficient for ~m/s precision) or 'jpl' if available
        # Ideally, one would use 'de430' or 'de432s', but 'builtin' is safe for fallback
        ephemeris = 'builtin' 
        
        t = Time(bjd_tdb, format='jd', scale='tdb')
        
        # 1. Get Sun's position and velocity relative to SSB
        # Note: 'sun' in get_body_barycentric... returns pos/vel of Sun relative to SSB
        with solar_system_ephemeris.set(ephemeris):
            pos_sun, vel_sun = get_body_barycentric_posvel('sun', t)
            pos_earth, _ = get_body_barycentric_posvel('earth', t)
            
        # 2. Calculate Line-of-Sight Vector (Earth -> Sun)
        # Vector from Earth to Sun
        r_vec = pos_sun - pos_earth
        r_unit = r_vec / r_vec.norm()
        
        # 3. Project Sun's Barycentric Velocity onto Line-of-Sight
        # Dot product: v_sun . r_unit
        # We want the radial component. Positive = receding from Earth.
        v_reflex = vel_sun.xyz.dot(r_unit.xyz)
        
        # Convert to km/s (standard for FITS)
        return v_reflex.to(u.km / u.s).value
        
    except Exception as e:
        print(f"Warning: Astropy ephemeris calculation failed ({e}). Returning 0.0.")
        return 0.0

# --- 2. ROBUST LOADING & FILTERING ---
def load_harps_ccf_native(folder_path):
    """
    Loads HARPS-N CCF FITS files.
    Reads strict quality flags for filtering.
    """
    file_list = glob.glob(os.path.join(folder_path, "*.fits"))
    file_list.sort()
    
    if not file_list:
        return None

    dataset = []

    for f in file_list:
        try:
            with fits.open(f) as hdul:
                # Handle different FITS structures (Target vs Primary)
                target_hdu = hdul[0] if (hdul[0].data is not None and hdul[0].data.size > 0) else hdul[1]
                header = target_hdu.header
                data = target_hdu.data
                
                # Helper to get keywords
                def get_val(keys, default=None):
                    for h in [header, hdul[0].header]:
                        for k in keys:
                            if k in h: return h[k]
                    return default

                # --- Time & RV ---
                bjd = get_val(['HIERARCH TNG QC BJD', 'BJD', 'MJD-OBS'])
                rv_raw = get_val(['HIERARCH TNG QC CCF RV', 'RV_RAW', 'HIERARCH ESO QC CCF RV']) 
                # Note: headers often label the Barycentric RV as just 'CCF RV' or 'RV_RAW'.
                
                if bjd is None or rv_raw is None: continue
                
                # --- Quality Parameters (Zoe-P1 / Cameron-2021) ---
                # Quality Flag: > 0.9 is GOOD (Cloud Free)
                obs_quality = get_val(['HIERARCH TNG QC OBS_QUALITY', 'OBS_QUALITY', 'SPECTRO_ANALYSIS_QUALFLAG'], 0.0)
                
                # Extinction: Should be < 0.1 m/s (check units!)
                # DACE headers use 'RV_DIFF_EXTINCTION' (m/s). HARPS headers use 'HIERARCH TNG QC EXT CORR'
                extinction = get_val(['HIERARCH TNG QC EXT CORR', 'RV_DIFF_EXTINCTION'], 100.0)
                
                snr = get_val(['HIERARCH TNG QC SPECTRO SNR50', 'SNR50', 'SNR'], 0.0)

                # --- Grid Construction ---
                step = float(get_val(['HIERARCH TNG RV STEP', 'CDELT1', 'DELTAV'], 0.25))
                start = float(get_val(['HIERARCH TNG RV START', 'CRVAL1', 'STARTV']))
                
                # Handle 2D data (Order-merged vs Single order)
                if data.ndim == 2:
                    ccf_profile = data[-1, :] # Usually last row is merged CCF
                else:
                    ccf_profile = data
                
                if not np.isfinite(ccf_profile).all():
                    ccf_profile = np.nan_to_num(ccf_profile, nan=np.nanmedian(ccf_profile))

                n_pixels = len(ccf_profile)
                if start is None:
                    crpix = float(get_val(['CRPIX1'], n_pixels // 2))
                    start = 0.0 - (crpix * step) # Approximate center if missing

                native_grid = start + np.arange(n_pixels) * step

                dataset.append({
                    'filename': os.path.basename(f),
                    'bjd': float(bjd),
                    'rv_raw': float(rv_raw), # Barycentric RV
                    'ccf': ccf_profile.astype(np.float64),
                    'grid': native_grid.astype(np.float64),
                    'snr': float(snr),
                    'obs_quality': float(obs_quality),
                    'extinction': float(extinction)
                })

        except Exception as e:
            # print(f"Skipping {f}: {e}")
            continue

    return dataset

def filter_observations(dataset, csv_path='public_release_timeseries.csv'):
    """
    Filters a list of observation dictionaries based on metadata stored in a CSV file.
    
    The function excludes observations where:
    - rv_diff_extinction < 0.1
    - AND obs_quality > 0.99
    
    Parameters:
    - dataset (list): A list of dictionaries, each containing a 'filename' key.
    - csv_path (str): Path to the public_release_timeseries.csv file.
    
    Returns:
    - list: The filtered dataset.
    """
    
    # 1. Load the metadata from the CSV file
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Returning original dataset.")
        return dataset
        
    # We only need specific columns to save memory
    try:
        df_meta = pd.read_csv(csv_path, usecols=['filename', 'rv_diff_extinction', 'obs_quality'])
    except ValueError as e:
        print(f"Error reading CSV: {e}")
        return dataset

    # 2. Identify the files that should be EXCLUDED
    # Criteria: rv_diff_extinction < 0.1 AND obs_quality > 0.99
    exclusion_mask = (df_meta['rv_diff_extinction'] < 0.1) & (df_meta['obs_quality'] > 0.9)
    excluded_filenames = set(df_meta.loc[exclusion_mask, 'filename'])

    # 3. Filter the dataset
    # We keep 'd' if its filename is NOT in the excluded set
    filtered_dataset = [
        d for d in dataset 
        if d.get('filename') not in excluded_filenames
    ]
    
    # Print a small summary
    original_count = len(dataset)
    filtered_count = len(filtered_dataset)
    print(f"Filtering complete: {original_count - filtered_count} observations removed.")
    print(f"Remaining observations: {filtered_count}")

    return filtered_dataset
# --- 3. THE 2-STEP SHIFTING LOGIC ---
def align_and_resample_proper(dataset, target_grid):
    """
    Performs the 2-step coordinate shift:
    1. Barycentric -> Heliocentric (Remove Solar System Planets)
    2. Heliocentric -> Rest Frame (Remove Local Activity)
    """
    if not dataset: return None, None, None

    aligned_ccfs = []
    bjds = []
    snrs = []
    
    for data in dataset:
        native_grid = data['grid']
        native_ccf = data['ccf']
        bjd = data['bjd']
        
        # --- Step 1: Barycentric -> Heliocentric ---
        # Calculate the Sun's reflex velocity due to planets
        # Note: If header has 'BERV_BARY_TO_HELIO', use that. Otherwise calculate.
        # We calculate it here to be "proper" and robust.
        v_reflex_ssb = calculate_solar_reflex_velocity(bjd) # km/s
        
        # Shift the GRID to the Heliocentric frame.
        # If the Star is moving at v_reflex relative to Barycenter, 
        # the Heliocentric frame is (v_bary - v_reflex).
        # We subtract v_reflex from the grid coordinates.
        helio_grid = native_grid - v_reflex_ssb
        
        # --- Step 2: Heliocentric -> Rest Frame ---
        # We now want to center the line at 0. 
        # The remaining shift is the "Activity Induced RV".
        # We determine this by fitting the CCF *in the Heliocentric frame*.
        
        # Interpolate onto a temporary dense grid to find the minimum/center
        temp_interpolator = interp1d(helio_grid, native_ccf, kind='cubic', fill_value="extrapolate")
        
        # Find the minimum (approximate RV activity)
        # Search range limited to likely activity shifts (+/- 5 km/s around 0)
        search_grid = np.linspace(-5, 5, 1000)
        temp_profile = temp_interpolator(search_grid)
        min_idx = np.argmin(temp_profile)
        rv_activity = search_grid[min_idx]
        
        # Refine with a simplified gaussian fit or quadratic peak finding could be done here
        # For this pipeline, the minimum is a robust estimator for the "shift to zero" step
        
        # --- Final Transformation ---
        # We sample the original CCF at: Target_Grid + v_reflex + rv_activity
        # This effectively shifts the CCF to 0 in the rest frame.
        
        final_shift_points = target_grid + v_reflex_ssb + rv_activity
        
        # Interpolate from NATIVE grid directly to avoid double interpolation errors
        final_interpolator = interp1d(native_grid, native_ccf, kind='cubic', fill_value="extrapolate")
        aligned_ccf = final_interpolator(final_shift_points)
        
        # --- Normalization (Continuum) ---
        # Normalize by wings (outer 15 pixels)
        continuum = np.median(np.concatenate([aligned_ccf[:15], aligned_ccf[-15:]]))
        if continuum > 0:
            aligned_ccf /= continuum
            
        aligned_ccfs.append(aligned_ccf)
        bjds.append(bjd)
        snrs.append(data['snr'])

    return np.array(aligned_ccfs), np.array(bjds), np.array(snrs)

def compute_weighted_master(ccf_array, snr_array):
    weights = snr_array ** 2
    if np.sum(weights) == 0: return np.median(ccf_array, axis=0)
    weights_matrix = weights[:, np.newaxis]
    return np.sum(ccf_array * weights_matrix, axis=0) / np.sum(weights_matrix, axis=0)