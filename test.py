import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import sys
from astropy.io import fits
from scipy.interpolate import interp1d

# Constants for the CCF Grid
# We want a common grid centered at 0 km/s for all observations
TARGET_GRID_POINTS = 161
# Grid range +/- 20 km/s is standard for solar-type stars (covers the line width)
TARGET_RV_GRID = np.linspace(-20, 20, TARGET_GRID_POINTS) 

def get_header_val(header, keys, default=None):
    """Helper to safely extract value from header using multiple possible keys."""
    for k in keys:
        if k in header:
            return header[k]
    return default

def process_single_observation(row, data_dir):
    """
    Reads a FITS file defined in the CSV row and shifts it to the Heliocentric rest frame
    centered at 0 km/s using the CSV provided metadata.
    """
    # Construct file path based on CSV 'date_night' and 'filename'
    # The CSV usually groups files in subfolders named after the date
    date_folder = str(row['date_night'])
    filename = row['filename'].replace(".fits", "_CCF_A.fits")
    filepath = os.path.join(data_dir, date_folder, filename)

    if not os.path.exists(filepath):
        # Fallback: try searching just by filename in the data_dir if structure differs
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            return None

    try:
        with fits.open(filepath) as hdul:
            header = hdul[0].header
            data = hdul[1].data

            # ---------------------------------------------------------
            # 1. READ NATIVE GRID
            # ---------------------------------------------------------
            # We need to know the velocity grid of the FITS file
            step = float(get_header_val(header, ['HIERARCH TNG RV STEP', 'CDELT1', 'step'], 0.82))
            rv_start_header = get_header_val(header, ['HIERARCH TNG RV START', 'CRVAL1', 'start'])
            
            if rv_start_header is None:
                # Fallback calculation if explicit start key missing
                crval = float(get_header_val(header, ['CRVAL1'], 0.0))
                crpix = float(get_header_val(header, ['CRPIX1'], 1.0))
                rv_start = crval - (crpix - 1) * step
            else:
                rv_start = float(rv_start_header)

            # Extract the CCF Profile (Handle 2D order-merged vs 1D arrays)
            # If 2D (N_orders, N_pixels), usually the last row is the combined average
            ccf_profile = data[-1, :] if data.ndim == 2 else data
            native_n = len(ccf_profile)
            
            # Reconstruct the Native Velocity Grid
            native_grid = rv_start + np.arange(native_n) * step

            # ---------------------------------------------------------
            # 2. CALCULATE SHIFT (The Core Logic)
            # ---------------------------------------------------------
            # Per instructions:
            # Shift = Gaussian Fit Velocity (RV_RAW) + Planetary Correction (BERV_TO_HELIO)
            # Note: RV_RAW in the CSV is the measured velocity of the star. 
            # To center the star at 0, we must shift BY that amount.
            
            rv_raw = row['rv_raw'] # The Gaussian centroid of the CCF
            berv_corr = row['berv_bary_to_helio'] # Correction to Heliocentric frame
            
            # The total velocity of the line center in the native frame
            total_velocity_center = rv_raw 
            
            shift_to_zero = total_velocity_center 

            # ---------------------------------------------------------
            # 3. INTERPOLATION (Translational Shifting)
            # ---------------------------------------------------------
            # Create interpolator for the native data
            interpolator = interp1d(native_grid, ccf_profile, kind='cubic', fill_value='extrapolate')
            
            # We evaluate the native grid at (Target_Grid + Center_Velocity) to shift center to 0
            shifted_ccf = interpolator(TARGET_RV_GRID + shift_to_zero)

            # Normalize by the continuum (wings)
            continuum_region = np.concatenate([shifted_ccf[:10], shifted_ccf[-10:]])
            continuum_val = np.median(continuum_region)
            if continuum_val > 0:
                shifted_ccf = shifted_ccf / continuum_val

            return shifted_ccf

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def main():
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    csv_path = "public_release_timeseries.csv"
    base_data_dir = "DATA" # Ensure this points to where your .fits files/folders are
    
    # Template Date (Quiet Sun)
    template_date_str = "2018-03-29" 
    
    # ---------------------------------------------------------
    # 1. LOAD AND FILTER METADATA
    # ---------------------------------------------------------
    print("Loading CSV metadata...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")

    # Quality Filtering
    if df['drs_quality'].dtype == object:
        df['drs_quality'] = df['drs_quality'].map({'TRUE': True, 'FALSE': False, True: True, False: False})

    good_data_mask = (
        ((df['drs_quality'] == True) & 
        (df['obs_quality'] > 0.9)) | (df['date_night'] == template_date_str)
    )
    
    clean_df = df[good_data_mask].copy()
    print(f"Rows after quality filtering: {len(clean_df)}")

    # ---------------------------------------------------------
    # 2. GENERATE TEMPLATE (MASTER CCF)
    # ---------------------------------------------------------
    print(f"\n--- Generating Master Template (Date: {template_date_str}) ---")
    
    clean_df['date_str'] = clean_df['date_night'].astype(str)
    template_rows = clean_df[clean_df['date_str'] == template_date_str]
    
    if len(template_rows) == 0:
        print(f"Error: No good observations found for template date {template_date_str}")
        return

    template_ccfs = []
    print(f"Found {len(template_rows)} observations for template.")

    for idx, row in template_rows.iterrows():
        ccf = process_single_observation(row, base_data_dir)
        if ccf is not None:
            template_ccfs.append(ccf)

    if not template_ccfs:
        print("Failed to process any template files. Check paths.")
        return

    # Create Master CCF (Mean of shifted template observations)
    master_ccf = np.mean(template_ccfs, axis=0)
    
    # ---------------------------------------------------------
    # 3. PROCESS ALL TARGETS AND CALCULATE DELTA CCF
    # ---------------------------------------------------------
    print("\n--- Processing Full Time Series ---")
    
    results = [] 
    
    for idx, row in clean_df.iterrows():
        ccf = process_single_observation(row, base_data_dir)
        
        if ccf is not None:
            # CALCULATE DELTA CCF
            delta_ccf = ccf - master_ccf
            
            results.append({
                'bjd': row['date_bjd'],
                'rv_raw': row['rv_raw'],
                'delta_ccf': delta_ccf
            })
            
            if len(results) % 50 == 0:
                sys.stdout.write(f"\rProcessed {len(results)}/{len(clean_df)}")
                sys.stdout.flush()

    print(f"\nSuccessfully processed {len(results)} observations.")

    # ---------------------------------------------------------
    # 4. VISUALIZATION (Figure 4 Style)
    # ---------------------------------------------------------
    if not results:
        return

    rv_raws = np.array([r['rv_raw'] for r in results])
    mean_rv = np.mean(rv_raws)
    
    # Setup Figure 4 layout (Top: Mean Profile, Bottom: Residuals)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, 
                             gridspec_kw={'height_ratios': [1, 2]})
    
    # PANEL 1: Average Line Profile
    ax1 = axes[0]
    ax1.plot(TARGET_RV_GRID, master_ccf, color='black', linewidth=2)
    ax1.set_ylabel("Normalized Flux")
    ax1.set_title("Average Line Profile (Template)")
    ax1.grid(True, alpha=0.3, linestyle=':')

    # PANEL 2: Residuals colored by RV
    ax2 = axes[1]
    
    print("Plotting residuals... (this may take a moment)")
    
    # Plotting loop
    # Using low alpha because thousands of lines can obscure structure
    for res in results:
        rv = res['rv_raw']
        delta = res['delta_ccf']
        
        # Color Logic: Red if Redshifted relative to mean, Blue if Blueshifted
        color = 'r' if rv > mean_rv else 'b'
        
        ax2.plot(TARGET_RV_GRID, delta, color=color, alpha=0.05, linewidth=1)

    ax2.set_ylabel("Residual Flux ($\Delta$CCF)")
    ax2.set_xlabel("Velocity (km/s)")
    ax2.set_title("Residuals Colored by RV Shift")
    ax2.set_xlim(-20, 20)
    
    # Add horizontal zero line
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color='r', label='Redshifted (RV > Mean)'),
        Line2D([0], [0], color='b', label='Blueshifted (RV < Mean)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()