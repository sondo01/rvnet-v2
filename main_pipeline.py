import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure local imports work
sys.path.append(os.getcwd())

import pipeline_utils as pipe

def run_proper_pipeline():
    base_data_dir = "DATA"
    
    # Standard Grid (Zoe-P1)
    TARGET_GRID = np.linspace(-20, 20, 161)
    
    # --- 1. Master Reference (Quiet Star) ---
    # Zoe-P1 Paper Explicit Date: 2018-03-29
    quiet_date = "2018-03-29"
    print(f"\n--- Generating Master Reference from {quiet_date} ---")
    
    ref_path = os.path.join(base_data_dir, quiet_date)
    
    # Load
    ref_raw = pipe.load_harps_ccf_native(ref_path)
    if not ref_raw:
        print(f"Error: Could not find reference data in {ref_path}")
        return

    # Filter (Strict)
  
    ref_clean = pipe.filter_observations(ref_raw)
    
    if not ref_clean:
        print("Error: All reference frames filtered out.")
        return

    # Align & Resample (The 2-step proper shift)
    ref_ccfs, _, ref_snrs = pipe.align_and_resample_proper(ref_clean, TARGET_GRID)
    
    # Weighted Master
    master_reference_ccf = pipe.compute_weighted_master(ref_ccfs, ref_snrs)
    print("Master Reference created.")

    # --- 2. Process Target Dates ---
    target_dates = ["2016-03-29","2015-09-17", "2017-03-07", "2017-08-13"]
    
    for date in target_dates:
        print(f"\n--- Processing Target Date: {date} ---")
        path = os.path.join(base_data_dir, date)
        
        target_raw = pipe.load_harps_ccf_native(path)
        if not target_raw: continue
        
        target_clean = pipe.filter_observations(target_raw)
        if not target_clean:
            print(f"Skipping {date}: Filtered out.")
            continue
            
        # Align (Proper 2-step)
        target_ccfs, _, target_snrs = pipe.align_and_resample_proper(target_clean, TARGET_GRID)
        
        # Calculate Residuals (Observed - Master)
        # Note: Both are now in Rest Frame (centered at 0), so direct subtraction is valid
        individual_residuals = target_ccfs - master_reference_ccf
        
        # Daily Master
        daily_master = pipe.compute_weighted_master(target_ccfs, target_snrs)
        daily_residual = daily_master - master_reference_ccf
        
        # --- Visualization ---
        plt.figure(figsize=(10, 6))
        plt.plot(TARGET_GRID, individual_residuals.T, color='grey', alpha=0.1)
        plt.plot(TARGET_GRID, daily_residual, color='red', linewidth=2, label='Daily Mean Residual')
        plt.title(f"Residuals: {date}\n(Heliocentric Corrected -> Rest Frame)")
        plt.xlabel("Velocity (km/s)")
        plt.ylabel("Normalized Flux Delta")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    run_proper_pipeline()