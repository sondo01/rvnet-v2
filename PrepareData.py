import numpy as np
import os
from astropy.io import fits
import sys
from scipy import stats
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
import logging

# Set up logging to print to console with a specific format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to sys.path to ensure local imports (like rv_net) work correctly
sys.path.append(os.getcwd())

try:
    from rv_net.master_shifting import master_shifting
except ImportError:
    # Warning if the external dependency is missing. 
    # This allows the script to be imported/checked even if dependencies aren't fully set up.
    logger.warning("Could not import rv_net.master_shifting. Ensure the module is in the python path.")
    master_shifting = None

class HARPSDataPipeline:
    """
    Pipeline for processing HARPS-N CCF (Cross-Correlation Function) data.
    
    This class handles:
    1. Loading FITS files from nightly directories.
    2. Extracting RV (Radial Velocity), FWHM, Contrast, and Bisector Span values.
    3. Aggregating nightly observations into weighted averages.
    4. Correcting for planetary signals (using 'master_shifting').
    5. Generating a dataset suitable for Deep Learning models (saved as .npz).
    """
    
    # Constants used for grid resampling and default steps
    TARGET_GRID_POINTS = 49
    TARGET_RV_GRID = np.linspace(-20, 20, TARGET_GRID_POINTS)
    DEFAULT_STEP_SIZE = 0.82
    
    def __init__(self, base_data_dir: str = "DATA", quiet_date: str = "2018-03-29", 
                 public_release_file: str = 'public_release_timeseries.csv'):
        """
        Initialize the pipeline.

        Args:
            base_data_dir (str): Path to the directory containing date-based subdirectories of FITS files.
            quiet_date (str): The date string (YYYY-MM-DD) identified as a 'quiet' star day, used as a reference.
            public_release_file (str): Path to the CSV file containing public released RV data.
        """
        self.base_data_dir = Path(base_data_dir)
        self.quiet_date = quiet_date
        self.public_release_file = Path(public_release_file)
        self.public_release_values: Optional[pd.DataFrame] = None
        
    def load_public_release_data(self) -> None:
        """
        Loads the public release timeseries data from the CSV file.
        
        This data provides validated RV, FWHM, Contrast, and Bisector values 
        which are preferred over values in the FITS headers for consistency.
        """
        if not self.public_release_file.exists():
            logger.error(f"Public release file not found: {self.public_release_file}")
            return

        self.public_release_values = pd.read_csv(self.public_release_file, index_col='filename')
        # Remove any leading/trailing spaces from column names to avoid KeyErrors
        self.public_release_values.columns = self.public_release_values.columns.str.strip()
        logger.info(f"Loaded public release data from {self.public_release_file}")

    @staticmethod
    def get_header_val(header: fits.Header, keys: List[str], default: Any = None) -> Any:
        """
        Helper method to safely extract a value from a FITS header using multiple possible keys.
        
        Args:
            header (fits.Header): The FITS header object.
            keys (List[str]): A list of potential keys to look for.
            default (Any): The value to return if none of the keys are found.

        Returns:
            The value associated with the first matching key, or the default value.
        """
        for k in keys:
            if k in header:
                return header[k]
        return default

    def calculate_weight(self, header: fits.Header) -> float:
        """
        Calculates the weight of an observation based on Spectral S/N or RV Error.
        
        Priority:
        1. Spectral S/N (Order 69) -> Weight = SNR^2
        2. RV Error -> Weight = 1 / (RV_Error^2)
        3. Default -> 1.0

        Args:
            header (fits.Header): FITS header containing quality control metrics.

        Returns:
            float: The calculated weight.
        """
        # Try to get Spectral S/N (Order 69 for combined CCF)
        snr = self.get_header_val(header, ['HIERARCH TNG QC SPECTRO SN69', 'HIERARCH TNG QC ORDER69 SNR', 'SNR'], None)
        
        # Try to get RV Error
        rv_err = self.get_header_val(header, ['HIERARCH TNG QC CCF RV ERROR', 'HIERARCH ESO QC CCF RV ERROR'], None)
        
        if snr is not None and float(snr) > 0:
            return float(snr) ** 2
        elif rv_err is not None and float(rv_err) > 0:
            return 1.0 / (float(rv_err) ** 2)
        return 1.0

    def process_single_fits(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """
        Reads and processes a single FITS file.
        
        Extracts BJD, RV, CCF profile, and other parameters. 
        It uses the external 'public_release_values' dataframe to source accurate RV parameters.

        Args:
            filepath (Path): Path to the FITS file.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing extracted parameters, 
                                      or None if the file is invalid or missing corresponding release data.
        """
        if self.public_release_values is None:
            logger.error("Public release data not loaded. Call load_public_release_data() first.")
            return None

        fname = filepath.name
        # The file name in the release file doesn't contain "_CCF_A", so we strip it.
        release_name = fname.replace("_CCF_A.fits", ".fits") 
        
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                data = hdul[1].data
                
                # Extract Params from Header
                bjd = self.get_header_val(header, ['MJD-OBS', 'BJD', 'HIERARCH TNG QC BJD'])
                
                # Get RV and other params from release file (convert m/s to km/s)
                try:
                    row = self.public_release_values.loc[release_name]
                except KeyError:
                    # logger.warning(f"File {release_name} not found in public release data.")
                    return None

                # Conversions from m/s to km/s
                rv = row['rv'] / 1000.0
                fwhm = row['fwhm'] / 1000.0
                cont = row['contrast'] / 1000.0 
                bis = row['bis_span'] / 1000.0
                berv = row['berv_bary_to_helio'] / 1000.0
                rv_diff_extinction = row['rv_diff_extinction'] / 1000.0
                
                # Basic Validation
                if any(v is None for v in [bjd, rv, fwhm, cont, berv, rv_diff_extinction]):
                    return None
                
                if float(fwhm) == 0.0 or float(cont) == 0.0:
                    return None
                
                # Compute removed_planet_rvs for CCF shifting
                removed_planet_rvs = berv + rv_diff_extinction
                
                # Grid Parameters
                step = float(self.get_header_val(header, ['HIERARCH TNG RV STEP', 'CDELT1'], self.DEFAULT_STEP_SIZE))
                rv_start = float(self.get_header_val(header, ['HIERARCH TNG RV START'], 0.0))

                # CCF Processing
                # The data might be a 2D array (Orders x Pixels) or 1D (Combined). source uses last row for combined.
                ccf_profile = data[-1, :] if data.ndim == 2 else data
                native_n = len(ccf_profile)        
                ccf_profile = ccf_profile.astype(np.float64)

                weight = self.calculate_weight(header)

                return {
                    'bjd': float(bjd),
                    'rv': float(rv),
                    'removed_planet_rvs': float(removed_planet_rvs),
                    'ccf': ccf_profile,
                    'fwhm': float(fwhm),
                    'cont': float(cont),
                    'bis': float(bis),
                    'weight': weight,
                    'step': step,
                    'native_n': native_n,
                    'rv_start': rv_start
                }
        except Exception as e:
            logger.error(f"Error reading {filepath.name}: {e}")
            return None

    def compute_nightly_average(self, nightly_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Computes the weighted average for a list of observations (usually from the same night).
        
        Args:
            nightly_data (List[Dict]): A list of dictionaries, where each dict is a processed observation.

        Returns:
            Optional[Dict]: A dictionary containing the weighted partial averages (RV, FWHM, CCF, etc.),
                            or None if weights sum to zero.
        """
        # Be explicit about what to stack to ensure correct types/shapes
        stack_keys = ['bjd', 'rv', 'fwhm', 'cont', 'bis', 'weight', 'ccf', 'removed_planet_rvs']
        arrays = {k: np.array([d[k] for d in nightly_data]) for k in stack_keys}
        
        weights = arrays['weight']
        sum_weights = np.sum(weights)
        
        if sum_weights == 0:
            return None

        # Calculate Weighted Averages for scalar values
        avg_data = {}
        for k in ['bjd', 'rv', 'fwhm', 'cont', 'bis', 'removed_planet_rvs']:
            avg_data[k] = np.sum(arrays[k] * weights) / sum_weights
        
        # Calculate Weighted Average for CCF (Vectorized Broadcasting)
        # Shape: (N_obs, N_pixels) * (N_obs, 1) -> Sum across observations -> (N_pixels,)
        avg_data['ccf'] = np.sum(arrays['ccf'] * weights[:, np.newaxis], axis=0) / sum_weights

        return avg_data

    def load_harps_ccf_data(self, folder_path: Path) -> Optional[Dict[str, Any]]:
        """
        Iterates through FITS files in a directory, processes them, and bins them into nightly averages.
        
        Args:
            folder_path (Path): Directory containing the FITS files for a specific period/target.

        Returns:
            Optional[Dict]: A dictionary containing arrays of aggregated data (BJD, RV, CCF, etc.) 
                            ready for the shifting pipeline.
        """
        file_list = sorted(list(folder_path.glob("*.fits")))
        if not file_list:
            logger.warning(f"No FITS files found in {folder_path}")
            return None

        logger.info(f"Loading {len(file_list)} files from {folder_path}...")

        # Group observations by Night (Night ID is typically the folder name itself)
        night_groups: Dict[str, List[Dict[str, Any]]] = {}
        step_list = []
        native_n_list = []

        valid_count = 0
        for f in file_list:
            obs = self.process_single_fits(f)
            if obs:
                night_id = folder_path.name 
                if night_id not in night_groups:
                    night_groups[night_id] = []
                night_groups[night_id].append(obs)
                
                step_list.append(obs['step'])
                native_n_list.append(obs['native_n'])
                valid_count += 1
            # else:
            #     print("No obs")

        if valid_count == 0:
            return None

        logger.info(f"Binning {valid_count} exposures into {len(night_groups)} nights...")

        # Compute averages for each night and collect them into lists
        final_data = {k: [] for k in ['bjd', 'rvh', 'ccfBary', 'fwhm', 'cont', 'bis', 'removed_planet_rvs']}
        
        for night in sorted(night_groups.keys()):        
            avg = self.compute_nightly_average(night_groups[night])
            
            if avg:
                final_data['bjd'].append(avg['bjd'])
                final_data['rvh'].append(avg['rv'])
                final_data['ccfBary'].append(avg['ccf'])
                final_data['fwhm'].append(avg['fwhm'])
                final_data['cont'].append(avg['cont'])
                final_data['bis'].append(avg['bis'])
                final_data['removed_planet_rvs'].append(avg['removed_planet_rvs'])

        # Convert lists to appropriate numpy arrays for downstream processing
        return {
            'bjd': np.array(final_data['bjd'], dtype=np.float64),
            'rvh': np.array(final_data['rvh'], dtype=np.float64),
            'removed_planet_rvs': np.array(final_data['removed_planet_rvs'], dtype=np.float64),
            'ccfBary': final_data['ccfBary'], # List of arrays
            'fwhm': np.array(final_data['fwhm'], dtype=np.float64),
            'cont': np.array(final_data['cont'], dtype=np.float64),
            'bis': np.array(final_data['bis'], dtype=np.float64),
            'step': 0.82, # Resampled Grid Step
            'native_step': np.median(step_list) if step_list else self.DEFAULT_STEP_SIZE,
            'native_n': int(stats.mode(native_n_list, keepdims=True)[0][0]) if native_n_list else 49
        }

    def shift_to_zero(self, data_dict: Dict[str, Any], is_reference_generation: bool = False) -> Union[pd.DataFrame, np.ndarray, None]:
        """
        Runs the 'master_shifting' pipeline on the dataset.
        
        This process aligns CCFs to a common frame, removes planetary signals, and performs
        shifting (to zero or median) to isolate the stellar activity signal.

        Args:
            data_dict (Dict): The dictionary of data returned by `load_harps_ccf_data`.
            is_reference_generation (bool): If True, returns only the first CCF (used for master reference).
                                            If False, returns the full output DataFrame.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Processed DataFrame or specific CCF array.
        """
        # Ensure master_shifting is available
        if master_shifting is None:
            logger.error("master_shifting function is not available.")
            return None

        # Execute the external master_shifting function
        output = master_shifting(
            bjd=data_dict['bjd'],
            ccfBary=data_dict['ccfBary'],
            rvh=data_dict['rvh'],
            ref_frame_shift="off", 
            removed_planet_rvs=data_dict['removed_planet_rvs'], 
            zero_or_median="zero", 
            fwhm=data_dict['fwhm'],
            cont=data_dict['cont'],
            bis=data_dict['bis']
        )

        # Extract Shifted CCFs
        # output is a DataFrame from master_shifting that contains 'CCF_normalized_list'
        aligned_normalized_ccfs = output['CCF_normalized_list'].tolist()

        if is_reference_generation:
            # For the reference date (a quiet star date), we only need the first CCF to serve as the master reference
            return aligned_normalized_ccfs[0]
        else:
            return output

    def _stack_column(self, full_df: pd.DataFrame, col_name: str) -> np.ndarray:
        """
        Helper method to stack columns that contain arrays, handling potentially jagged arrays 
        (different lengths) by trimming to the minimum common length.

        Args:
            full_df (pd.DataFrame): The DataFrame containing the column.
            col_name (str): The name of the column to stack.

        Returns:
            np.ndarray: A stacked 2D numpy array.
        """
        # Extract, flatten to 1D (fixes (N,1) vs (N,) mismatches), and trim
        raw_values = [np.array(x).flatten() for x in full_df[col_name]]
        
        # Determine the minimum length among all rows to handle ragged arrays
        lengths = [len(x) for x in raw_values]
        if not lengths:
            return np.array([])
        min_len = min(lengths)
        
        # If lengths vary, warn the user
        if len(set(lengths)) > 1:
            logger.warning(f"Inhomogeneous lengths detected in column '{col_name}'. "
                  f"Range: {min(lengths)} - {max(lengths)}. Trimming to {min_len}.")
            
        # Trim all arrays to the minimum length
        trimmed_values = [x[:min_len] for x in raw_values]
            
        return np.vstack(trimmed_values)

    def _normalize_arr(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalizes an array (or matrix) by subtracting the median and dividing by the standard deviation.
        
        Args:
            arr (np.ndarray): Input array.

        Returns:
            np.ndarray: Normalized array.
        """
        med = np.median(arr, axis=0)
        std = np.std(arr, axis=0)
        # Handle zero std to avoid division by zero
        if np.isscalar(std):
            if std == 0: std = 1.0
        else:
            std[std == 0] = 1.0
        return (arr - med) / std

    def create_npz_files(self, dataframe_list: List[pd.DataFrame], master_reference_ccf: np.ndarray, outfile_name: str) -> None:
        """
        Final aggregation step.
        
        Concatenates all processed DataFrames, computes residuals against the master reference, 
        performs normalization, and saves the final dataset to a .npz file.

        Args:
            dataframe_list (List[pd.DataFrame]): List of DataFrames from `shift_to_zero`.
            master_reference_ccf (np.array): The master reference CCF (quiet star).
            outfile_name (str): The filename for the output .npz file.
        """
        if not dataframe_list:
            logger.warning("No data collected to save.")
            return

        # 1. Concatenate all daily DataFrames into one master DataFrame
        full_df = pd.concat(dataframe_list, ignore_index=True)

        logger.info(f"Aggregating data... Total observations: {len(full_df)}")

        # 3. Extract Basic Columns
        bjd = full_df['BJD'].to_numpy(dtype=np.float64)
        vrad_star = full_df['vrad_star'].to_numpy(dtype=np.float64) # This is 'rvh'
        fwhm = full_df['fwhm'].to_numpy(dtype=np.float64)
        cont = full_df['cont'].to_numpy(dtype=np.float64)
        bis = full_df['bis'].to_numpy(dtype=np.float64)

        # 4. Extract CCF Lists (Full Size) and handle potential jaggedness
        og_ccf_list = self._stack_column(full_df, 'og_ccf_list')
        jup_shifted_CCF_data_list = self._stack_column(full_df, 'jup_shifted_CCF_data_list')
        zero_shifted_CCF_list = self._stack_column(full_df, 'zero_shifted_CCF_list')
        CCF_normalized_list = self._stack_column(full_df, 'CCF_normalized_list')

        # 5. Extract Shift Parameters
        mu_og_list = full_df['mu_og_list'].to_numpy(dtype=np.float64)
        mu_jup_list = full_df['mu_jup_list'].to_numpy(dtype=np.float64)
        mu_zero_list = full_df['mu_zero_list'].to_numpy(dtype=np.float64)
        
        # 6. Shift by RV (Placeholder or scalar as per requirements)
        shift_by_rv = np.array(0.0) 

        # 7. Compute Residuals
        master_reference_ccf = np.array(master_reference_ccf)

        if CCF_normalized_list.ndim > 1:
            current_ccf_len = CCF_normalized_list.shape[1]
            
            # Verify compatibility with master reference
            if master_reference_ccf.shape[0] != current_ccf_len:
                logger.warning(f"master_reference_ccf length ({master_reference_ccf.shape[0]}) "
                      f"does not match data length ({current_ccf_len}). Trimming master ref.")
                master_reference_ccf = master_reference_ccf[:current_ccf_len]

            # Subtract master reference (quiet CCF) from normalized CCF to isolate activity
            cff_residual_list = CCF_normalized_list - master_reference_ccf
            
            # Create cutoffs (removing edges)
            CCF_normalized_list_cutoff = CCF_normalized_list[:, 1:-2]
            CCF_residual_list_cutoff = cff_residual_list[:, 1:-2]

            # Rescale/Standardize the residuals
            ccf_residual_rescaled = self._normalize_arr(cff_residual_list)
            ccf_residual_rescaled_cutoff = ccf_residual_rescaled[:, 1:-2]
        else:
            # Handle edge case where CCF lists might be empty
            logger.warning("CCF_normalized_list has unexpected shape. Skipping residual computation.")
            cff_residual_list = np.array([])
            CCF_normalized_list_cutoff = np.array([])
            CCF_residual_list_cutoff = np.array([])
            ccf_residual_rescaled = np.array([])
            ccf_residual_rescaled_cutoff = np.array([])

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

        logger.info(f"Successfully saved {outfile_name}")

    def run(self, opt: str = "test") -> None:
        """
        Main runner method for the pipeline.
        
        Args:
            opt (str): Option for data selection. 
                       "full" -> Process all dates.
                       "test" -> Process a hardcoded subset of dates for verification.
        """
        if not self.base_data_dir.exists():
            logger.error(f"Directory '{self.base_data_dir}' not found.")
            return

        # Load public release data first
        self.load_public_release_data()
        if self.public_release_values is None:
            return

        # Manage option to select target dates
        if opt == "full":
            all_subdirs = sorted([d.name for d in self.base_data_dir.iterdir() if d.is_dir()])
            target_dates = [d for d in all_subdirs if d != self.quiet_date]
        elif opt == "test":
            target_dates = ["2015-07-29", "2016-03-29", "2015-09-17", "2017-03-07", "2017-08-13"]
        else:
            # Fallback or other options
            target_dates = []
            logger.warning(f"Unknown option: {opt}. Using empty target list.")
    
        logger.info(f"Found {len(target_dates)} target dates.")
        logger.info("--- Starting Pipeline ---")
        
        # 1. Generate Master Reference (from the quiet date)
        logger.info(f"Processing Reference: {self.quiet_date}")
        ref_path = self.base_data_dir / self.quiet_date
        
        if not ref_path.exists():
            logger.error(f"Reference directory {ref_path} does not exist.")
            return
            
        ref_data = self.load_harps_ccf_data(ref_path)

        if ref_data is None:
            logger.error("Could not load reference data. Aborting.")
            return

        master_reference = self.shift_to_zero(ref_data, is_reference_generation=True)

        # 2. Process Targets
        collected_residuals = []
        collected_dataframes = []

        for date in target_dates:
            logger.info(f"Processing Target: {date}")
            target_path = self.base_data_dir / date
            
            if not target_path.exists():
                logger.warning(f"Target path {target_path} does not exist. Skipping.")
                continue
                
            target_data = self.load_harps_ccf_data(target_path)
            
            if target_data is None:
                continue
                
            output = self.shift_to_zero(target_data)
            collected_dataframes.append(output)
            collected_residuals.append((date, output))

        # 3. Create .npz file
        self.create_npz_files(
            collected_dataframes, 
            master_reference, 
            'New_HARPS_ready_for_TF_records.npz'
        )
        logger.info(f"Number of dates processed: {len(collected_residuals)}")

if __name__ == "__main__":
    pipeline = HARPSDataPipeline()
    
    if len(sys.argv) > 1:
        input_arg = sys.argv[1]
        if input_arg == "help":
            print("Usage: python PrepareData.py [option]")
            print("Options:")
            print("  test  : (default) plot residual CCFs of specific dates.")
            print("  full  : plot residual CCFs of the full data set.")
        else:
            pipeline.run(input_arg)
    else:
        pipeline.run()
