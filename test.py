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
    print("Success 1")
except ImportError:
    # Fallback if master_shifting.py is in the same directory
    try:
        from master_shifting import master_shifting
        print("Success 2")
    except ImportError:
        print("Error 3")
        print("Warning: master_shifting module not found.")

# Add the current directory to path to ensure local imports work

