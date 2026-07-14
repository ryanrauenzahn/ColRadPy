"""
** IN DEVELOPMENT **

Goal: 
    To generate a 3-dimensional plot with axes temperature, density, and intensity.

Outline:
    1. Import required libraries and define file paths.
    2. Create the temperature and density parameter grids. Temp. range (0-15 eV), density range (1e12-1e15, stepwise).
    3. Load the carbon atomic data into ColRadPy.
    4. Compute the ionization balance across the parameter space.
    5. Calculate the selected spectral line intensity for every temperature-density combination.
    6. Store the calculated intensities in a two-dimensional array.
    7. Generate a three-dimensional surface plot of the results.
    8. Formate and display the completed figure.
    
    
Sections:
    1. Paths: defines the paths to the ADAS data files in this same root directory (located in the atomic_data folder). Files are in the ADF04 form and are named like "mom97_ls#c0.dat". 
    2. Variables: 
    3. File Check: small function that checks to see if every required ADF04 file exists.

Author: Ryan Rauenzahn
Date: 07/14/2026

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from colradpy import colradpy
from colradpy.ionization_balance_class import ionization_balance

# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ATOMIC_DATA_DIR = PROJECT_ROOT / "atomic_data"

FILES = [
    str(ATOMIC_DATA_DIR / "mom97_ls#c0.dat"), # C I
    str(ATOMIC_DATA_DIR / "mom97_ls#c1.dat"),
    str(ATOMIC_DATA_DIR / "mom97_ls#c2.dat"),
    str(ATOMIC_DATA_DIR / "mom97_ls#c3.dat"),
    str(ATOMIC_DATA_DIR / "mom97_ls#c4.dat"),
    str(ATOMIC_DATA_DIR / "mom97_ls#c5.dat"), # C VI
]

# ============================================================
# Variables
# ============================================================

# Plasma conditions

TE = np.linspace(1.0, 100, 15)

METAS = [
    np.array([0]),
    np.array([0, 1]),
    np.array([0, 1]),
    np.array([0]),
    np.array([0, 1]),
    np.array([0]),
    
]

# Physical Constants

EV_TO_ERG = 1.60218e-12


#============================================================
# File Check
# ============================================================

def check_input_files(files):
    """ Verify that every required ADF04 file exists """
    for file in files:
        if not Path(file).exists():
            raise FileNotFoundError(
                f"Missing ADF04 file: {file}"
            )







# ============================================================
# Main
# ============================================================

def main():
    check_input_files(FILES)
    
    
    
    print("\nDone")
    
if __name__ == "__main__":
    main()