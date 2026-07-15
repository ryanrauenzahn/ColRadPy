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
    1. Paths: defines the relative paths to the ADAS data files in this same root directory (located in the atomic_data folder). Files are in the ADF04 form and are named like "mom97_ls#c0.dat". 
    2. Variables: define temp_grid to be from 0.1 eV to 15 eV and dens_grid to be 1e12 through 1e15, evenly spaced. So there will be (150)*(28)=4200 unique temperature-density combinations.
    3. File Check: small function that checks to see if every required ADF04 file exists.
    4. Calculate ionization balance: ColRadPy constructs the ionization balance matrix by passing in all the variables supplied to it and then it solves the no source ionization balance problem. No source menas there's not a continuous injection of fresh carbon into the system. Stored in the ion_bal ojbect are the fractions (like f_CIII(T_e, n_e)).
    
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

files = [
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

# Plasma conditions / Temperature and density grids

temp_grid = np.linspace(0.1, 15.0, 150)

density_grid = np.concatenate([
    np.arange(1, 10) * 1.0e12,
    np.arange(1, 10) * 1.0e13,
    np.arange(1, 10) * 1.0e14,
    np.array([1e15])
])

metas = [
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

#============================================================
# Calculate ionization balance
# ============================================================

ion_bal = ionization_balance(
    files,
    metas,
    temp_grid,
    density_grid,
    use_recombination=True,
    use_recombination_three_body=True,
    use_ionization=True,
    suppliment_with_ecip=True,
)

ion_bal.populate_ion_matrix()
ion_bal.solve_time_independent()

#============================================================
# Ionization-balance output inspection
# ============================================================

pops_ss = ion_bal.data["processed"]["pops_ss"]

print("\n--- Ionization-balance output ---")
print("Steady-state population shape:", pops_ss.shape)
print("Number of temperatures:", len(temp_grid))
print("Number of densities:", len(density_grid))

population_sum = np.sum(pops_ss, axis=0)

print("Population-sum shape:", population_sum.shape)
print("Minimum population sum:", np.min(population_sum))
print("Maximum population sum:", np.max(population_sum))

# ============================================================
# Main
# ============================================================

def main():
    check_input_files(files)
    
    print("\nDone")
    
if __name__ == "__main__":
    main()