# Purpose:Loads atomic data for CIII and CIV, runs colradpy calculation for each species at the hard-coded electron temperature and density, it finds the spectral lines nearest to 97.7 and 155 nm, extracts their PECs, and calculates the ratio.
# Author: Ryan Rauenzahn
# Origin Date: 03/12/2026
# Last Edit Date: 5/29/2026
# Notes: I think the actual intensity is the PEC ratio multiplied with the number density ratio, so expand on this program. The inital run at 60eV gives 8.4x10-1 and the Chaplin paper shows about 10-3 for the density of 5x10+14. 


from pathlib import Path
from colradpy import colradpy
import numpy as np

# Loads a species and calls in colradpy to run. 
def run_species(file_path, metastables, Te, ne):
    cr = colradpy(
        str(file_path),
        np.array(metastables),
        np.array([Te]),
        np.array([ne])
    )

    cr.solve_cr()
    # extracts array of transition wavelengths
    wave = cr.data["processed"]["wave_vac"]
    # extracts PEC for each transition
    pecs = cr.data["processed"]["pecs"]
    # ***reducing the PEC array to one PEC value per line
    pec_total = np.sum(pecs[:, :, 0, 0], axis=1)

    return wave, pec_total

# Searches for a wavelength near a target value
def find_line(wave, pec, target, tol=1.0):
    idx = np.where(np.abs(wave - target) < tol)[0]

    if len(idx) == 0:
        return None

    best = idx[np.argmax(pec[idx])]
    return wave[best], pec[best]

# Plasma parameters (the temperature needs to be done over a range next)
Te = 60.0
ne = 5e14

# Retrieve saved files
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
base = PROJECT_ROOT / "atomic_data"
# CIII file
c2_file = base / "mom97_ls#c2.dat"
# CIV file
c3_file = base / "mom97_ls#c3.dat"

wave_c2, pec_c2 = run_species(c2_file, [0], Te, ne)
wave_c3, pec_c3 = run_species(c3_file, [0], Te, ne)

sort_idx_c3 = np.argsort(pec_c3)[::-1]

c3_line = find_line(wave_c2, pec_c2, 97.7, tol=1.0)
c4_line = find_line(wave_c3, pec_c3, 155.0, tol=1.0)

# Outputs 
print("\n--- Results ---")

if c3_line:
    print(f"C III (~97.7 nm): wavelength={c3_line[0]:.6f}, PEC={c3_line[1]:.6e}")
else:
    print("C III line not found")

if c4_line:
    print(f"C IV (~155 nm): wavelength={c4_line[0]:.6f}, PEC={c4_line[1]:.6e}")
else:
    print("C IV line not found")

# ratio computation
if c3_line and c4_line:
    ratio = c3_line[1] / c4_line[1]
    print(f"Line ratio (CIII / CIV) = {ratio:.6e}")
