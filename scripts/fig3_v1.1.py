"""
Chaplin Figure 3 Recreation

This script computes the ionization balance for three electron
densities. The spectral line calculations will be added next.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from colradpy import colradpy
from colradpy.ionization_balance_class import ionization_balance


# Paths

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
atomic_data_dir = PROJECT_ROOT / "atomic_data"

files = [
    str(atomic_data_dir / "mom97_ls#c0.dat"),  # C I
    str(atomic_data_dir / "mom97_ls#c1.dat"),  # C II
    str(atomic_data_dir / "mom97_ls#c2.dat"),  # C III
    str(atomic_data_dir / "mom97_ls#c3.dat"),  # C IV
    str(atomic_data_dir / "mom97_ls#c4.dat"),  # C V
    str(atomic_data_dir / "mom97_ls#c5.dat"),  # C VI
]

for file in files:
    if not Path(file).exists():
        raise FileNotFoundError(f"Missing file: {file}")


# Plasma Conditions

Te = np.linspace(1.0, 100.0, 100)

densities = [
    1.0e14,
    5.0e14,
    2.0e15,
]

metas = [
    np.array([0]),      # C I
    np.array([0, 1]),   # C II
    np.array([0, 1]),   # C III
    np.array([0]),      # C IV
    np.array([0, 1]),   # C V
    np.array([0]),      # C VI
]

def get_line_pec(adf04_file, target_wave_nm, Te, ne, metastables):
    """
    Load one ADF04 file, solve the collisional-radiative model,
    find the line closest to target_wave_nm, and return its PEC.

    Parameters
    ----------
    adf04_file : str
        Path to the ADF04 file for one ion.
    target_wave_nm : float
        Desired wavelength in nm.
    Te : ndarray
        Electron temperature grid in eV.
    ne : ndarray
        Electron density grid in cm^-3.
    metastables : ndarray
        Metastable indices for this ion.

    Returns
    -------
    actual_wave : float
        Closest wavelength found in the model.
    pec : ndarray
        PEC values versus temperature for this density.
    """

    cr = colradpy(
        adf04_file,
        metastables,
        Te,
        ne,
        use_ionization=False,
        suppliment_with_ecip=False,
        use_recombination=False,
        use_recombination_three_body=False,
        use_cx=False,
    )

    cr.solve_cr()

    waves = np.asarray(cr.data["processed"]["wave_vac"])
    pecs = np.asarray(cr.data["processed"]["pecs"])

    idx = np.argmin(np.abs(waves - target_wave_nm))
    actual_wave = waves[idx]

    print(f"Target {target_wave_nm:.1f} nm -> using {actual_wave:.4f} nm")

    # pecs shape is usually:
    # [transition, metastable, temperature, density]
    pec = pecs[idx, 0, :, 0]

    return actual_wave, pec

# Main Loop

for density in densities:

    print(f"\n==============================")
    print(f"Electron Density = {density:.2e} cm^-3")
    print(f"==============================")

    ne = np.array([density])

    ib = ionization_balance(
        files,
        metas,
        Te,
        ne,
        use_ionization=True,
        suppliment_with_ecip=True,
        use_recombination_three_body=True,
        use_recombination=True,
        use_cx=False,
        keep_charge_state_data=False,
    )

    ib.populate_ion_matrix()
    ib.solve_no_source()

    wave_977, pec_977 = get_line_pec(
        files[2],       # C III
        97.7,
        Te,
        ne,
        metas[2],
    )

    wave_155, pec_155 = get_line_pec(
        files[3],       # C IV
        155.0,
        Te,
        ne,
        metas[3],
    )

    print("C III PEC shape:", pec_977.shape)
    print("C IV PEC shape:", pec_155.shape)
    print("First few C III PECs:", pec_977[:5])
    print("First few C IV PECs:", pec_155[:5])

    print("Ionization balance solved successfully.")

print("\nDone.")