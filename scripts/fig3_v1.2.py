"""
Chaplin Figure 3 Recreation

This script is intended to compute the ionization balance for three electron
densities. Then it computes the line ratio 97.7/155 nm and plots 
the ratio as a function of temperature (in eV).

There are some debugging prints along with sanity check plots that 

** This is still in development **


Author: Ryan Rauenzahn
Date: July 2, 2026
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

frac_ratios = {}
pec_ratios = {}
line_ratios = {}

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
    
    print("PEC levels:")
    print(cr.data["processed"]["pec_levels"][idx])
    
    print("PEC for metastable 0:")
    print(pecs[idx,0,55,0])

    if pecs.shape[1] > 1:
        print("PEC for metastable 1:")
        print(pecs[idx,1,55,0])

    print(f"Target {target_wave_nm:.1f} nm -> using {actual_wave:.4f} nm")

    # pecs shape is usually:
    # [transition, metastable, temperature, density]
    pec = pecs[idx, 0, :, 0]
    
    for key in ["pecs", "plt", "pls", "prb", "pops", "pop_lvl"]:
        arr = cr.data["processed"].get(key)
        if arr is not None:
            print(key, np.shape(arr))

    return actual_wave, pec

def get_charge_fractions_from_eigen(ib):
    """
    Extract steady-state charge-state fractions from ionization_balance eigenvectors.

    State ordering for your metas:
        0      C I
        1-2    C II
        3-4    C III
        5      C IV
        6-7    C V
        8      C VI
        9      C VII / bare carbon
    """

    eigvals = ib.data["processed"]["eigen_val"]      # shape: (Te, ne, states)
    eigvecs = ib.data["processed"]["eigen_vec"]      # shape: (Te, ne, states, states)

    nT = eigvals.shape[0]
    charge_fractions = np.zeros((nT, 7))  # C I through C VII

    for iT in range(nT):
        # Find eigenvalue closest to zero
        k0 = np.argmin(np.abs(eigvals[iT, 0, :]))

        # Corresponding eigenvector
        v = np.real(eigvecs[iT, 0, :, k0])

        # Make sign positive
        if np.sum(v) < 0:
            v *= -1

        # Normalize
        v = v / np.sum(v)

        charge_fractions[iT, 0] = v[0]          # C I
        charge_fractions[iT, 1] = v[1] + v[2]   # C II
        charge_fractions[iT, 2] = v[3] + v[4]   # C III
        charge_fractions[iT, 3] = v[5]          # C IV
        charge_fractions[iT, 4] = v[6] + v[7]   # C V
        charge_fractions[iT, 5] = v[8]          # C VI
        charge_fractions[iT, 6] = v[9]          # C VII

    return charge_fractions


def print_lines_near(adf04_file, Te, ne, metastables, center_nm, width_nm=1.0):
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

    lo = center_nm - width_nm
    hi = center_nm + width_nm

    idxs = np.where((waves >= lo) & (waves <= hi))[0]

    print(f"\nLines near {center_nm} nm in {Path(adf04_file).name}:")
    for idx in idxs:
        print(
            f"idx={idx:4d}, "
            f"wave={waves[idx]:10.4f} nm, "
            f"pec60={pecs[idx, 0, np.argmin(np.abs(Te - 60.0)), 0]:.3e}, "
            f"levels={cr.data['processed']['pec_levels'][idx]}"
        )
        
        
        
# Main Loop

# Store the line ratio for each density
ratios = {}

for density in densities:

    print(f"\n==============================")
    print(f"Electron Density = {density:.2e} cm^-3")
    print(f"==============================")

    ne = np.array([density])

    print_lines_near(files[3], Te, ne, metas[3], 155.0, width_nm=1.0)
    

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
    
    print("Default initial abundance:")
    print(ib.data["user"]["init_abund"])

    print("\nDefault solution times:")
    print(ib.data["user"]["soln_times"])
    
    ib.solve_no_source()

    charge_fractions = get_charge_fractions_from_eigen(ib)

    f_ciii = charge_fractions[:, 2]
    f_civ = charge_fractions[:, 3]

    idx60 = np.argmin(np.abs(Te - 60))

    print("\nCharge state fractions at 60 eV")

    labels = ["CI","CII","CIII","CIV","CV","CVI","CVII"]

    for i, lab in enumerate(labels):
        print(f"{lab:4s}: {charge_fractions[idx60, i]:.6e}")

    print("Sum:", np.sum(charge_fractions[idx60]))

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

    #DEB - used for visual comparison with Chaplin FIG 3
    line_ratio_scale = 10.0
    ratio = line_ratio_scale * (f_ciii * pec_977) / (f_civ * pec_155)
    ratios[density] = ratio
    
    frac_ratio = f_ciii / f_civ
    pec_ratio = pec_977 / pec_155
    line_ratio = line_ratio_scale * frac_ratio * pec_ratio

    frac_ratios[density] = frac_ratio
    pec_ratios[density] = pec_ratio
    line_ratios[density] = line_ratio

# Plotting

plt.figure(figsize=(6, 4))

for density in densities:
    plt.semilogy(
        Te,
        ratios[density],
        marker="o",
        markevery=3,
        linewidth=1.5,
        markersize=5,
        label=fr"$n_e = {density:.1e}\ \mathrm{{cm^{{-3}}}}$",
    )

plt.xlabel(r"$T_e$ (eV)")
plt.ylabel(r"C III 977 $\AA$ / C IV 1550 $\AA$")
plt.title("Carbon Line Ratio")

plt.xlim(0, 100)
plt.ylim(1e-4, 1e1)

plt.grid(True, which="both", alpha=0.3)
plt.legend()

plt.tight_layout()


out_file = PROJECT_ROOT / "fig3_v1_1_line_ratio.png"
plt.savefig(out_file, dpi=150)
plt.show()

print(f"\nSaved: {out_file}")

plt.figure(figsize=(6, 4))

for density in densities:
    plt.semilogy(
        Te,
        frac_ratios[density],
        marker="o",
        markevery=3,
        linewidth=1.5,
        markersize=4,
        label=fr"$n_e = {density:.1e}$",
    )

plt.xlabel(r"$T_e$ (eV)")
plt.ylabel(r"$f_{\mathrm{CIII}} / f_{\mathrm{CIV}}$")
plt.xlim(0, 100)
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))

for density in densities:
    plt.semilogy(
        Te,
        pec_ratios[density],
        marker="o",
        markevery=3,
        linewidth=1.5,
        markersize=4,
        label=fr"$n_e = {density:.1e}$",
    )

plt.xlabel(r"$T_e$ (eV)")
plt.ylabel(r"$PEC_{977} / PEC_{155}$")
plt.xlim(0, 100)
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("\nDone.")