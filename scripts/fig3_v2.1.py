"""
Chaplin Figure 3 Recreation — Version 3

Purpose:
    Compute the C III 97.7 nm / C IV 155.0 nm line ratio using ColRadPy.

Version 3 strategy:
    - Preserve the working ColRadPy calls from fig3_v1.2.py.
    - Clean up the structure.
    - Keep useful diagnostics.
    - Remove noisy temporary debugging prints.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from colradpy import colradpy
from colradpy.ionization_balance_class import ionization_balance


# ============================================================
# Paths and input files
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ATOMIC_DATA_DIR = PROJECT_ROOT / "atomic_data"

FILES = [
    str(ATOMIC_DATA_DIR / "mom97_ls#c0.dat"),  # C I
    str(ATOMIC_DATA_DIR / "mom97_ls#c1.dat"),  # C II
    str(ATOMIC_DATA_DIR / "mom97_ls#c2.dat"),  # C III
    str(ATOMIC_DATA_DIR / "mom97_ls#c3.dat"),  # C IV
    str(ATOMIC_DATA_DIR / "mom97_ls#c4.dat"),  # C V
    str(ATOMIC_DATA_DIR / "mom97_ls#c5.dat"),  # C VI
]

for file in FILES:
    if not Path(file).exists():
        raise FileNotFoundError(f"Missing file: {file}")


# ============================================================
# Plasma conditions
# ============================================================

TE = np.linspace(1.0, 100.0, 100)

DENSITIES = [
    1.0e14,
    5.0e14,
    2.0e15,
]

METAS = [
    np.array([0]),      # C I
    np.array([0, 1]),   # C II
    np.array([0, 1]),   # C III
    np.array([0]),      # C IV
    np.array([0, 1]),   # C V
    np.array([0]),      # C VI
]

CIII_WAVE_NM = 97.7
CIV_WAVE_NM = 155.0

# Temporary empirical scale for debugging.
LINE_RATIO_SCALE = 10.0


# ============================================================
# Helper functions
# ============================================================

def solve_single_ion(adf04_file, metastables, te, ne):
    """
    Solve a single-ion collisional-radiative model.
    """

    cr = colradpy(
        adf04_file,
        metastables,
        te,
        ne,
        use_ionization=False,
        suppliment_with_ecip=False,
        use_recombination=False,
        use_recombination_three_body=False,
        use_cx=False,
    )

    cr.solve_cr()
    return cr


def get_line_pec(adf04_file, target_wave_nm, te, ne, metastables):
    """
    Find the PEC closest to target_wave_nm for a single ion.
    """

    cr = solve_single_ion(adf04_file, metastables, te, ne)

    waves = np.asarray(cr.data["processed"]["wave_vac"])
    pecs = np.asarray(cr.data["processed"]["pecs"])
    pec_levels = cr.data["processed"]["pec_levels"]

    idx = np.argmin(np.abs(waves - target_wave_nm))
    actual_wave = waves[idx]

    # Shape is usually:
    #   [transition, metastable, temperature, density]
    pec = pecs[idx, 0, :, 0]

    print(f"\nLine search for {Path(adf04_file).name}")
    print(f"Target wavelength: {target_wave_nm:.4f} nm")
    print(f"Chosen wavelength: {actual_wave:.4f} nm")
    print(f"Transition index:  {idx}")
    print(f"PEC levels:        {pec_levels[idx]}")

    return actual_wave, pec


def print_lines_near(adf04_file, te, ne, metastables, center_nm, width_nm=1.0):
    """
    Print candidate lines near a target wavelength.
    Useful for verifying that the selected transition is reasonable.
    """

    cr = solve_single_ion(adf04_file, metastables, te, ne)

    waves = np.asarray(cr.data["processed"]["wave_vac"])
    pecs = np.asarray(cr.data["processed"]["pecs"])

    lo = center_nm - width_nm
    hi = center_nm + width_nm

    idxs = np.where((waves >= lo) & (waves <= hi))[0]
    idx_ref = np.argmin(np.abs(te - 60.0))

    print(f"\nCandidate lines near {center_nm:.1f} nm in {Path(adf04_file).name}:")

    for idx in idxs:
        print(
            f"idx={idx:4d}, "
            f"wave={waves[idx]:10.4f} nm, "
            f"pec60={pecs[idx, 0, idx_ref, 0]:.3e}, "
            f"levels={cr.data['processed']['pec_levels'][idx]}"
        )


def build_ionization_balance(te, ne):
    """
    Build and solve the ionization balance object.

    This preserves the known-working ColRadPy API from fig3_v1.2.py.
    """

    ib = ionization_balance(
        FILES,
        METAS,
        te,
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

    return ib


def get_charge_fractions_from_eigen(ib):
    """
    Extract steady-state charge-state fractions from ionization_balance eigenvectors.

    State ordering for current METAS:
        0      C I
        1-2    C II
        3-4    C III
        5      C IV
        6-7    C V
        8      C VI
        9      C VII / bare carbon
    """

    eigvals = ib.data["processed"]["eigen_val"]
    eigvecs = ib.data["processed"]["eigen_vec"]

    nT = eigvals.shape[0]
    charge_fractions = np.zeros((nT, 7))

    for iT in range(nT):
        k0 = np.argmin(np.abs(eigvals[iT, 0, :]))

        v = np.real(eigvecs[iT, 0, :, k0])

        if np.sum(v) < 0:
            v *= -1

        v = v / np.sum(v)

        charge_fractions[iT, 0] = v[0]          # C I
        charge_fractions[iT, 1] = v[1] + v[2]   # C II
        charge_fractions[iT, 2] = v[3] + v[4]   # C III
        charge_fractions[iT, 3] = v[5]          # C IV
        charge_fractions[iT, 4] = v[6] + v[7]   # C V
        charge_fractions[iT, 5] = v[8]          # C VI
        charge_fractions[iT, 6] = v[9]          # C VII

    return charge_fractions


def print_charge_fractions_at_temperature(te, charge_fractions, target_te=60.0):
    """
    Print charge fractions at one temperature.
    """

    idx = np.argmin(np.abs(te - target_te))
    labels = ["C I", "C II", "C III", "C IV", "C V", "C VI", "C VII"]

    print(f"\nCharge-state fractions at T_e = {te[idx]:.2f} eV")

    for i, label in enumerate(labels):
        print(f"{label:5s}: {charge_fractions[idx, i]:.6e}")

    print(f"Sum:   {np.sum(charge_fractions[idx]):.6e}")


def compute_ratio_for_density(density):
    """
    Compute the C III 97.7 nm / C IV 155.0 nm ratio for one density.
    """

    print("\n" + "=" * 70)
    print(f"Electron density = {density:.3e} cm^-3")
    print("=" * 70)

    ne = np.array([density])

    # Optional but useful diagnostic:
    print_lines_near(FILES[2], TE, ne, METAS[2], CIII_WAVE_NM, width_nm=1.0)
    print_lines_near(FILES[3], TE, ne, METAS[3], CIV_WAVE_NM, width_nm=1.0)

    ib = build_ionization_balance(TE, ne)
    charge_fractions = get_charge_fractions_from_eigen(ib)

    print_charge_fractions_at_temperature(TE, charge_fractions, target_te=60.0)

    f_ciii = charge_fractions[:, 2]
    f_civ = charge_fractions[:, 3]

    wave_977, pec_977 = get_line_pec(
        FILES[2],
        CIII_WAVE_NM,
        TE,
        ne,
        METAS[2],
    )

    wave_155, pec_155 = get_line_pec(
        FILES[3],
        CIV_WAVE_NM,
        TE,
        ne,
        METAS[3],
    )

    frac_ratio = f_ciii / f_civ
    pec_ratio = pec_977 / pec_155

    line_ratio = LINE_RATIO_SCALE * frac_ratio * pec_ratio

    return {
        "density": density,
        "wave_977": wave_977,
        "wave_155": wave_155,
        "f_ciii": f_ciii,
        "f_civ": f_civ,
        "frac_ratio": frac_ratio,
        "pec_977": pec_977,
        "pec_155": pec_155,
        "pec_ratio": pec_ratio,
        "line_ratio": line_ratio,
    }


def plot_line_ratio(results):
    """
    Plot final line ratio.
    """

    plt.figure(figsize=(6, 4))

    for density, result in results.items():
        plt.semilogy(
            TE,
            result["line_ratio"],
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

    out_file = PROJECT_ROOT / "fig3_v3_line_ratio.png"
    plt.savefig(out_file, dpi=150)
    plt.show()

    print(f"\nSaved: {out_file}")


def plot_component_ratio(results, key, ylabel, title):
    """
    Generic helper for plotting diagnostic ratios.
    """

    plt.figure(figsize=(6, 4))

    for density, result in results.items():
        plt.semilogy(
            TE,
            result[key],
            marker="o",
            markevery=3,
            linewidth=1.5,
            markersize=4,
            label=fr"$n_e = {density:.1e}$",
        )

    plt.xlabel(r"$T_e$ (eV)")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xlim(0, 100)

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    results = {}

    for density in DENSITIES:
        results[density] = compute_ratio_for_density(density)

    plot_line_ratio(results)

    plot_component_ratio(
        results,
        key="frac_ratio",
        ylabel=r"$f_{\mathrm{CIII}} / f_{\mathrm{CIV}}$",
        title="Charge-State Fraction Ratio",
    )

    plot_component_ratio(
        results,
        key="pec_ratio",
        ylabel=r"$PEC_{977} / PEC_{155}$",
        title="PEC Ratio",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()