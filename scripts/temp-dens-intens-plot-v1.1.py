"""

** IN DEVELOPMENT **

3D Plotting of Line Intensity

Goal:

Outline:

Notes: 

Author: Ryan Rauenzahn
Date: July 22, 2026
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
    str(ATOMIC_DATA_DIR / "mom97_ls#c0.dat"),  # C I
    str(ATOMIC_DATA_DIR / "mom97_ls#c1.dat"),  # C II
    str(ATOMIC_DATA_DIR / "mom97_ls#c2.dat"),  # C III
    str(ATOMIC_DATA_DIR / "mom97_ls#c3.dat"),  # C IV
    str(ATOMIC_DATA_DIR / "mom97_ls#c4.dat"),  # C V
    str(ATOMIC_DATA_DIR / "mom97_ls#c5.dat"),  # C VI
]

# ============================================================
# Plasma conditions
# ============================================================

TE = np.linspace(1.0, 50, 50)

DENSITIES = np.logspace(12,15,30)

METAS = [
    np.array([0]),      # C I
    np.array([0, 1]),   # C II
    np.array([0, 1]),   # C III
    np.array([0]),      # C IV
    np.array([0, 1]),   # C V
    np.array([0]),      # C VI
]

CIII_TARGET_NM = 97.7
CIV_TARGET_NM = 155.0

# Temporary diagnostic scale - NOT PHYSICAL - set to 1.0 for normal scale
LINE_RATIO_SCALE = 1.0

# ============================================================
# File checks
# ============================================================

def check_input_files(files):
    """Verify that all required ADF04 files exist."""

    for file in files:
        if not Path(file).exists():
            raise FileNotFoundError(f"Missing file: {file}")


# ============================================================
# Single-ion CR model / PEC extraction
# ============================================================

def solve_single_ion(adf04_file, metastables, te, ne):
    """
    Solve the collisional-radiative model for one ionization stage.

    This calculation gives access to PECs for individual spectral lines.
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
    Find the PEC closest to the desired wavelength.

    Returns
    -------
    actual_wave : float
        Wavelength actually selected from the ADF04/ColRadPy output.
    pec : ndarray
        PEC as a function of electron temperature.
    """

    cr = solve_single_ion(adf04_file, metastables, te, ne)

    print("\nProcessed keys:")
    print(sorted(cr.data["processed"].keys()))
    
    waves = np.asarray(cr.data["processed"]["wave_vac"])
    pecs = np.asarray(cr.data["processed"]["pecs"])
    pec_levels = cr.data["processed"]["pec_levels"]

    idx = np.argmin(np.abs(waves - target_wave_nm))
    actual_wave = waves[idx]
    
    idx60 = np.argmin(np.abs(te - 60.0))

    print("\n  PEC-related quantities at 60 eV:")

    for key in ["pecs", "plt", "pls", "prb"]:
        arr = cr.data["processed"].get(key)

        if arr is None:
            print(f"    {key}: not found")
            continue

        arr = np.asarray(arr)
        print(f"    {key} shape: {arr.shape}")

        try:
            if arr.ndim == 4:
                vals = arr[idx, :, idx60, 0]
                print(f"    {key} by metastable: {vals}")
                print(f"    {key} summed:        {np.sum(vals):.6e}")

            elif arr.ndim == 3:
                vals = arr[idx, :, idx60]
                print(f"    {key} by metastable: {vals}")
                print(f"    {key} summed:        {np.sum(vals):.6e}")

            elif arr.ndim == 2:
                print(f"    {key}: {arr[idx, idx60]:.6e}")

            else:
                print(f"    {key}: cannot parse ndim={arr.ndim}")

        except Exception as err:
            print(f"    {key}: could not index cleanly -> {err}")

    pec_by_meta = pecs[idx, :, :, 0]
    pec = np.sum(pec_by_meta, axis=0)
    
    print(" PEC by metastable at 60 eV:")

    idx60 = np.argmin(np.abs(te - 60.0))
    for m in range(pec_by_meta.shape[0]):
        print(f"    metastable {m}: {pec_by_meta[m, idx60]:.6e}")
    print(f"    summed:       {pec[idx60]:.6e}")

    print(f"\nSelected line from {Path(adf04_file).name}")
    print(f"  Target wavelength: {target_wave_nm:.4f} nm")
    print(f"  Actual wavelength: {actual_wave:.4f} nm")
    print(f"  Transition index:  {idx}")
    print(f"  PEC levels:        {pec_levels[idx]}")

    return actual_wave, pec


def print_lines_near(adf04_file, te, ne, metastables, center_nm, width_nm=10.0):
    """
    Print candidate spectral lines near a target wavelength.
    """

    cr = solve_single_ion(adf04_file, metastables, te, ne)

    waves = np.asarray(cr.data["processed"]["wave_vac"])
    pecs = np.asarray(cr.data["processed"]["pecs"])

    lower = center_nm - width_nm
    upper = center_nm + width_nm

    idxs = np.where((waves >= lower) & (waves <= upper))[0]
    idx60 = np.argmin(np.abs(te - 60.0))

    print(f"\nCandidate lines near {center_nm:.1f} nm in {Path(adf04_file).name}:")

    if len(idxs) == 0:
        print("  No lines found in this wavelength window.")
        return

    for idx in idxs:
        print(
            f"  idx={idx:4d}, "
            f"wave={waves[idx]:10.4f} nm, "
            f"PEC(60 eV)={pecs[idx, 0, idx60, 0]:.3e}, "
            f"levels={cr.data['processed']['pec_levels'][idx]}"
        )


# ============================================================
# Ionization balance
# ============================================================

def build_ionization_balance(te, ne):
    """
    Build and solve the carbon ionization balance.

    ColRadPy internally performs CR calculations to generate effective
    ionization/recombination coefficients, then solves the charge-state balance.
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
    Print charge-state fractions at one representative temperature.
    """

    idx = np.argmin(np.abs(te - target_te))

    labels = [
        "C I",
        "C II",
        "C III",
        "C IV",
        "C V",
        "C VI",
        "C VII",
    ]

    print(f"\nCharge-state fractions at T_e = {te[idx]:.2f} eV:")

    for i, label in enumerate(labels):
        print(f"  {label:5s}: {charge_fractions[idx, i]:.6e}")

    print(f"  Sum:   {np.sum(charge_fractions[idx]):.6e}")


# ============================================================
# Ratio calculation
# ============================================================

def compute_result_for_density(density):
    """
    Compute all useful quantities for one electron density.
    """

    print("\n" + "=" * 70)
    print(f"Electron density = {density:.3e} cm^-3")
    print("=" * 70)

    ne = np.array([density])

    # Diagnostic line searches
    print_lines_near(FILES[2], TE, ne, METAS[2], CIII_TARGET_NM, width_nm=10.0)
    print_lines_near(FILES[3], TE, ne, METAS[3], CIV_TARGET_NM, width_nm=10.0)

    # Ionization balance
    ib = build_ionization_balance(TE, ne)
    charge_fractions = get_charge_fractions_from_eigen(ib)

    print_charge_fractions_at_temperature(
        TE,
        charge_fractions,
        target_te=60.0,
    )

    f_ciii = charge_fractions[:, 2]
    f_civ = charge_fractions[:, 3]

    # Independent PEC calculations
    wave_977, pec_977 = get_line_pec(
        FILES[2],
        CIII_TARGET_NM,
        TE,
        ne,
        METAS[2],
    )

    wave_155, pec_155 = get_line_pec(
        FILES[3],
        CIV_TARGET_NM,
        TE,
        ne,
        METAS[3],
    )

    frac_ratio = f_ciii / f_civ
    pec_ratio = pec_977 / pec_155

    line_ratio = LINE_RATIO_SCALE * frac_ratio * pec_ratio

    return {
        "density": density,
        "charge_fractions": charge_fractions,
        "f_ciii": f_ciii,
        "f_civ": f_civ,
        "wave_977": wave_977,
        "wave_155": wave_155,
        "pec_977": pec_977,
        "pec_155": pec_155,
        "frac_ratio": frac_ratio,
        "pec_ratio": pec_ratio,
        "line_ratio": line_ratio,
    }


# ============================================================
# Plotting
# ============================================================

def plot_line_ratio(results):
    """Plot the final C III / C IV line ratio."""

    plt.figure(figsize=(6, 4))

    plot_indices = [0, 7, 14, 21, 29]
    
    target_densities = [3e13, 8e14, 5e15]

    for target in target_densities:

        density = min(results.keys(), key=lambda d: abs(d - target))

        result = results[density]

        plt.semilogy(
            TE,
            result["line_ratio"],
            label=fr"$n_e = {density:.2e}$",
        )
    
    """ for density, result in results.items():
        plt.semilogy(
            TE,
            result["line_ratio"],
            marker="o",
            markevery=3,
            linewidth=1.5,
            markersize=5,
            label=fr"$n_e = {density:.1e}\ \mathrm{{cm^{{-3}}}}$",
        ) """

    plt.xlabel(r"$T_e$ (eV)")
    plt.ylabel(r"Line Ratio ($I_{97.7}$ / $I_{155}$)")
    plt.title("Carbon Line Ratio vs. Electron Temperature")

    plt.xlim(0, 50)
    plt.ylim(1e-5, 1e1)

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.tight_layout()

    out_file = PROJECT_ROOT / "3d_line_ratio_line_ratio.png"
    plt.savefig(out_file, dpi=150)
    plt.show()

    print(f"\nSaved: {out_file}")


def plot_component_ratio(results, key, ylabel, title):
    """Plot a diagnostic component ratio."""

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

def plot_heatmap(results):
    """Plot the C III 97.7 nm / C IV 155 nm line-ratio heat map."""

    densities = np.array(sorted(results.keys()))

    line_ratio_surface = np.column_stack(
        [results[d]["line_ratio"] for d in densities]
    )

    # Keep only the temperature range used in the useful portion of Figure 3
    temperature_mask = (TE >= 5.0) & (TE <= 50.0)

    temperatures = TE[temperature_mask]
    ratio_for_plot = line_ratio_surface[temperature_mask, :]
    
    # Normalize every temperature to the lowest density
    normalized_ratio = ratio_for_plot / ratio_for_plot[:, 0][:, np.newaxis]

    log_ratio = np.log10(normalized_ratio)
    
    for target_T in [10, 20, 40]:

        i = np.argmin(np.abs(temperatures - target_T))

        print(f"\nTemperature = {temperatures[i]:.1f} eV")

        for j, d in enumerate(densities):
            print(f"{d:.2e}  {ratio_for_plot[i, j]:.4e}")


    fig, ax = plt.subplots(figsize=(8, 5))

    print("temperatures:", temperatures.shape)
    print("densities:", densities.shape)
    print("log_ratio:", log_ratio.shape)
    print("log_ratio.T:", log_ratio.T.shape)

    T_mesh, N_mesh = np.meshgrid(temperatures, densities)

    mesh = ax.pcolormesh(
        T_mesh,
        N_mesh,
        log_ratio.T,
        shading="nearest",
        cmap="viridis",
    )

    ax.set_yscale("log")

    ax.set_ylabel(r"Electron Density ($n_e$, cm$^{-3}$)")
    ax.set_xlabel(r"Electron Temperature ($T_e$, eV)")
    ax.set_title(r"$\log_{10}\left[\frac{(I_{97.7}/I_{155})}{(I_{97.7}/I_{155})_{n_{e,\min}}}\right]$")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Relative to lowest density")

    fig.tight_layout()
    plt.show()

def make_plots(results):
    """Create all plots."""

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
    
    plot_heatmap(results)
    

def plot_line_ratio_surface(results):
    densities = list(results.keys())

    line_ratio_surface = np.column_stack(
        [results[d]["line_ratio"] for d in densities]
    )

# ============================================================
# Main
# ============================================================

def main():
    check_input_files(FILES)

    results = {}

    for density in DENSITIES:
        results[density] = compute_result_for_density(density)

    make_plots(results)


    print("\nDone.")


if __name__ == "__main__":
    main()