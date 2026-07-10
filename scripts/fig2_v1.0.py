"""
**STILL IN DEVELOPMENT**

Chaplin Figure 2 Recreation

This script is intended to compute a synthetic carbon spectrum using ColRadPy.

For each carbon ion:
    1. Build a ColRadPy collisional-radiative model.
    2. Extract transition wavelengths and photon emissivity coefficients (PECs).
    3. Weight the PECs by the corresponding ion fraction.
    4. Represent each transition with a Gaussian line profile.
    5. Sum all transitions onto a common wavelength grid.

The initial goal is to reproduce the modeled spectra shown in Figure 2
of Chaplin for the T=20eV and T=40eV plots.

Author: Ryan Rauenzahn
Date: July 10, 2026
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

# ============================================================
# Plasma conditions
# ============================================================

TEMPERATURE = 40.0          # eV
DENSITY = 5.0e14            # cm^-3

WAVELENGTH_MIN = 50.0       # nm
WAVELENGTH_MAX = 300.0      # nm

# Artificial Gaussian width used only to make discrete lines visible.
# This is not yet intended to reproduce the actual instrumental width.
SIGMA_NM = 0.20


# ============================================================
# Carbon ADF04 files
# ============================================================

FILES = [
    str(ATOMIC_DATA_DIR / "mom97_ls#c0.dat"),   # C I
    str(ATOMIC_DATA_DIR / "mom97_ls#c1.dat"),   # C II
    str(ATOMIC_DATA_DIR / "mom97_ls#c2.dat"),   # C III
    str(ATOMIC_DATA_DIR / "mom97_ls#c3.dat"),   # C IV
    str(ATOMIC_DATA_DIR / "mom97_ls#c4.dat"),   # C V
    str(ATOMIC_DATA_DIR / "mom97_ls#c5.dat"),   # C VI
]

ION_LABELS = [
    "C I",
    "C II",
    "C III",
    "C IV",
    "C V",
    "C VI",
]


# ============================================================
# Helper function: Gaussian line profile
# ============================================================

def gaussian(wavelength_grid, line_center, sigma):
    """
    Return a unit-peak Gaussian centered at line_center.
    """

    return np.exp(
        -0.5 * ((wavelength_grid - line_center) / sigma) ** 2
    )


# ============================================================
# Compute carbon ionization balance
# ============================================================

temperature_grid = np.array([TEMPERATURE])
density_grid = np.array([DENSITY])


print("\n--- Building carbon ionization balance ---")

ion_balance = ionization_balance(
    FILES,
    metas=[[0]] * len(FILES),
    temp_grid=temperature_grid,
    dens_grid=density_grid,
    use_ionization=True,
    use_recombination=True,
    suppliment_with_ecip=True,
)

ion_balance.populate_ion_matrix()
ion_balance.solve_no_source()


# Inspect available keys first.
print("\nIonization-balance processed keys:")
print(ion_balance.data["processed"].keys())

ion_fractions = np.squeeze(
    ion_balance.data["processed"]["ionization_balance"]
)


print("\n--- Ion fractions ---")

for label, fraction in zip(ION_LABELS, ion_fractions):
    print(f"{label:5s}: {fraction:.6e}")


# ============================================================
# Create common wavelength grid
# ============================================================

wavelength_grid = np.linspace(
    WAVELENGTH_MIN,
    WAVELENGTH_MAX,
    15000,
)

total_spectrum = np.zeros_like(wavelength_grid)


# ============================================================
# Run each ion separately and build its spectrum
# ============================================================

ion_spectra = {}


for file, ion_label, ion_fraction in zip(
    FILES,
    ION_LABELS,
    ion_fractions,
):

    print(f"\n--- Processing {ion_label} ---")

    model = colradpy(
        file,
        metas=[0],
        temp_grid=temperature_grid,
        electron_den=density_grid,
        use_ionization=True,
        use_recombination=True,
        suppliment_with_ecip=True,
    )

    model.solve_cr()

    # Vacuum wavelengths are stored in Angstroms.
    wavelengths_nm = (
        np.asarray(model.data["processed"]["wave_vac"]) / 10.0
    )

    # PEC dimensions are typically:
    #
    #     transition x metastable x density x temperature
    #
    # For one metastable, one density, and one temperature:
    pecs = np.asarray(
        model.data["processed"]["pecs"]
    )[:, 0, 0, 0]


    # Keep only transitions inside the Figure 2 wavelength range.
    mask = (
        (wavelengths_nm >= WAVELENGTH_MIN)
        & (wavelengths_nm <= WAVELENGTH_MAX)
        & np.isfinite(wavelengths_nm)
        & np.isfinite(pecs)
        & (pecs > 0.0)
    )

    wavelengths_nm = wavelengths_nm[mask]
    pecs = pecs[mask]


    # Weight the PECs by the ion fraction.
    line_strengths = ion_fraction * pecs


    # Build this ion's spectrum.
    spectrum = np.zeros_like(wavelength_grid)

    for wavelength, strength in zip(
        wavelengths_nm,
        line_strengths,
    ):

        spectrum += strength * gaussian(
            wavelength_grid,
            wavelength,
            SIGMA_NM,
        )


    ion_spectra[ion_label] = spectrum
    total_spectrum += spectrum


    # Print strongest transitions for diagnostics.
    if len(line_strengths) > 0:

        strongest = np.argsort(line_strengths)[::-1][:10]

        print("Strongest modeled transitions:")

        for index in strongest:

            print(
                f"    {wavelengths_nm[index]:8.3f} nm"
                f"    PEC = {pecs[index]:.3e}"
                f"    weighted = {line_strengths[index]:.3e}"
            )


# ============================================================
# Normalize spectrum
# ============================================================

if np.max(total_spectrum) > 0.0:
    total_spectrum_normalized = (
        total_spectrum / np.max(total_spectrum)
    )
else:
    total_spectrum_normalized = total_spectrum


# ============================================================
# Plot total model spectrum
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    wavelength_grid,
    total_spectrum_normalized,
    linewidth=1.2,
)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Normalized intensity")
ax.set_title(
    rf"Synthetic Carbon Spectrum: "
    rf"$T_e={TEMPERATURE:g}$ eV, "
    rf"$n_e={DENSITY:.1e}$ cm$^{{-3}}$"
)

ax.set_xlim(WAVELENGTH_MIN, WAVELENGTH_MAX)
ax.set_ylim(bottom=0.0)

ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ============================================================
# Plot individual ion contributions
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))

for ion_label, spectrum in ion_spectra.items():

    if np.max(spectrum) > 0.0:

        ax.plot(
            wavelength_grid,
            spectrum / np.max(total_spectrum),
            label=ion_label,
            linewidth=1.0,
        )


ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Intensity / maximum total intensity")
ax.set_title(
    rf"Carbon Ion Contributions: "
    rf"$T_e={TEMPERATURE:g}$ eV, "
    rf"$n_e={DENSITY:.1e}$ cm$^{{-3}}$"
)

ax.set_xlim(WAVELENGTH_MIN, WAVELENGTH_MAX)
ax.set_ylim(bottom=0.0)

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()
