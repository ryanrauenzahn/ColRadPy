"""
** IN DEVELOPMENT **

Goal: 
    To generate a 3-dimensional plot with axes temperature, density, and intensity ratio.

Outline:
    1. Import required libraries and define file paths.
    2. Create the temperature and density parameter grids. Temp. range (0-15 eV), density range (1e12-1e15, stepwise).
    3. Load the carbon atomic data into ColRadPy.
    4. Compute the ionization balance across the parameter space.
    5. Calculate the selected spectral line intensity for every temperature-density combination.
    6. Store the calculated intensities in a two-dimensional array.
    7. Generate a three-dimensional surface plot of the results.
    8. Generate a two-dimensional plot to view side profile of density variation.
    9. Generate a zoomed-in "heat map" to show density variation (since it's small).
    10. Generate an interactive HTML page of the three-dimensional plot. 
    
    
Sections:
    1. Paths: defines the relative paths to the ADAS data files in this same root directory (located in the atomic_data folder). Files are in the ADF04 form and are named like "mom97_ls#c0.dat". 
    2. Variables: define temp_grid to be from 0.1 eV to 15 eV and dens_grid to be 1e12 through 1e15, evenly spaced. So there will be (150)*(28)=4200 unique temperature-density combinations.
    3. File Check: small function that checks to see if every required ADF04 file exists.
    4. Retrive ionization balance-output: ColRadPy constructs the ionization balance matrix by passing in all the variables supplied to it and then it solves the no source ionization balance problem. No source means there's not a continuous injection of fresh carbon into the system. Stored in the ion_bal object are the fractions (like f_CIII(T_e, n_e)).
    5. Build the C III CR model: we need to know how bright the lines from one C III ion are. To get this information, we solve the CR model, which tells us how many ions are sitting in each excitied level under the specified plasma conditions. It's from this information that we can find the emitted light.
    6. Locate the C III 97.7 nm transition: Need to know which transition in the wave_air output from the CR model corresponds to the 97.7 nm transition.
    7. Retrieve ionization balance output: Extracts the steady-state ion fractions from the solved ionization balance. These fractions represent the relative abundance of each metastable charge state and provide the population weights needed when calculating spectral emissivities
    8. Build the C III CR model: Constructs and solves the collisional-radiative model for C III. This determines the photon emissivity coefficients (PECs) describing how efficiently a single C III ion emits each spectral line through electron-impact excitation and recombination.
    9. Locate the C III 97.7 nm transition: Searches the list of wavelengths produced by the CR model and identifies the transition closest to 97.7 nm. The resulting transition index is used to extract the correct photon emissivity coefficients from the full PEC array.
    10. Extract the C III 97.7 nm PECs: Extracts the photon emissivity coefficients associated with the selected transition. These coefficients contain separate contributions from excitation of each metastable state and from recombination into C III.
    11. Extract populations that drive the C III line: Retrieves the ionization-balance populations corresponding to the metastable states that contribute to the selected C III transition. These populations provide the weighting factors applied to each PEC contribution.
    12. Construct the C III emissivity coefficient: Multiplies each photon emissivity coefficient by the appropriate metastable population and sums the individual excitation and recombination contributions to obtain the total C III 97.7 nm emissivity coefficient.
    13. Build the C IV CR model: Constructs and solves the collisional-radiative model for C IV. This determines the photon emissivity coefficients (PECs) describing how efficiently a single C IV ion emits each spectral line through electron-impact excitation and recombination.
    14. Locate the C IV 155 nm transition: Searches the list of wavelengths produced by the CR model and identifies the transition closest to 155 nm. The resulting transition index is used to extract the correct photon emissivity coefficients from the full PEC array.
    15. 
    16. 
    17. 
    18. Compute the C III / C IV line ratio: Computes the C III/C IV intensity ratio at every temperature-density point while avoiding division by zero in regions where the C IV emissivity is negligible.
    19. Create a ratio array for visualization: Removes physically meaningless ratios that arise when both spectral lines have extremely small emissivities. The masking is applied only for visualization and does not alter the underlying calculated line ratio.
    20. Create mesh grids for 3D plotting: Converts the one-dimensional temperature and density arrays into two-dimensional coordinate grids required for surface plotting.
    21. Logarithmic line ratio for visualization: Converts the line ratio to base-10 logarithmic scale to improve visualization over the wide dynamic range.
    22. 3D surface plot: Generates a three-dimensional surface showing the dependence of the C III/C IV line ratio on electron temperature and density.
    23. 
    24.
    25. Generate an interactive HTML page for the 3D plot using Plotly. 
    
    
Author: Ryan Rauenzahn
Date: 07/14/2026

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
# Retrieve ionization-balance output 
# ============================================================

pops_ss = ion_bal.data["processed"]["pops_ss"]

# ============================================================
# Build the C III CR model
# ============================================================

ciii = colradpy(
    files[2],
    metas[2],
    temp_grid,
    density_grid,
    use_recombination=True,
    use_recombination_three_body=True,
    use_ionization=True,
    suppliment_with_ecip=True,
)

ciii.solve_cr()

# ============================================================
# Locate the C III 97.7 nm transition
# ============================================================

target_wave = 97.7

wave_air = ciii.data["processed"]["wave_air"]

ciii_line_index = np.argmin(np.abs(wave_air - target_wave))

print("\n--- C III target transition ---")
print("Requested wavelength:", target_wave, "nm")
print("Closest wavelength:", wave_air[ciii_line_index], "nm")
print("Transition index:", ciii_line_index)

# ============================================================
# Extract the C III 97.7 nm PECs
# ============================================================
pecs = ciii.data["processed"]["pecs"]

ciii_line_pecs = pecs[ciii_line_index, :, :, :]

# ============================================================
# Extract populations that drive the C III 97.7 nm line
# ============================================================

ciii_meta0_fraction = pops_ss[3, :, :]
ciii_meta1_fraction = pops_ss[4, :, :]
civ_meta0_fraction = pops_ss[5, :, :]

# ============================================================
# Construct the C III 97.7 nm emissivity coefficient
# ============================================================

ciii_97_excitation_meta0 = (
    ciii_meta0_fraction * ciii_line_pecs[0, :, :]
)

ciii_97_excitation_meta1 = (
    ciii_meta1_fraction * ciii_line_pecs[1, :, :]
)

ciii_97_recombination = (
    civ_meta0_fraction * ciii_line_pecs[2, :, :]
)

ciii_97_emissivity_coeff = (
    ciii_97_excitation_meta0
    + ciii_97_excitation_meta1
    + ciii_97_recombination
)

# ============================================================
# Build the C IV CR model
# ============================================================

civ = colradpy(
    files[3],
    metas[3],
    temp_grid,
    density_grid,
    use_recombination=True,
    use_recombination_three_body=True,
    use_ionization=True,
    suppliment_with_ecip=True,
)

civ.solve_cr()

# ============================================================
# Locate the C IV 155 nm transition
# ============================================================

target_wave_civ = 155.0

civ_wave_air = civ.data["processed"]["wave_air"]

civ_line_index = np.argmin(
    np.abs(civ_wave_air - target_wave_civ)
)

# ============================================================
# Inspect the C IV 154.86 nm PEC contributions
# ============================================================

civ_pecs = civ.data["processed"]["pecs"]

civ_line_pecs = civ_pecs[civ_line_index, :, :, :]

# ============================================================
# Extract populations that drive the C IV 154.86 nm line
# ============================================================

civ_meta0_fraction = pops_ss[5, :, :]
cv_meta0_fraction = pops_ss[6, :, :]
cv_meta1_fraction = pops_ss[7, :, :]

# ============================================================
# Construct the C IV 154.86 nm emissivity coefficient
# ============================================================

civ_155_excitation = (
    civ_meta0_fraction * civ_line_pecs[0, :, :]
)

civ_155_recombination_meta0 = (
    cv_meta0_fraction * civ_line_pecs[1, :, :]
)

civ_155_recombination_meta1 = (
    cv_meta1_fraction * civ_line_pecs[2, :, :]
)

civ_155_emissivity_coeff = (
    civ_155_excitation
    + civ_155_recombination_meta0
    + civ_155_recombination_meta1
)

# ============================================================
# Compute the C III / C IV line ratio
# ============================================================

line_ratio = np.divide(
    ciii_97_emissivity_coeff,
    civ_155_emissivity_coeff,
    out=np.full_like(ciii_97_emissivity_coeff, np.nan),
    where=civ_155_emissivity_coeff > 0,
)

# ============================================================
# Create a ratio array for visualization
# ============================================================

relative_cutoff = 1e-60

ciii_plot_threshold = (
    relative_cutoff * np.nanmax(ciii_97_emissivity_coeff)
)

civ_plot_threshold = (
    relative_cutoff * np.nanmax(civ_155_emissivity_coeff)
)

plot_mask = (
    (ciii_97_emissivity_coeff > ciii_plot_threshold)
    & (civ_155_emissivity_coeff > civ_plot_threshold)
)

line_ratio_plot = np.where(
    plot_mask,
    line_ratio,
    np.nan,
)

# ============================================================
# Create mesh grids for 3D plotting
# ============================================================

density_mesh, temp_mesh = np.meshgrid(
    density_grid,
    temp_grid,
)

# ============================================================
# Logarithmic line ratio for visualization
# ============================================================

log_line_ratio = np.log10(line_ratio_plot)

# ============================================================
# 3D surface plot
# ============================================================

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surface = ax.plot_surface(
    temp_mesh,
    np.log10(density_mesh),
    log_line_ratio,
    cmap="viridis",
    edgecolor="white",
    linewidth=0.15,
    antialiased=True,
)

ax.contour(
    temp_mesh,
    np.log10(density_mesh),
    log_line_ratio,
    levels=12,
    colors="black",
    linewidths=0.8,
    zdir="z",
    offset=np.nanmin(log_line_ratio),
)

ax.set_xlabel("Electron temperature (eV)")
ax.set_ylabel(r"log$_{10}$(Electron density [cm$^{-3}$])")
ax.set_zlabel(r"log$_{10}$(I$_{97.7}$/I$_{155}$)")

fig.colorbar(
    surface,
    shrink=0.7,
    aspect=20,
    label="Line ratio",
)

plt.title("C III 97.7 nm / C IV 155 nm Line Ratio")

ax.view_init(
    elev=30,
    azim=-60
)

plt.tight_layout()

plt.savefig(
    "CIII_CIV_LineRatio_3D.png",
    dpi=300,
    bbox_inches="tight",
)

print("Figure saved as CIII_CIV_LineRatio_3D.png")

plt.show()

# ============================================================
# Plot temperature slices at selected densities
# ============================================================

selected_densities = [
    1e12,
    1e13,
    1e14,
    5e14,
    1e15,
]

plt.figure(figsize=(8, 6))

for target_density in selected_densities:

    density_index = np.argmin(
        np.abs(density_grid - target_density)
    )

    actual_density = density_grid[density_index]

    plt.semilogy(
        temp_grid,
        line_ratio_plot[:, density_index],
        label=f"{actual_density:.2e} cm$^{{-3}}$",
    )

plt.xlabel("Electron temperature (eV)")
plt.ylabel(r"$I_{97.7}/I_{155}$")
plt.title("C III 97.7 nm / C IV 155 nm Line Ratio")
plt.grid(True, which="both", alpha=0.3)
plt.legend()

plt.show()

# ============================================================
# Zoomed heat map showing density sensitivity
# ============================================================

plt.figure(figsize=(7, 5))

# Temperature range to zoom in on
temp_mask = (temp_grid >= 1.0) & (temp_grid <= 8.0)

# Filled contour plot
contour = plt.contourf(
    temp_grid[temp_mask],
    density_grid,
    log_line_ratio[temp_mask, :].T,
    levels=60,
    cmap="viridis",
)

# Density axis should be logarithmic
plt.yscale("log")

# Labels
plt.xlabel("Electron Temperature (eV)")
plt.ylabel(r"Electron Density (cm$^{-3}$)")
plt.title(r"Zoomed C III 97.7 nm / C IV 155 nm Line Ratio")

# Colorbar
cbar = plt.colorbar(contour)
cbar.set_label(r"$\log_{10}(I_{97.7}/I_{155})$")

plt.tight_layout()

plt.savefig(
    "CIII_CIV_ZoomedHeatmap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# ============================================================
# Interactive Plotly 3D surface
# ============================================================

fig = go.Figure(
    data=[
        go.Surface(
            x=temp_mesh,
            y=np.log10(density_mesh),
            z=log_line_ratio,
            colorscale="Plasma",
            colorbar=dict(
                title="log₁₀(Line Ratio)"
            ),
        )
    ]
)

fig.update_layout(

    title="C III 97.7 nm / C IV 155 nm Line Ratio",

    scene=dict(

        xaxis_title="Electron Temperature (eV)",

        yaxis_title="log₁₀(Electron Density [cm⁻³])",

        zaxis_title="log₁₀(I97.7/I155)",

        camera=dict(
            eye=dict(
                x=1.6,
                y=-1.8,
                z=1.1,
            )
        ),
    ),

    width=1000,
    height=700,
)

fig.write_html(
    "CIII_CIV_LineRatio_Interactive.html"
)

fig.show()

# ============================================================
# Main
# ============================================================

def main():
    check_input_files(files)
    
    print("\nDone")
    
if __name__ == "__main__":
    main()