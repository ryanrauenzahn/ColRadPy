from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from colradpy import colradpy

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
atomic_data_dir = PROJECT_ROOT / "atomic_data"

files = {
    "C II":  atomic_data_dir / "mom97_ls#c1.dat",
    "C III": atomic_data_dir / "mom97_ls#c2.dat",
    "C IV":  atomic_data_dir / "mom97_ls#c3.dat",
    "C V":   atomic_data_dir / "mom97_ls#c4.dat",
    "C VI":  atomic_data_dir / "mom97_ls#c5.dat",
}

# Plasma grid
temperature_grid = np.linspace(1, 100, 500)  # eV
ne = np.array([5.0e14])                      # cm^-3

# Extract SCD and ACD from each atomic data file
gcr_scd = {}
gcr_acd = {}

for label, filepath in files.items():
    print(f"\nRunning ColRadPy for {label}: {filepath}")

    cr = colradpy(
        str(filepath),
        metas=np.array([0]),
        temp_grid=temperature_grid,
        electron_den=ne,
        use_recombination=True,
        use_ionization=True,
        suppliment_with_ecip=True,
    )

    cr.solve_cr()
    scd = cr.data["processed"]["scd"]
    acd = cr.data["processed"]["acd"]
    print("SCD shape:", scd.shape)
    print("ACD shape:", acd.shape)
    gcr_scd[label] = scd[0, 0, :, 0]
    gcr_acd[label] = acd[0, 0, :, 0]

# Build charge-state ionization balance
stage_labels = ["C II", "C III", "C IV", "C V", "C VI", "C VII"]
n_stages = len(stage_labels)
n_temps = len(temperature_grid)

fractions = np.zeros((n_stages, n_temps))
scd_keys = ["C II", "C III", "C IV", "C V", "C VI"]
acd_keys = ["C II", "C III", "C IV", "C V", "C VI"]

for t_idx in range(n_temps):
    matrix = np.zeros((n_stages, n_stages))

    # Ionization upward: C z -> C z+1
    for s, key in enumerate(scd_keys):
        rate = gcr_scd[key][t_idx] * ne[0]
        matrix[s, s] -= rate
        matrix[s + 1, s] += rate

    # Recombination downward: C z+1 -> C z
    for s, key in enumerate(acd_keys):
        rate = gcr_acd[key][t_idx] * ne[0]
        matrix[s + 1, s + 1] -= rate
        matrix[s, s + 1] += rate

    # Replace one equation with normalization: sum fractions = 1 (100%) total
    matrix[-1, :] = 1.0

    rhs = np.zeros(n_stages)
    rhs[-1] = 1.0

    try:
        fractions[:, t_idx] = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        fractions[:, t_idx] = np.nan

# Clean numerical noise
fractions[np.abs(fractions) < 1e-30] = 0.0

print("\nPopulation shums")
print(np.nanmin(np.sum(fractions, axis = 0)), np.nanmax(np.sum(fractions, axis = 0)))

# Plot
fig, ax = plt.subplots(figsize=(7, 5))

plot_labels = ["C III", "C IV", "C V", "C VI"]
plot_indices = [1, 2, 3, 4]

for idx, label in zip(plot_indices, plot_labels):
    ax.semilogy(
        temperature_grid,
        fractions[idx, :],
        linewidth = 1.8,
        label = label,
    )

ax.set_xlabel("T (eV)", fontsize=13)
ax.set_ylabel("Ionization Fraction", fontsize=13)
ax.set_title(
    r"$n_e = 5 \times 10^{14}$ cm$^{-3}$",
    fontsize=12,
)
ax.set_xlim(0, 100)
ax.set_ylim(1e-6, 1.5)
ax.legend(fontsize = 10, loc="center right")
ax.grid(True, which = "both", linestyle = "--", alpha = 0.4)

plt.tight_layout()

out_file = PROJECT_ROOT / "fig4_ionization_balance.png"
plt.savefig(out_file, dpi=150)
plt.show()

print(f"\nFigure saved to {out_file}")