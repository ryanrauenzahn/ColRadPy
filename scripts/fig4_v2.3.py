"""
Reproduces Figure 4 from Chaplin et al.

This script computes the equilibrium carbon ionization balance using
ColRadPy and ADAS ADF04 atomic data. The populations of the explicitly
tracked metastable states are summed to obtain the total charge-state
fractions, which are plotted as a function of electron temperature at

    ne = 5 × 10^14 cm^-3.

Atomic data:
    mom97_ls#c0.dat  (C I)
    ...
    mom97_ls#c5.dat  (C VI)

Author: Ryan Rauenzahn
Date: June 17, 2026
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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

# Metastable levels to explicitly track for each ion
# The indices correspond to level number in each ADF04 file

metas = [
    np.array([0]),      # C I
    np.array([0, 1]),   # C II
    np.array([0, 1]),   # C III
    np.array([0]),      # C IV
    np.array([0, 1]),   # C V
    np.array([0]),      # C VI
]

# Electron temperature and density grid over which the ionization balance is calculated

Te = np.linspace(1.0, 100.0, 500)  # eV
ne = np.array([5.0e14])            # cm^-3

# Initial abundance

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

# Assemble the rate matrix containing ionization and recombination transitions between the tracked metastable states.

ib.populate_ion_matrix()

n_states = ib.data["ion_matrix"].shape[0]

# Start with all carbon neutral and evolve the coupled rate equations until the populations reach steady state.

n0 = np.zeros(n_states)
n0[0] = 1.0

# Time grid for the transient evolution.

times = np.geomspace(1e-8, 200.0, 300)

ib.solve_no_source(n0=n0, td_t=times)

# Time-dependent metastable populations

pops = np.real(ib.data["processed"]["pops_td"])

# Take the final time step (where there is steady-state populations).

state_pops = pops[:, -1, :, 0]

# Sum the metastable populations belonging to the same ionization stage to obtain the total charge-state fraction

charge_labels = [
    "C I",
    "C II",
    "C III",
    "C IV",
    "C V",
    "C VI",
]

charge_fractions = {}
offset = 0

for label, metastables in zip(charge_labels, metas):
    n_meta = len(metastables)

    charge_fractions[label] = np.sum(
        state_pops[offset:offset + n_meta, :],
        axis=0,
    )

    offset += n_meta

# Diagnostics

print("\n--- Charge-state diagnostics ---")

total = np.zeros_like(Te)

for label in charge_labels:
    total += charge_fractions[label]
    max_val = np.max(charge_fractions[label])
    max_T = Te[np.argmax(charge_fractions[label])]
    print(f"{label:5s}: max={max_val:.3e} at T={max_T:.1f} eV")

print("charge sum min:", np.min(total))
print("charge sum max:", np.max(total))

# Plotting ionization balance

fig, ax = plt.subplots(figsize=(7, 5))

plot_config = [
    ("C III", "s", "black"),
    ("C IV", "o", "red"),
    ("C V", "^", "blue"),
    ("C VI", "v", "green"),
]

stride = max(1, len(Te) // 15)

for label, marker, color in plot_config:
    y = np.clip(charge_fractions[label], 1e-30, None)

    ax.semilogy(
        Te,
        y,
        color=color,
        linewidth=1.8,
        label=label,
    )

    ax.semilogy(
        Te[::stride],
        y[::stride],
        linestyle="None",
        marker=marker,
        color=color,
        markersize=5,
    )

ax.set_xlabel("T (eV)", fontsize=13)
ax.set_ylabel("Ionization fraction", fontsize=13)
ax.set_title(
    r"Carbon ionization balance, $n_e = 5\times10^{14}$ cm$^{-3}$",
    fontsize=11,
)

ax.set_xlim(0, 100)
ax.set_ylim(1e-6, 1.5)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(fontsize=10, loc="center right")

plt.tight_layout()

out_file = PROJECT_ROOT / "fig4_v2_3_ionization_balance_class.png"
plt.savefig(out_file, dpi=150)
plt.show()

print(f"\nSaved: {out_file}")