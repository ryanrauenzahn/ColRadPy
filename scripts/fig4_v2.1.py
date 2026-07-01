from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from colradpy.ionization_balance_class import ionization_balance

# --------------------------------------------------
# Paths
# --------------------------------------------------

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

for f in files:
    if not Path(f).exists():
        raise FileNotFoundError(f"Missing file: {f}")

# --------------------------------------------------
# Metastables copied from official ColRadPy example
# --------------------------------------------------

metas = np.array([
    np.array([0]),
    np.array([0,1]),
    np.array([0,1]),
    np.array([0]),
    np.array([0,1]),
    np.array([0]),
], dtype=object)

# --------------------------------------------------
# Plasma conditions
# --------------------------------------------------

Te = np.linspace(1.0, 100.0, 500)
ne = np.array([5.0e14])

# --------------------------------------------------
# Initial abundance
# --------------------------------------------------

n_states = sum(len(m) for m in metas)

initial_abundances = np.zeros(n_states)
initial_abundances[0] = 1.0

source = np.zeros(n_states)

# --------------------------------------------------
# Build ionization balance object
# --------------------------------------------------

print("\n--- Building ionization balance object ---")

ib = ionization_balance(
    files,
    metas,
    Te,
    ne,
    keep_charge_state_data=False,
    use_cx=False,
    temp_dens_pair=False,
    scale_file_ioniz=True,
    init_abund=initial_abundances,
    source=source,
)

# --------------------------------------------------
# Populate matrix
# --------------------------------------------------

print("\n--- Populating ion matrix ---")

ib.populate_ion_matrix()

print("ion_matrix shape:",
      ib.data["ion_matrix"].shape)

# --------------------------------------------------
# Solve steady-state
# --------------------------------------------------

print("\n--- Solving steady-state ---")

ib.solve_time_independent()

if "pops_ss" not in ib.data["processed"]:
    raise RuntimeError("No pops_ss produced.")

pops_ss = np.real(
    ib.data["processed"]["pops_ss"][:, :, 0]
)

print("pops_ss shape:", pops_ss.shape)

# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

for i in range(pops_ss.shape[0]):
    peak_idx = np.argmax(pops_ss[i])
    print(
        f"state {i}: "
        f"max={np.max(pops_ss[i]):.3e} "
        f"at T={Te[peak_idx]:.1f} eV"
    )

# --------------------------------------------------
# Collapse metastables into charge states
# --------------------------------------------------

charge_fractions = {}

offset = 0

charge_labels = [
    "C I",
    "C II",
    "C III",
    "C IV",
    "C V",
    "C VI",
]

for label, metastables in zip(charge_labels, metas):

    n_meta = len(metastables)

    charge_fractions[label] = np.sum(
        pops_ss[offset:offset+n_meta],
        axis=0
    )

    offset += n_meta

# --------------------------------------------------
# Normalization check
# --------------------------------------------------

total = np.zeros_like(Te)

for label in charge_labels:
    total += charge_fractions[label]

print(
    "\nCharge-state sum:",
    np.min(total),
    np.max(total)
)

# --------------------------------------------------
# Plot Fig 4
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

plot_species = [
    ("C III", "s"),
    ("C IV", "o"),
    ("C V", "^"),
    ("C VI", "v"),
]

stride = len(Te) // 15

for label, marker in plot_species:

    y = np.clip(
        charge_fractions[label],
        1e-30,
        None
    )

    ax.semilogy(
        Te,
        y,
        linewidth=1.8,
        label=label
    )

    ax.semilogy(
        Te[::stride],
        y[::stride],
        linestyle="None",
        marker=marker,
        markersize=5
    )

ax.set_xlabel("T (eV)")
ax.set_ylabel("Ionization Fraction")

ax.set_title(
    r"$n_e = 5\times10^{14}\ {\rm cm^{-3}}$"
)

ax.set_xlim(0, 100)
ax.set_ylim(1e-6, 1.5)

ax.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.4
)

ax.legend()

plt.tight_layout()

outfile = (
    PROJECT_ROOT
    / "fig4_recreated_using_pops_ss.png"
)

plt.savefig(outfile, dpi=150)
plt.show()

print(f"\nSaved: {outfile}")