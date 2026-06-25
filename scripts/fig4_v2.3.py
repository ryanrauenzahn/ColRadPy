from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Use the newer ionization_balance_class version.
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

# Metastables
metas = [
    np.array([0]),      # C I
    np.array([0, 1]),   # C II
    np.array([0, 1]),   # C III
    np.array([0]),      # C IV
    np.array([0, 1]),   # C V
    np.array([0]),      # C VI
]

# Plasma grid

Te = np.linspace(1.0, 100.0, 500)  # eV
ne = np.array([5.0e14])            # cm^-3


# Initial abundance

print("\n--- Building ionization_balance_class object ---")

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

# Populate ionization matrix

print("\n--- Populating ionization matrix ---")

ib.populate_ion_matrix()

ion_matrix = ib.data["ion_matrix"]
n_states = ion_matrix.shape[0]

print("ion_matrix shape:", ion_matrix.shape)
print("number of tracked states:", n_states)

# Solve time-dependent balance long enough to approximate steady state

n0 = np.zeros(n_states)
n0[0] = 1.0

# Similar spirit to official example: evolve far enough to settle.
times = np.geomspace(1e-8, 200.0, 300)

print("\n--- Solving ionization balance ---")

ib.solve_no_source(n0=n0, td_t=times)

# For ionization_balance_class, output lives here:
pops = np.real(ib.data["processed"]["pops_td"])

print("pops_td shape:", pops.shape)

# Expected shape: [state, time, Te, ne]
state_pops = pops[:, -1, :, 0]

print("state_pops shape:", state_pops.shape)
print("state sum min:", np.min(np.sum(state_pops, axis=0)))
print("state sum max:", np.max(np.sum(state_pops, axis=0)))

# Collapse metastable-resolved states into charge states

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

for label, meta in zip(charge_labels, metas):
    n_meta = len(meta)

    charge_fractions[label] = np.sum(
        state_pops[offset:offset + n_meta, :],
        axis=0,
    )

    offset += n_meta

# Safety check for extra terminal states.
if offset < n_states:
    print(f"\nWarning: {n_states - offset} extra tracked state(s) after C VI.")
    for i in range(offset, n_states):
        max_val = np.max(state_pops[i, :])
        max_T = Te[np.argmax(state_pops[i, :])]
        print(f"extra state {i}: max={max_val:.3e} at T={max_T:.1f} eV")

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

# Plot ionization balance

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