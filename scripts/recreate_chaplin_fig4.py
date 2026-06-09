from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from colradpy.ionization_balance_class import ionization_balance

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
atomic_data_dir = PROJECT_ROOT / "atomic_data"

files = [
    str(atomic_data_dir / "mom97_ls#c1.dat"),  # C II
    str(atomic_data_dir / "mom97_ls#c2.dat"),  # C III
    str(atomic_data_dir / "mom97_ls#c3.dat"),  # C IV
    str(atomic_data_dir / "mom97_ls#c4.dat"),  # C V
    str(atomic_data_dir / "mom97_ls#c5.dat"),  # C VI
]

for f in files:
    if not Path(f).exists():
        raise FileNotFoundError(f"Missing file: {f}")

metas = [np.array([0]) for _ in files]

Te = np.linspace(1.0, 100.0, 500)  # eV
ne = np.array([5.0e14])            # cm^-3

# stops the ionizing upward issue on C VI
use_ionization = [True, True, True, True, False]

print("\n--- Building ColRadPy ionization balance object ---")

ib = ionization_balance(
    files,
    metas,
    Te,
    ne,
    use_ionization=use_ionization,
    use_recombination=True,
    suppliment_with_ecip=True,
    use_recombination_three_body=True,
    use_cx=False,
    keep_charge_state_data=False,
)

ib.populate_ion_matrix()

ion_matrix = ib.data["ion_matrix"]
n_states = ion_matrix.shape[0]

print("ion_matrix shape:", ion_matrix.shape)
print("number of tracked states:", n_states)

# show GCR shapes

print("\n--- GCR shape diagnostics ---")

for i, file_path in enumerate(files):
    g = ib.data["cr_data"]["gcrs"][str(i)]

    print(f"\nFile {i}: {Path(file_path).name}")
    print("qcd shape:", g["qcd"].shape)
    print("scd shape:", g["scd"].shape)
    print("acd shape:", g["acd"].shape)
    print("xcd shape:", g["xcd"].shape)

# Solve ionization balance time-dependently

n0 = np.zeros(n_states)
n0[0] = 1.0

times = np.array([1e-3])

print("\n--- Solving time-dependent ionization balance ---")
ib.solve_no_source(n0=n0, td_t=times)

pops = ib.data["processed"]["pops_td"]
state_pops = np.real(pops[:, -1, :, 0])

print("pops_td shape:", pops.shape)
print("state_pops shape:", state_pops.shape)
print("state population sum min:", np.min(np.sum(state_pops, axis=0)))
print("state population sum max:", np.max(np.sum(state_pops, axis=0)))


print("\n--- Raw state diagnostics ---")
for i in range(n_states):
    max_val = np.max(state_pops[i, :])
    max_T = Te[np.argmax(state_pops[i, :])]
    print(f"state {i}: max={max_val:.3e} at T={max_T:.1f} eV")

# First-pass charge fractions
charge_fractions = {
    "C II":  state_pops[0, :],
    "C III": state_pops[1, :],
    "C IV":  state_pops[2, :],
    "C V":   state_pops[3, :],
    "C VI":  state_pops[4, :],
}

print("\n--- Charge-state diagnostics ---")
total_charge_fraction = np.zeros_like(Te)

for label, arr in charge_fractions.items():
    total_charge_fraction += arr
    print(
        f"{label:5s}: max={np.max(arr):.3e} "
        f"at T={Te[np.argmax(arr)]:.1f} eV"
    )

print("charge fraction partial sum min:", np.min(total_charge_fraction))
print("charge fraction partial sum max:", np.max(total_charge_fraction))

# Plot

fig, ax = plt.subplots(figsize=(7, 5))

plot_config = [
    ("C III", "s"),
    ("C IV", "o"),
    ("C V", "^"),
    ("C VI", "v"),
]

stride = max(1, len(Te) // 15)

for label, marker in plot_config:
    y = np.clip(charge_fractions[label], 1e-30, None)

    ax.semilogy(
        Te,
        y,
        linewidth=1.8,
        label=label,
    )

    ax.semilogy(
        Te[::stride],
        y[::stride],
        marker=marker,
        linestyle="None",
        markersize=5,
    )

ax.set_xlabel("T (eV)", fontsize=13)
ax.set_ylabel("Ionization fraction", fontsize=13)
ax.set_title(
    r"Carbon ionization balance from ColRadPy, $n_e = 5 \times 10^{14}$ cm$^{-3}$",
    fontsize=11,
)

ax.set_xlim(0, 100)
ax.set_ylim(1e-5, 1.5)
ax.legend(fontsize=10, loc="center right")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()

out_file = PROJECT_ROOT / "fig4_ionization_balance_colradpy_class.png"
plt.savefig(out_file, dpi=150)
plt.show()

print(f"\nFigure saved to {out_file}")