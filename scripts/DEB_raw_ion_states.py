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

metas = [np.array([0]) for _ in files]

Te = np.linspace(1.0, 100.0, 500)
ne = np.array([5.0e14])

use_ionization = [True, True, True, True, False]

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

n_states = ib.data["ion_matrix"].shape[0]

n0 = np.zeros(n_states)
n0[0] = 1.0

times = np.array([1e-3])
ib.solve_no_source(n0=n0, td_t=times)

pops = ib.data["processed"]["pops_td"]
state_pops = np.real(pops[:, -1, :, 0])

print("\n--- Raw state diagnostics ---")
print("state_pops shape:", state_pops.shape)
print("sum min:", np.min(np.sum(state_pops, axis=0)))
print("sum max:", np.max(np.sum(state_pops, axis=0)))

for i in range(n_states):
    max_val = np.max(state_pops[i, :])
    max_T = Te[np.argmax(state_pops[i, :])]

    vals = []
    for target_T in [40, 50, 60, 70, 80, 90, 100]:
        idx = np.argmin(np.abs(Te - target_T))
        vals.append(f"{target_T}eV={state_pops[i, idx]:.3e}")

    print(f"state {i}: max={max_val:.3e} at T={max_T:.1f} eV | " + ", ".join(vals))

# Plot every raw state
fig, ax = plt.subplots(figsize=(8, 5))

for i in range(n_states):
    y = np.clip(state_pops[i, :], 1e-30, None)
    ax.semilogy(Te, y, linewidth=1.5, label=f"state {i}")

ax.set_xlabel("T (eV)")
ax.set_ylabel("Raw state population")
ax.set_title("Raw ionization-balance state populations")
ax.set_xlim(0, 100)
ax.set_ylim(1e-8, 1.5)
ax.legend(fontsize=8, loc="center right")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()