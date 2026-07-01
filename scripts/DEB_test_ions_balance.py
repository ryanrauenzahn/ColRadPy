from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from colradpy import colradpy

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
atomic_data_dir = PROJECT_ROOT / "atomic_data"

Te = np.linspace(1.0, 100.0, 500)
ne = np.array([5.0e14])

files = {
    "C IV": atomic_data_dir / "mom97_ls#c3.dat",
    "C V":  atomic_data_dir / "mom97_ls#c4.dat",
    "C VI": atomic_data_dir / "mom97_ls#c5.dat",
}

rates = {}

for label, path in files.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    print(f"\nRunning ColRadPy for {label}: {path.name}")

    cr = colradpy(
        str(path),
        metas=np.array([0]),
        temp_grid=Te,
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

    rates[label] = {
        "scd": np.real(scd[0, 0, :, 0]),
        "acd": np.real(acd[0, 0, :, 0]),
    }

# Effective rate coefficients:
# C IV file: SCD = C IV -> C V, ACD = C V -> C IV
# C V file:  SCD = C V  -> C VI, ACD = C VI -> C V
# C VI file: mostly useful to inspect, but C VI -> C VII may be problematic

fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(Te, rates["C IV"]["scd"], label="SCD C IV → C V")
ax.semilogy(Te, rates["C IV"]["acd"], label="ACD C V → C IV")
ax.semilogy(Te, rates["C V"]["scd"], label="SCD C V → C VI")
ax.semilogy(Te, rates["C V"]["acd"], label="ACD C VI → C V")

ax.set_xlabel("T (eV)")
ax.set_ylabel("Rate coefficient (cm^3/s)")
ax.set_title("High-stage carbon SCD/ACD coefficients from ColRadPy")
ax.set_xlim(0, 100)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()

# Ratio controlling C VI / C V balance approximately:
# f_CVI / f_CV ≈ SCD_CV / ACD_CV
ratio_cvi_cv = rates["C V"]["scd"] / rates["C V"]["acd"]

fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(Te, ratio_cvi_cv, label="SCD(C V → C VI) / ACD(C VI → C V)")

ax.axhline(1.0, linestyle="--", linewidth=1.2)

ax.set_xlabel("T (eV)")
ax.set_ylabel("Approximate C VI / C V balance ratio")
ax.set_title("Approximate C VI/C V balance from ColRadPy rates")
ax.set_xlim(0, 100)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()

# Print crossing estimate where ratio crosses 1
idx = np.argmin(np.abs(ratio_cvi_cv - 1.0))
print("\nApproximate C VI/C V crossover from rate ratio:")
print(f"T ≈ {Te[idx]:.2f} eV")
print(f"SCD_CV / ACD_CV ≈ {ratio_cvi_cv[idx]:.3e}")

print("\nSelected values:")
for target_T in [40, 50, 60, 70, 80, 90, 100]:
    idx = np.argmin(np.abs(Te - target_T))
    print(
        f"T={Te[idx]:.1f} eV | "
        f"SCD_CV={rates['C V']['scd'][idx]:.3e}, "
        f"ACD_CV={rates['C V']['acd'][idx]:.3e}, "
        f"ratio={ratio_cvi_cv[idx]:.3e}"
    )