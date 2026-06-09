from pathlib import Path
import numpy as np
from colradpy.ionization_balance_class import ionization_balance

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
base = PROJECT_ROOT / "atomic_data"

files = [
    str(base / "mom97_ls#c1.dat"),  # C II
    str(base / "mom97_ls#c2.dat"),  # C III
    str(base / "mom97_ls#c3.dat"),  # C IV
]

metas = [
    np.array([0]),
    np.array([0]),
    np.array([0]),
]

Te = np.array([40.0])
ne = np.array([5e14])

ib = ionization_balance(files, metas, Te, ne)

ib.populate_ion_matrix()
ib.solve_time_independent()

print("\nIon matrix shape:")
print(ib.data["ion_matrix"].shape)

print("\nIon matrix:")
print(ib.data["ion_matrix"][:, :, 0, 0])

print("\nColumn sums:")
print(np.sum(ib.data["ion_matrix"][:, :, 0, 0], axis=0))

print("\nProcessed keys:")
print(ib.data["processed"].keys())

print("\nSteady-state populations:")
print(ib.data["processed"]["pops_ss"])

print("\nShape:")
print(ib.data["processed"]["pops_ss"].shape)

print("\nInitial abundance:")
print(ib.data["user"]["init_abund"])

n0 = np.zeros(6)
n0[0] = 1.0

times = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4])

ib.solve_no_source(n0=n0, td_t=times)

print("\nTime-dependent populations:")
print(ib.data["processed"]["pops_td"])

print("\nFinal populations:")
print(ib.data["processed"]["pops_td"][:, -1, 0, 0])

print("\nFinal sum:")
print(np.sum(ib.data["processed"]["pops_td"][:, -1, 0, 0]))