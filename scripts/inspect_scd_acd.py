from pathlib import Path
from colradpy import colradpy
import numpy as np

base = Path("../atomic_data")

files = {
    "C II": "mom97_ls#c1.dat",
    "C III": "mom97_ls#c2.dat",
    "C IV": "mom97_ls#c3.dat",
}

for ion, filename in files.items():

    cr = colradpy(
        str(base / filename),
        np.array([0]),
        np.array([60.0]),
        np.array([5e14])
    )

    cr.solve_cr()

    scd = cr.data["processed"]["scd"][0,0,0,0]
    acd = cr.data["processed"]["acd"][0,0,0,0]

    print(f"\n{ion}")
    print(f"SCD = {scd:.6e}")
    print(f"ACD = {acd:.6e}")