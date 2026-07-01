from pathlib import Path
import numpy as np
from colradpy.ionization_balance_class import ionization_balance

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
atomic_data_dir = PROJECT_ROOT / "atomic_data"

files = [
    str(atomic_data_dir / "mom97_ls#c1.dat"),
    str(atomic_data_dir / "mom97_ls#c2.dat"),
    str(atomic_data_dir / "mom97_ls#c3.dat"),
    str(atomic_data_dir / "mom97_ls#c4.dat"),
]

metas = [np.array([0]) for _ in files]

Te = np.array([60.0])
ne = np.array([5.0e14])

ib = ionization_balance(files, metas, Te, ne, keep_charge_state_data=True)
ib.populate_ion_matrix()

print("\nTOP LEVEL KEYS")
print(ib.data.keys())

print("\nUSER KEYS")
print(ib.data["user"].keys())

print("\nCR_DATA KEYS")
print(ib.data["cr_data"].keys())

print("\nGCRS KEYS")
print(ib.data["cr_data"]["gcrs"].keys())

print("\nION MATRIX SHAPE")
print(ib.data["ion_matrix"].shape)

for i in range(len(files)):
    print(f"\n--- FILE {i}: {Path(files[i]).name} ---")
    g = ib.data["cr_data"]["gcrs"][str(i)]
    print("gcr keys:", g.keys())
    for key in g.keys():
        try:
            print(key, "shape:", g[key].shape)
        except Exception:
            print(key, type(g[key]))

print("\nSTAGE DATA KEYS")
print(ib.data["cr_data"]["stage_data"].keys())

for i in range(len(files)):
    sd = ib.data["cr_data"]["stage_data"][str(i)]
    print(f"\n--- STAGE DATA {i}: {Path(files[i]).name} ---")
    print("top keys:", sd.keys())

    if "atomic" in sd:
        print("atomic keys:", sd["atomic"].keys())

        for possible_key in [
            "metas",
            "energy",
            "ion_pot",
            "config",
            "eissner",
            "S",
            "L",
            "w",
            "zpla",
        ]:
            if possible_key in sd["atomic"]:
                val = sd["atomic"][possible_key]
                print(f"atomic['{possible_key}']:", val)

    if "input_file" in sd:
        print("input_file keys:", sd["input_file"].keys())

    if "processed" in sd:
        print("processed keys:", sd["processed"].keys())