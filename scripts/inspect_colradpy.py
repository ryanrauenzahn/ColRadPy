from colradpy import colradpy
import numpy as np

cr = colradpy(
    "../atomic_data/mom97_ls#c2.dat",
    np.array([0]),
    np.array([60.0]),
    np.array([5e14])
)

cr.solve_cr()

# Look at what keys are being saved from input and what keys are being calculated by colradpy
print("\nTOP LEVEL KEYS")
print(cr.data.keys())

print("\nPROCESSED KEYS")
print(cr.data["processed"].keys())

print("\nMETHODS")
print([m for m in dir(cr) if not m.startswith("_")])

# Inspect the shapes of some of the important processed keys
print("scd shape:", cr.data["processed"]["scd"].shape)
print("acd shape:", cr.data["processed"]["acd"].shape)
print("pecs shape:", cr.data["processed"]["pecs"].shape)

# Inspect the values of the SCD and ACD
print("scd:", cr.data["processed"]["scd"])
print("acd:", cr.data["processed"]["acd"])