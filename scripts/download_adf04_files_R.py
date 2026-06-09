# Purpose:Downloads ADAS ADF04 atomic data files for carbon ions and saves them into atomic_data.
# Author: Ryan Rauenzahn
# Origin Date: 03/10/2026
# Last Edit Date: 06/08/2026
# Notes: Currently set to download carbon II, III, IV, V, and VI files.

from urllib.request import urlretrieve
from pathlib import Path

# Save directly into your project atomic_data folder
save_dir = Path("atomic_data")
save_dir.mkdir(parents=True, exist_ok=True)

base_url = "http://open.adas.ac.uk/download/adf04/"

# Carbon ADF04 files
files = {
    "mom97_ls#c0.dat": "adas][6/mom97_ls][c0.dat",  # C I
    "mom97_ls#c1.dat": "adas][6/mom97_ls][c1.dat",  # C II
    "mom97_ls#c2.dat": "adas][6/mom97_ls][c2.dat",  # C III
    "mom97_ls#c3.dat": "adas][6/mom97_ls][c3.dat",  # C IV
    "mom97_ls#c4.dat": "adas][6/mom97_ls][c4.dat",  # C V
    "mom97_ls#c5.dat": "adas][6/mom97_ls][c5.dat",  # C VI
}

for filename, remote_suffix in files.items():
    out_file = save_dir / filename
    url = base_url + remote_suffix

    print(f"\nDownloading {filename}")
    print(f"From: {url}")
    print(f"To:   {out_file}")

    try:
        urlretrieve(url, out_file)

        text = out_file.read_text(errors="ignore")

        if "OPEN-ADAS Error" in text or "<!DOCTYPE html" in text or "<html" in text.lower():
            print(f"FAILED: {filename} saved an HTML error page instead of atomic data.")
            out_file.unlink(missing_ok=True)
            continue

        print(f"SUCCESS: saved {filename}")

        print("First 5 lines preview:")
        with open(out_file, "r", errors="ignore") as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                print(repr(line.rstrip()))

    except Exception as e:
        print(f"FAILED: {filename}")
        print(f"Reason: {e}")