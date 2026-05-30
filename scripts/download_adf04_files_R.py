## Purpose:Downloads ADAS ADF04 atomic data files for carbon ions and saves them into examples/input/test-downloads.
## Author: Ryan Rauenzahn
## Origin Date: 03/10/2026
## Last Edit Date: 5/29/2026
## Notes: Currently set to download carbon II, III, and IV files.

from urllib.request import urlretrieve
from pathlib import Path

# set where to store downloaded files
save_dir = Path("examples/input/test-downloads")
save_dir.mkdir(parents=True, exist_ok=True)

# connect to source (ADAS base URL)
base_url = "http://open.adas.ac.uk/download/adf04/"
remote_suffix = "adas][6/mom97_ls][c1.dat"
out_file = save_dir / "mom97_ls#c1.dat"

url = base_url + remote_suffix

# Confirmation messages that retreiving worked
print("Downloading from:", url)
urlretrieve(url, out_file)
print("Saved to:", out_file)

# ADF04 files to download from source
files = {
    "mom97_ls#c2.dat": "adas][6/mom97_ls][c2.dat",
    "mom97_ls#c3.dat": "adas][6/mom97_ls][c3.dat",
}

for filename, remote_suffix in files.items():
    out_file = save_dir / filename
    url = base_url + remote_suffix

    print(f"\nDownloading {filename}")
    print(f"From: {url}")
    print(f"To:   {out_file}")

    try:
        urlretrieve(url, out_file)

        # Gives error message if it's downloading an HTML page
        text = out_file.read_text(errors="ignore")
        if "OPEN-ADAS Error" in text or "<!DOCTYPE html" in text:
            print(f"FAILED: {filename} saved an HTML error page instead of atomic data.")
            continue
        print(f"SUCCESS: saved {filename}")

        # Print first few lines so to verify it looks like real ADF04 data
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
        
