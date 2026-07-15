#!/usr/bin/env python
"""Download the RedDB molecule + reaction tables from Harvard Dataverse.

RedDB: Sengul et al., Scientific Data 2022 (doi:10.1038/s41597-022-01832-2),
data DOI 10.7910/DVN/F3QFSQ. Files are served via a redirect to S3.
"""
import os, sys, urllib.request

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FILES = {  # Dataverse datafile id -> local name
    "6573579": "RedDBv2_reaction.tab",
    "6573578": "RedDBv2_molecule.tab",
}
BASE = "https://dataverse.harvard.edu/api/access/datafile/"

def main():
    os.makedirs(RAW, exist_ok=True)
    for fid, name in FILES.items():
        out = os.path.join(RAW, name)
        if os.path.exists(out):
            print(f"[skip] {name} already present"); continue
        print(f"[get ] {name} ...")
        # follow the 303 -> S3 redirect
        req = urllib.request.Request(BASE + fid, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req) as r, open(out, "wb") as f:
            f.write(r.read())
        print(f"       -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
