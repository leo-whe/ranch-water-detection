"""Apply the NWI-interior 'ignore' pass of 03_make_labels.py to already generated label tiles."""
import glob, os, runpy
from multiprocessing import Pool
ml = runpy.run_path("scripts/03_make_labels.py", run_name="labels")
def work(p): return ml["unpenalise_nwi_interior"](p)
if __name__ == "__main__":
    with Pool(24) as p: list(p.imap_unordered(work, sorted(glob.glob(f"data/labels/{os.environ.get('NAIP_YEAR', '2022')}/*.tif"))))
    print("ok")
