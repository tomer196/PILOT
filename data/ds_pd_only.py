import pathlib
import h5py
import os

# change orig and dist folder
orig_folder = '/ext_mnt/tomer/multicoil_val/'
dist_folder = '/home/tomerweiss/Datasets/multicoil_val'
files = list(pathlib.Path(orig_folder).iterdir())

i = 0
for fname in sorted(files):
    with h5py.File(fname, 'r') as data:
        if data.attrs['acquisition'] == 'CORPD_FBK':   # should be 'CORPD_FBK' or 'CORPDFS_FBK'
            os.system(f"cp {fname} {dist_folder + '/' + fname.name}")
            i += 1
            print(f"{i}, {dist_folder + '/' + fname.name}")

