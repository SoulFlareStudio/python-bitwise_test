import numpy as np
import h5py

filename_pattern = "data/bits_1000x65536_{}_mutate.hdf5"
# filename_pattern = "bits_10000x262144_{}_mutate.hdf5"

storage_types = ["bool", "byte", "uint32"]

for storage in storage_types:
    filename = filename_pattern.format(storage)
    print(f"Computing runtimes for {storage} storage.")

    try:
        with h5py.File(filename, "r") as f:
            vector = f["vector"][()]
            matrix = f["matrix"][()]
            info = dict()
            info.update(**f.attrs)
    except IOError:
        print("File with data not found, skipping.")
        continue

    print(f"Loaded data from '{filename}' of type '{storage}'.")
