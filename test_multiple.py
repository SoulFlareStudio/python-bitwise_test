import pandas as pd
import numpy as np
from importlib import import_module
import datetime
from itertools import product
import os
import argparse


# Add entries to include testing scripts
implementations = {
    "numpy": {  # name of test/method to be shown
        # name of the script including the testing function (without ".py")
        "script": "proc_numpy"
    },
    "bitarray": {
        "script": "proc_bitarray"
    }
}


# Variable setup
n_runs = 100
n_reps = 3

stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

section_names = ["xor", "bit conversion", "bit count", "hamming distance"]
column_names = [a + b for a, b in product(section_names, ["", " STD"])]
storage_types = ["bool", "byte", "uint32"]

n_vector = 1000
n_elements = 2**16
datapath = "data"

# Parsing console inputs
parser = argparse.ArgumentParser()
parser.add_argument("--skip", "-s", choices=storage_types, action="append", default=[],
                    help=f"Choose which storage types to skip (missing data are skipped automatically). Can be used multiple times to skip multiple storage types.")
parser.add_argument("--path", "-p", default=datapath,
                    help=f"Path to the data folder. (default = 'data')")
parser.add_argument("--n_vector", "-nv", default=n_vector,
                    help="Number of vectors in data. Used to locate the correct datafile. Python expressions can be used, i.e. 1e3 == 1000.")
parser.add_argument("--n_elements", "-ne", default=n_elements,
                    help="Number of elements in each vector (i.e. vector length). Used to locate the correct datafile. . Python expressions can be used, i.e. 2**16 == 65536")

args = parser.parse_args()

storage_types = [st for st in storage_types if st not in args.skip]
datapath = args.path
n_vector = int(eval(args.n_vector))
n_elements = int(eval(args.n_elements))

# Input/output data handling
filename_pattern = os.path.join(datapath, f"bits_{n_vector}x{n_elements}_{{}}_mutate.hdf5")
df = pd.DataFrame(columns=column_names, index=pd.MultiIndex.from_product((implementations.keys(), storage_types)))

# Pre-loop setups
print(f"Testing runtime of Hamming distance for {n_vector} vectors of length {n_elements} compared to a single vector of length {n_elements}")
current_storage = storage_types[0]

# Main loop
for (implementation, impl_info) in implementations.items():
    print(f"--------------------\nRunning test for '{implementation}'...")
    test_script = import_module(impl_info["script"])  # not very nice but works
    for output in test_script.run_tests(filename_pattern, storage_types, n_runs, n_reps):
        if output[0] == "info":
            print(output[1])
        elif output[0] == "storage":
            current_storage = output[1]
            print(f"\nTesting '{current_storage}' storage.")
        elif output[0] in section_names:
            section = output[0]
            runtime = output[1]
            std_dev = output[2]
            df.loc[(implementation, current_storage), section] = runtime
            df.loc[(implementation, current_storage), section + " STD"] = std_dev
            print(f"> {section} run time: {runtime:.4g}s (\u00B1{std_dev:.4g}s)")
        else:
            print(f"! Erroneous yield: {output} !")

print("--------------------")
table_filename = f"output_{n_vector}x{n_elements}_{stamp}.csv"
print("Done.")
print(f"Saving results to {table_filename}.")
df.to_csv(table_filename)
