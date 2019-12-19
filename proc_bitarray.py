import numpy as np
import h5py
from timeit import repeat
from functools import partial
import bitarray


filename_pattern = "bits_1000x65536_{}_mutate.hdf5"
# filename_pattern = "bits_10000x262144_{}_mutate.hdf5"


def compute_xor(vector, matrix):
    result = np.bitwise_xor(vector.T, matrix)
    return result


def compute_nonzero(bits):
    return np.count_nonzero(bits, 1)


def hamming(vector, matrix, vector_length):
    # compute XOR
    xor = np.bitwise_xor(vector.T, matrix)

    if storage == "bool":
        result = np.sum(xor, 1)
    elif storage == "byte":
        result = np.count_nonzero(np.unpackbits(xor[:, ::-1]).reshape(-1, vector_length).astype(np.bool), 1)
    elif storage == "uint32":
        result = np.count_nonzero(np.unpackbits(xor.view(np.uint8)[:, ::-1]).reshape(-1, vector_length).astype(np.bool), 1)

    return result


def run_stats(run_list, runs):
    normalized = np.array(run_list) / runs
    mean = np.mean(normalized)
    std = np.std(normalized)
    return f"avg = {mean} (std = {std})"


storage = "byte"
filename = filename_pattern.format(storage)
print(f"Computing runtimes for {storage} storage.")

with h5py.File(filename, "r") as f:
    vector = f["vector"][()]
    matrix = f["matrix"][()]
    info = dict()
    info.update(**f.attrs)

vector = np.unpackbits(xor[:, ::-1]).reshape(-1, vector_length).astype(np.bool)
# result = hamming(vector, matrix)


result = hamming(vector, matrix, info["vector_length"])
print(f"Result check: {np.unique(result)}")

setup = "from __main__ import hamming"
n_runs = 100
n_reps = 3

time_result = repeat(partial(compute_xor, vector, matrix), setup=setup, repeat=n_reps, number=n_runs)
print(f"XOR computation runtime: {run_stats(time_result, n_runs)}")

xor = compute_xor(vector, matrix)
time_result = repeat(partial(compute_bits, xor, storage, info["vector_length"]), setup=setup, repeat=n_reps, number=n_runs)
print(f"Bit unpacking runtime: {run_stats(time_result, n_runs)}")

bits = compute_bits(xor, storage, info["vector_length"])
time_result = repeat(partial(compute_nonzero, bits), setup=setup, repeat=n_reps, number=n_runs)
print(f"Active bit counting runtime: {run_stats(time_result, n_runs)}")

time_result = repeat(partial(hamming, vector, matrix, info["vector_length"]), setup=setup, repeat=n_reps, number=n_runs)
print(f"Hamming distance total runtime: {run_stats(time_result, n_runs)}")
