import numpy as np
import h5py
from timeit import repeat
from functools import partial
import sys
import bitarray
import argparse


# filename_pattern = "data/bits_1000x65536_{}_mutate.hdf5"
filename_pattern = "data/bits_10000x262144_{}_mutate.hdf5"

storage_types = ["bool", "byte", "uint32"]

parser = argparse.ArgumentParser()
parser.add_argument("--skip", "-s", choices=storage_types, action="append")


def compute_xor(vector, matrix):
    result = np.bitwise_xor(vector.T, matrix)
    return result


def compute_bits(xor, storage, vector_length):
    # unpack or pack the data in the xor matrix so that they can be further processed (bits counted)
    if storage == "bool":
        result = np.packbits(xor, axis=1)
    elif storage == "byte":
        result = xor
    elif storage == "uint32":
        order = sys.byteorder  # each 4 bytes have to be reversed if byteorder is "little"
        if order == "little":
            result = xor.view(np.uint8).reshape(-1, 4)[:, ::-1].reshape(-1, vector_length)
        elif order == "big":
            result = xor.view(np.uint8)

    return result


def bitifize(vec):
    b = bitarray.bitarray(endian="big")
    b.frombytes(vec.tobytes())
    return b


def compute_nonzero(bits):  # i.e. bit count / pop count
    return [bitifize(v).count() for v in bits]


def hamming(vector, matrix, vector_length):
    # compute XOR
    xor = compute_xor(vector, matrix)

    bits = compute_bits(xor, storage, vector_length)

    result = compute_nonzero(bits)

    return result


def run_stats(run_list, runs):
    normalized = np.array(run_list) / runs
    mean = np.mean(normalized)
    std = np.std(normalized)
    return f"avg = {mean} (std = {std})"


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

    bit_vector = bitarray.bitarray(endian="big")
    bit_vector.frombytes(vector.tobytes())
    bit_matrix = bitarray.bitarray(endian="big")
    bit_matrix.frombytes(vector.tobytes())
    # result = hamming(vector, matrix)

    result = hamming(vector, matrix, info["vector_length"])
    print(f"Result check: {np.unique(result)}")

    setup = "from __main__ import hamming"
    n_runs = 100
    n_reps = 3

    time_result = repeat(partial(compute_xor, vector, matrix), repeat=n_reps, number=n_runs)
    print(f"XOR computation runtime: {run_stats(time_result, n_runs)}")
    xor = compute_xor(vector, matrix)

    time_result = repeat(partial(compute_bits, xor, storage, info["vector_length"]), setup=setup, repeat=n_reps, number=n_runs)
    print(f"Bit unpacking runtime: {run_stats(time_result, n_runs)}")
    bits = compute_bits(xor, storage, info["vector_length"])

    time_result = repeat(partial(compute_nonzero, bits), setup=setup, repeat=n_reps, number=n_runs)
    print(f"Active bit counting runtime: {run_stats(time_result, n_runs)}")

    time_result = repeat(partial(hamming, vector, matrix, info["vector_length"]), setup=setup, repeat=n_reps, number=n_runs)
    print(f"Hamming distance total runtime: {run_stats(time_result, n_runs)}")
