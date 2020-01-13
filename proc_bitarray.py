import numpy as np
import h5py
import sys
import bitarray
from time_utils import time_method


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
            result = xor.view(np.uint8).reshape(-1, 4)[:, ::-1].reshape(-1, vector_length // 8)
        elif order == "big":
            result = xor.view(np.uint8)

    return result


def bitifize(vec):
    b = bitarray.bitarray(endian="big")
    b.frombytes(vec.tobytes())
    return b


def compute_nonzero(bits):  # i.e. bit count / pop count
    return [bitifize(v).count() for v in bits]


def hamming(vector, matrix, vector_length, storage):
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


def run_tests(filename_pattern: str, storage_types: list, n_runs: int, n_reps: int):
    for storage in storage_types:
        filename = filename_pattern.format(storage)
        yield "storage", storage

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

        result = hamming(vector, matrix, info["vector_length"], storage)
        yield ("info", f"Result check: {np.unique(result)}")

        time_result = time_method(
            compute_xor, vector, matrix, n_reps=n_reps, n_runs=n_runs)
        yield ("xor", *time_result)
        xor = compute_xor(vector, matrix)

        time_result = time_method(
            compute_bits, xor, storage, info["vector_length"], n_reps=n_reps, n_runs=n_runs)
        yield ("bit conversion", *time_result)
        bits = compute_bits(xor, storage, info["vector_length"])

        time_result = time_method(
            compute_nonzero, bits, n_reps=n_reps, n_runs=n_runs)
        yield ("bit count", *time_result)

        time_result = time_method(
            hamming, vector, matrix, info["vector_length"], storage, n_reps=n_reps, n_runs=n_runs)
        yield ("hamming distance", *time_result)
