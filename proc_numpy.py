import numpy as np
import h5py
from time_utils import time_method


def compute_xor(vector, matrix):
    result = np.bitwise_xor(vector.T, matrix)
    return result


def compute_bits(xor, storage, vector_length):
    if storage == "bool":
        result = xor
    elif storage == "byte":
        result = np.unpackbits(
            xor[:, ::-1]).reshape(-1, vector_length).astype(np.bool)
    elif storage == "uint32":
        result = np.unpackbits(xor.view(np.uint8)[:, ::-1]).reshape(-1, vector_length).astype(np.bool)

    return result


def compute_nonzero(bits):
    return np.count_nonzero(bits, 1)


def hamming(vector, matrix, vector_length, storage):
    # compute XOR
    xor = np.bitwise_xor(vector.T, matrix)

    if storage == "bool":
        result = np.sum(xor, 1)
    elif storage == "byte":
        result = np.count_nonzero(np.unpackbits(
            xor[:, ::-1]).reshape(-1, vector_length).astype(np.bool), 1)
    elif storage == "uint32":
        result = np.count_nonzero(np.unpackbits(
            xor.view(np.uint8)[:, ::-1]).reshape(-1, vector_length).astype(np.bool), 1)

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

        # result = hamming(vector, matrix)

        result = hamming(vector, matrix, info["vector_length"], storage)
        print(f"Result check: {np.unique(result)}")

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
