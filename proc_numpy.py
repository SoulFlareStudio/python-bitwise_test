import numpy as np
import h5py
from time_utils import time_method
from hamming import Hamming


class NumpyHamming(Hamming):

    def compute_xor(self):
        result = np.bitwise_xor(self.vector.T, self.matrix)
        return result

    def convert_bits(self):
        if self.storage == "bool":
            result = self.xor
        elif self.storage == "byte":
            result = np.unpackbits(self.xor[:, ::-1]).reshape(-1, self.vector_length).astype(np.bool)
        elif self.storage == "uint32":
            result = np.unpackbits(self.xor.view(np.uint8)[:, ::-1]).reshape(-1, self.vector_length).astype(np.bool)
        return result

    def pop_count(self):
        return np.count_nonzero(self.bits, 1)

    def quantize(self):
        if self.q_levels == 2:
            threshold = self.vector_length >> 1
            result = (np.array(self.distances) > threshold).astype(self.storage)
        else:
            raise NotImplementedError()
        return result

    # def hamming(self):
    #     # compute XOR
    #     xor = np.bitwise_xor(vector.T, matrix)

    #     if storage == "bool":
    #         result = np.sum(xor, 1)
    #     elif storage == "byte":
    #         result = np.count_nonzero(np.unpackbits(
    #             xor[:, ::-1]).reshape(-1, vector_length).astype(np.bool), 1)
    #     elif storage == "uint32":
    #         result = np.count_nonzero(np.unpackbits(
    #             xor.view(np.uint8)[:, ::-1]).reshape(-1, vector_length).astype(np.bool), 1)

    #     return result


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

        hamming = NumpyHamming(vector, matrix, info["vector_length"], storage)
        result = hamming.hamming()
        yield ("info", f"Result check: {np.unique(result)}")

        time_result = time_method(hamming.compute_xor, n_reps=n_reps, n_runs=n_runs)
        yield ("xor", *time_result)
        hamming.xor = hamming.compute_xor()

        time_result = time_method(hamming.convert_bits, n_reps=n_reps, n_runs=n_runs)
        yield ("bit conversion", *time_result)
        hamming.bits = hamming.convert_bits()

        time_result = time_method(hamming.pop_count, n_reps=n_reps, n_runs=n_runs)
        yield ("bit count", *time_result)
        hamming.distances = hamming.pop_count()

        time_result = time_method(hamming.quantize, n_reps=n_reps, n_runs=n_runs)
        yield ("quantization", *time_result)

        time_result = time_method(hamming.hamming, n_reps=n_reps, n_runs=n_runs)
        yield ("hamming distance", *time_result)
