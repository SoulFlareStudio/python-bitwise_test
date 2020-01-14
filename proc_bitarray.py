import numpy as np
import h5py
import sys
import bitarray
from time_utils import time_method
from hamming import Hamming


class BitarrayHamming(Hamming):

    def compute_xor(self):
        result = np.bitwise_xor(self.vector.T, self.matrix)
        return result

    def convert_bits(self):
        # unpack or pack the data in the xor matrix so that they can be further processed (bits counted)
        if self.storage == "bool":
            result = np.packbits(self.xor, axis=1)
        elif self.storage == "byte":
            result = self.xor
        elif self.storage == "uint32":
            order = sys.byteorder  # each 4 bytes have to be reversed if byteorder is "little"
            if order == "little":
                result = self.xor.view(np.uint8).reshape(-1, 4)[:, ::-1].reshape(-1, self.vector_length // 8)
            elif order == "big":
                result = self.xor.view(np.uint8)

        return result

    def bitifize(self, vec):
        b = bitarray.bitarray(endian="big")
        if isinstance(vec, np.ndarray):
            b.frombytes(vec.tobytes())
        else:
            b.extend(vec)
        return b

    def pop_count(self):  # i.e. bit count / pop count
        return [self.bitifize(v).count() for v in self.bits]

    def quantize(self):
        if self.q_levels == 2:
            threshold = self.vector_length >> 1
            result = self.bitifize(np.array(self.distances) > threshold)
        else:
            raise NotImplementedError()
        return result


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

        # bit_vector = bitarray.bitarray(endian="big")
        # bit_vector.frombytes(vector.tobytes())
        # bit_matrix = bitarray.bitarray(endian="big")
        # bit_matrix.frombytes(vector.tobytes())
        # result = hamming(vector, matrix)

        bit_hamming = BitarrayHamming(vector, matrix, info["vector_length"], storage)
        result = bit_hamming.hamming()
        yield ("info", f"Result check: {np.unique(result)}")

        time_result = time_method(bit_hamming.compute_xor, n_reps=n_reps, n_runs=n_runs)
        yield ("xor", *time_result)
        bit_hamming.xor = bit_hamming.compute_xor()

        time_result = time_method(bit_hamming.convert_bits, n_reps=n_reps, n_runs=n_runs)
        yield ("bit conversion", *time_result)
        bit_hamming.bits = bit_hamming.convert_bits()

        time_result = time_method(bit_hamming.pop_count, n_reps=n_reps, n_runs=n_runs)
        yield ("bit count", *time_result)
        bit_hamming.distances = bit_hamming.pop_count()

        time_result = time_method(bit_hamming.quantize, n_reps=n_reps, n_runs=n_runs)
        yield ("quantization", *time_result)

        time_result = time_method(bit_hamming.hamming, n_reps=n_reps, n_runs=n_runs)
        yield ("hamming distance", *time_result)
