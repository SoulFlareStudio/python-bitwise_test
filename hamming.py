import numpy as np


class Hamming():

    def __init__(self, vector, matrix, vector_length, storage=None, q_levels = 2):
        self.vector = vector
        self.matrix = matrix
        self.vector_length = vector_length
        self.storage = storage if storage is not None else str(self.vector.dtype)
        self.q_levels = q_levels

        self.xor = None  # the result of the XOR operation between the vector and the matrix
        self.bits = None  # the result of converting the bits into a useful for (after XOR)
        self.distances = None  # the decimal distances
        self.hq_distances = None  # the quantized hamming distances (final result)

    def compute_xor(self):
        raise NotImplementedError()

    def convert_bits(self):
        raise NotImplementedError()

    def pop_count(self):
        raise NotImplementedError()

    def quantize_distance(self):
        raise NotImplementedError()

    def hamming(self):
        self.xor = self.compute_xor()

        self.bits = self.convert_bits()

        self.distances = self.pop_count()

        self.hq_distances = self.quantize()

        return self.distances  # returns the "raw" distance as it is easier to check
