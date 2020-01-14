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
        # Copmute XOR between the vector and the matrix (use "self.vector" & "self.matrix" to obtain the input data)
        raise NotImplementedError()
        return None

    def convert_bits(self):
        # Do a bit packing/unpacking of the XOR result (stored as self.xor)
        # Might not be needed in some cases
        raise NotImplementedError()
        return None

    def pop_count(self):
        # Count active bits in the XOR result (stored as self.bits)
        # result should be the Hamming distance
        raise NotImplementedError()
        return None

    def quantize_distance(self):
        # Threshold or quantize the Hamming distances for each row of the "self.distances" variable
        # quantization levels are specified in the self.q_levels but for now only q_levels = 2 is used
        raise NotImplementedError()
        return None

    def hamming(self):
        # The full Hamming distance. This function just calls the individual methods
        self.xor = self.compute_xor()

        self.bits = self.convert_bits()

        self.distances = self.pop_count()

        self.hq_distances = self.quantize()

        return self.distances  # returns the "raw" distance as it is easier to check
