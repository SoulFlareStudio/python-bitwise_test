import pandas as pdimport numpy as np

implementations = {
    "numpy": {
        "script": "proc_numpy.py"
    },
    "bitarray": {
        "script": "proc_bitarray.py"
    }
}

column_names = ["XOR", "packing/unpacking", "bit count", "Hamming distance"]
datatypes = ["bool", "byte", "uint32"]

df = pandas.DataFrame(columns=column_names, index=pd.MultiIndex.from_product((implementations.keys(), datatypes)))
