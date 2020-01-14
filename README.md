# python-bitwise_test
Performance testing of various Python packages on bitwise operations

* **gen_data.py** contains data generator
    * can be called from terminal
    * first two arguments are number of vectors and vector length, e.g.:
      ```python gen_data.py 1e3 2**16 -s byte```
      - generates a vector of length 65536 and a matrix of size 1000x65536
      - some Python expressions work (e.g. 2**16)
      - stores data into HDF5 format as numpy array of type "uint8" (~byte)
      - call ```python gen_data.py -h``` for more info
    * other parameters need to be changed in the script itself
* **hamming.py** contains an abstract class for the Hamming distance calculation
* **test_multiple.py** calls the desired testing scripts (specify in the dictionary at the begining of the script)
  * some parameters like testing data can be specified via the command line, e.g.:
    ```python test_multiple.py -nv 1e3 -ne 2**16```
    - will try to find data file(s) in the form "bits_1000x65536_{storage}_mutate.hdf5" and run tests on them
    - call ```python test_multiple.py -h``` for more info