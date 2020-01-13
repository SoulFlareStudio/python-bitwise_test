from timeit import repeat
from functools import partial
import numpy as np

def run_stats(run_list, runs):
    normalized = np.array(run_list) / runs
    mean = np.mean(normalized)
    std = np.std(normalized)
    # return f"avg = {mean} (std = {std})"
    return mean, std

def time_method(method, *args, n_reps=5, n_runs=1000):
    time_result = repeat(partial(method, *args), repeat=n_reps, number=n_runs)
    return run_stats(time_result, n_runs)
