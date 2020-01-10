from timeit import repeat
from functools import partial

def run_stats(run_list, runs):
    normalized = np.array(run_list) / runs
    mean = np.mean(normalized)
    std = np.std(normalized)
    return f"avg = {mean} (std = {std})"

time_result = repeat(partial(compute_xor, vector, matrix), repeat=n_reps, number=n_runs)