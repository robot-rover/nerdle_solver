from multiprocessing import cpu_count, freeze_support
import sys
import numpy as np
import timeit

from nerdle_solver.entropy import expected_entropy
from nerdle_solver.combinations import get_sol_list, get_comb_list
from nerdle_solver.equation import array_to_eq, eq_to_array, eqs_to_array

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map

from nerdle_cuda import PythonClueContext

# Windows Sillyness
if __name__ == '__main__':
    freeze_support()

comb = get_comb_list(8)
sols = get_sol_list(8)
entropies = []
secret_array = eqs_to_array(sols)

def make_iter(guess_strs, clues, chunk_begin, num_secret):
    def do_iter(idx):
        entropy = expected_entropy(clues[idx-chunk_begin,:], num_secret)
        return (guess_strs[idx], entropy)
    return do_iter

# 7:19 Serial
# 5:24 w/ Cuda
# 0:48 w/ Threads and Cuda
if __name__ == '__main__':
    print('Calculating Entropies...')
    # entropies = process_map(do_iter, comb, chunksize=50, max_workers=8)
    show_progress = True
    if len(sys.argv) > 1 and sys.argv[1] == 'quiet':
        show_progress = False
    entropies = []
    guess_array = eqs_to_array(comb)
    secret_array = eqs_to_array(sols)
    num_guess = guess_array.shape[0]
    num_secret = secret_array.shape[0]
    batch_size = 2000
    clues = np.zeros((batch_size, num_secret), dtype=np.uint16)
    print("Opening CUDA Context")
    with PythonClueContext(batch_size, num_secret) as ctx:
        for chunk_begin in tqdm(range(0, num_guess, batch_size), disable=False):
            chunk_end = min(chunk_begin + batch_size, num_guess)
            ctx.generate_clue(guess_array[chunk_begin:chunk_end, :], secret_array, clues)
            tuples = thread_map(make_iter(comb, clues, chunk_begin, num_secret),
                range(chunk_begin, chunk_end), max_workers=cpu_count(), leave=False, disable=False)
            entropies.extend(tuples)

    entropies.sort(key=lambda tup: tup[1], reverse=True)

    print("Best 100 Starting Moves:")
    for guess, entropy in entropies[:100]:
        print(f'{guess}: {-entropy}')
