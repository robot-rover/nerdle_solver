from multiprocessing import cpu_count, freeze_support
import sys
import numpy as np
import timeit

from nerdle_solver.entropy import expected_entropy
from nerdle_solver.combinations import get_sol_list, get_comb_list
from nerdle_solver.equation import array_to_eq, eq_to_array, eqs_to_array

from tqdm import tqdm
# from tqdm.contrib.concurrent import thread_map, process_map

from nerdle_cuda import PythonClueContext

# Windows Sillyness
if __name__ == '__main__':
    freeze_support()

comb = get_comb_list(8)
sols = get_sol_list(8)
entropies = []
secret_array = eqs_to_array(sols)

def do_iter(guess):
    guess_array = eq_to_array(guess)
    entropy = expected_entropy(guess_array, secret_array)
    return (guess, entropy)

# 7:19 Serial
if __name__ == '__main__':
    print('Calculating Entropies...')
    # entropies = process_map(do_iter, comb, chunksize=50, max_workers=8)
    show_progress = True
    if len(sys.argv) > 1 and sys.argv[1] == 'quiet':
        show_progress = False
    entropies = []
    guess_array = eqs_to_array(comb)
    secret_array = eqs_to_array(comb)
    num_guess = guess_array.shape[0]
    num_secret = secret_array.shape[0]
    batch_size = 2000
    clues = np.zeros((batch_size, num_secret, 8), dtype=np.uint8)
    print("Opening CUDA Context")
    with PythonClueContext(batch_size, num_secret) as ctx:
        for chunk_begin in tqdm(range(0, num_guess, batch_size), disable=False):
            chunk_end = min(chunk_begin + batch_size, num_guess)
            ctx.generate_clue(guess_array[chunk_begin:chunk_end, :], secret_array, clues)
            for idx in tqdm(range(chunk_begin, chunk_end), leave=False, disable=False):
                guess_str = array_to_eq(guess_array[idx, :])
                entropy = expected_entropy(clues[idx-chunk_begin,:,:], num_secret)
                entropies.append((guess_str, entropy))

    entropies.sort(key=lambda tup: tup[1], reverse=True)

    print("Best 100 Starting Moves:")
    for guess, entropy in entropies[:100]:
        print(f'{guess}: {-entropy}')
