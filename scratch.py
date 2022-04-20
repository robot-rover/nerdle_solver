from distutils.ccompiler import show_compilers
from multiprocessing import cpu_count, freeze_support
import sys
import numpy as np

from nerdle_solver.entropy import expected_entropy
from nerdle_solver.combinations import get_sol_list, get_comb_list
from nerdle_solver.equation import eq_to_array, eqs_to_array

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map

from nerdle_cuda import PythonClueContext

# Windows Sillyness
if __name__ == '__main__':
    freeze_support()

comb = get_comb_list(8)
sols = get_sol_list(8)
entropies = []
secrets_array = eqs_to_array(sols)

def do_iter(guess):
    guess_array = eq_to_array(guess)
    entropy = expected_entropy(guess_array, secrets_array)
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
    secrets_array = eqs_to_array(sols)
    num_guess = len(comb)
    num_secrets = len(sols)
    batch_size = 5000
    clues = np.zeros((batch_size, batch_size, 8), dtype=np.uint8)
    with PythonClueContext(batch_size, batch_size) as ctx:
        for x_chunk in tqdm(range(0, num_secrets, batch_size), disable=False):
            x_lim = min(x_chunk + batch_size, num_secrets)
            for y_chunk in tqdm(range(0, num_guess, batch_size), leave=False, disable=False):
                y_lim = min(y_chunk + batch_size, num_guess)
                ctx.generate_clue(secrets_array[x_chunk:x_lim, :], guess_array[y_chunk:y_lim, :], clues)

    entropies.sort(key=lambda tup: tup[1], reverse=True)

    print("Best 100 Starting Moves:")
    for guess, entropy in entropies[:100]:
        print(f'{guess}: {-entropy}')
