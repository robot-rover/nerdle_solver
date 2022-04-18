from multiprocessing import cpu_count, freeze_support
import sys

from nerdle_solver.entropy import expected_entropy
from nerdle_solver.combinations import get_sol_list, get_comb_list
from nerdle_solver.equation import eq_to_array, eqs_to_array

from tqdm.contrib.concurrent import process_map

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
    entropies = process_map(do_iter, comb, chunksize=50, max_workers=cpu_count())
    entropies.sort(key=lambda tup: tup[1], reverse=True)

    print("Best 100 Starting Moves:")
    for guess, entropy in entropies[:100]:
        print(f'{guess}: {-entropy}')
