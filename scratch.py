from struct import unpack
import sys

from nerdle_solver.entropy import expected_entropy, generate_entropies
from nerdle_solver.combinations import get_sol_list, get_comb_list
from nerdle_solver.convert import array_to_eq, eqs_to_array, unpack_array

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
# Best is 48-32=16
if __name__ == '__main__':
    print('Calculating Entropies...')
    if len(sys.argv) > 1 and sys.argv[1] == 'quiet':
        show_progress = False
    entropies = []
    guess_array = eqs_to_array(comb)
    secret_array = eqs_to_array(sols)
    entropies = generate_entropies(guess_array, secret_array, progress=True)
    entropies.sort(key=lambda tup: tup[1], reverse=True)

    print("Best 100 Starting Moves:")
    for guess, entropy in entropies[:100]:
        eq = array_to_eq(unpack_array(guess, ord=15))
        print(f'{eq}: {-entropy}')
