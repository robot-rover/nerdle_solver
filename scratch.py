from struct import unpack
import sys
from nerdle_cuda.context import PythonCluePool
from nerdle_solver.combinations import COM_ARRAY, SOL_ARRAY

from nerdle_solver.entropy import expected_entropy, generate_entropies
from nerdle_solver.convert import array_to_eq, eqs_to_array, unpack_array
from simulate import BATCH_SIZE

BATCH_SIZE = 1000

entropies = []

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
    guess_array = COM_ARRAY
    secret_array = SOL_ARRAY
    with PythonCluePool(((BATCH_SIZE,secret_array.shape[0]),)) as pool:
        entropies = generate_entropies(guess_array, secret_array, batch_size=BATCH_SIZE, progress=True, pool=pool)
    entropies.sort(key=lambda tup: tup[1], reverse=True)

    print("Best 100 Starting Moves:")
    for guess, entropy in entropies[:100]:
        eq = array_to_eq(unpack_array(guess, ord=15))
        print(f'{eq}: {-entropy}')
