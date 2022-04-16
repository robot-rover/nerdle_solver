import sys

from nerdle_solver.entropy import expected_entropy
from nerdle_solver.combinations import get_sol_list, get_comb_list
from nerdle_solver.equation import eq_to_array, eqs_to_array

from tqdm import tqdm

comb = get_comb_list(8)
sols = get_sol_list(8)
entropies = []
secrets_array = eqs_to_array(sols)

# Disable tqdm for profiling
if len(sys.argv) > 1 and sys.argv[1] == 'profile':
    comb_iter = iter(comb)
else:
    comb_iter = tqdm(comb)

print('Calculating Entropies...')
for guess in comb_iter:
    guess_array = eq_to_array(guess)
    entropy = expected_entropy(guess_array, secrets_array)
    entropies.append((guess, entropy))
entropies.sort(key=lambda tup: tup[1], reverse=True)

print("Best 100 Starting Moves:")
for guess, entropy in entropies[:100]:
    print(f'{guess}: {-entropy}')