from math import log2
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map
from nerdle_solver.clues import generate_cluev
from nerdle_solver.convert import array_to_clues, eqs_to_array, pack_array

def expected_entropy(clue_codes, num_secrets):
    clue_codes_unique, counts = np.unique(clue_codes, return_counts=True)

    entropy = 0
    probability = counts / num_secrets
    entropy = probability * np.log2(1/probability)
    return sum(entropy)

def expected_entropyv(clue_codes, guesses, num_secret):
    def do_iter(idx):
        entropy = expected_entropy(clue_codes[idx,:], num_secret)
        return (guesses[idx], entropy)
    with ThreadPool() as pool:
        return pool.map(do_iter, range(clue_codes.shape[0]))

def _generate_entropies_npy(guess_array, secret_array, batch_size, progress, in_dict):
    guess_packed = pack_array(guess_array, 15, dtype=np.uint64)
    num_secret = secret_array.shape[0]
    def do_iter(idx):
        clues = generate_cluev(secret_array, guess_array[idx,:])
        entropy = expected_entropy(clues, num_secret)
        return (guess_packed[idx], entropy)
    tuples = thread_map(do_iter, range(guess_array.shape[0]), max_workers=cpu_count(), disable=not progress)
    if in_dict:
        return dict(tuples)
    else:
        return list(tuples)

def _generate_entropies_gpu_clue(guess_array, secret_array, batch_size, progress, in_dict):
    from nerdle_cuda import PythonClueContext
    num_guess = guess_array.shape[0]
    guess_packed = pack_array(guess_array, 15, dtype=np.uint64)
    num_secret = secret_array.shape[0]
    clues = np.zeros((batch_size, num_secret), dtype=np.uint16)
    entropies = {} if in_dict else []
    with PythonClueContext(batch_size, num_secret) as ctx:
        for chunk_begin in tqdm(range(0, num_guess, batch_size), disable=not progress):
            chunk_end = min(chunk_begin + batch_size, num_guess)
            ctx.generate_clue(guess_array[chunk_begin:chunk_end, :], secret_array, clues)
            tuples = expected_entropyv(clues[:chunk_end-chunk_begin], guess_packed[chunk_begin:], num_secret)
            if in_dict:
                entropies.update(tuples)
            else:
                entropies.extend(tuples)
    return entropies

def _generate_entropies_gpu(guess_array, secret_array, in_dict, gpu_sorted, batch_size, progress):
    from nerdle_cuda import PythonClueContext
    num_guess = guess_array.shape[0]
    guess_packed = pack_array(guess_array, 15, dtype=np.uint64)
    num_secret = secret_array.shape[0]
    entropy_array = np.zeros((batch_size))
    entropies = {} if in_dict else []
    with PythonClueContext(batch_size, num_secret) as ctx:
        for chunk_begin in tqdm(range(0, num_guess, batch_size), disable=not progress):
            chunk_end = min(chunk_begin + batch_size, num_guess)

            ctx.generate_entropies(guess_array[chunk_begin:chunk_end,:], secret_array, entropy_array, use_sort_alg=gpu_sorted)
            generator = ((packed_eq, entropy) for packed_eq, entropy in zip(guess_packed[chunk_begin:chunk_end], entropy_array))

            if in_dict:
                entropies.update(generator)
            else:
                entropies.extend(generator)
    return entropies

def generate_entropies(guess_array, secret_array, batch_size=1000, progress=False, in_dict=False):
    try:
        from nerdle_cuda import PythonClueContext
        return _generate_entropies_gpu(guess_array, secret_array, in_dict, gpu_sorted=True, batch_size=batch_size, progress=progress)
    except ImportError:
        _generate_entropies_npy(guess_array, secret_array, batch_size, progress, in_dict)
