from math import log2
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map
from nerdle_cuda.context import PythonClueContext
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

def _generate_entropies_npy(guess_array, secret_array, *args, progress, in_dict):
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

def _generate_entropies_gpu_clue(guess_array, secret_array, *args, batch_size, progress, in_dict):
    # TODO: Should return np array if in_dict is False
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

def _generate_entropies_gpu(guess_array, secret_array, *args, batch_size, in_dict, gpu_sorted, progress, pool, guess_packed):
    num_guess = guess_array.shape[0]
    if guess_packed is None:
        guess_packed = pack_array(guess_array, 15, dtype=np.uint64)
    num_secret = secret_array.shape[0]
    entropy_array = np.zeros((batch_size))
    entropies = {} if in_dict else np.zeros((num_guess))
    with PythonClueContext(batch_size, num_secret, pool=pool) as ctx:
        for chunk_begin in tqdm(range(0, num_guess, batch_size), disable=not progress):
            chunk_end = min(chunk_begin + batch_size, num_guess)

            ctx.generate_entropies(guess_array[chunk_begin:chunk_end,:], secret_array, entropy_array, use_sort_alg=gpu_sorted)

            if in_dict:
                generator = ((packed_eq, entropy) for packed_eq, entropy in zip(guess_packed[chunk_begin:chunk_end], entropy_array))
                entropies.update(generator)
            else:
                entropies[chunk_begin:chunk_end] = entropy_array
    return entropies

def generate_entropies(guess_array, secret_array, *args, batch_size=1000, gpu_sorted=True, progress=False, in_dict=False, pool=None, guess_packed=None):
    try:
        from nerdle_cuda import PythonClueContext
        if secret_array.shape[0] > 1000:
            print("WARN: Slow Path")
            return _generate_entropies_gpu(guess_array, secret_array, in_dict=in_dict, gpu_sorted=gpu_sorted, batch_size=batch_size, progress=progress, pool=pool, guess_packed=guess_packed)
        else:
            return _generate_entropies_gpu(guess_array, secret_array, in_dict=in_dict, gpu_sorted=gpu_sorted, batch_size=guess_array.shape[0], progress=progress, pool=pool, guess_packed=guess_packed)
    except ImportError:
        _generate_entropies_npy(guess_array, secret_array, batch_size=batch_size, progress=progress, in_dict=in_dict)
