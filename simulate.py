from multiprocessing import cpu_count
from multiprocessing.dummy import freeze_support
import threading
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
from nerdle_cuda.context import PythonCluePool
from nerdle_solver.game import NerdleGame
from nerdle_solver import solver
from nerdle_solver.combinations import COM_LIST, SOL_LIST
import matplotlib.pyplot as plt

PLAYER = solver.AutoNerdlePlayer
BATCH_SIZE = 1000
SOLUTIONS = SOL_LIST
PARALLEL = True
NAME = 'estimator'
SHOW_BAR = True
SAVE_ESTIMATOR = False
SAVE_ENTROPY = False
DEBUG = False

estimator = []
entropy = []
_TLS = threading.local()
def get_pool():
    if not hasattr(_TLS, "pool"):
        _TLS.pool = PythonCluePool([
            (BATCH_SIZE, len(SOL_LIST)),
            (len(COM_LIST), 1000)
        ])
        _TLS.pool.open()
    return _TLS.pool

def do_iter(solution):
    game = NerdleGame(PLAYER(pool=get_pool(), debug=DEBUG), solution, debug=DEBUG)
    game.start_game()
    if SAVE_ESTIMATOR:
        for tup in enumerate(reversed(game.player.estimator),1):
            # req_guesses, num_remaining
            estimator.append(tup)

    if SAVE_ENTROPY:
        for tup in enumerate(game.player.entropy, 1):
            entropy.append(tup)

    return 6 - game.guess_remaining - 1

if __name__ == "__main__":
    freeze_support()

    # 0-5 is wins, 6 is num losses
    stats = np.zeros((7), dtype=np.int64)

    if PARALLEL:
        results = thread_map(do_iter, SOLUTIONS, max_workers=8, chunksize=1, disable=not SHOW_BAR)
    else:
        results = []
        to_iter = SOLUTIONS
        if SHOW_BAR:
            to_iter = tqdm(to_iter)
        for solution in to_iter:
            results.append(do_iter(solution))
    arr = np.array(results)
    np.save(f"{NAME}.npy", arr)
    bins = np.bincount(np.array(results))
    stats[:bins.shape[0]] = bins
    mean = np.sum(stats[:6] * np.arange(1, 7)) / np.sum(stats[:6])

    print(stats)
    print('-------')
    for i in range(6):
        print(f'Win({i+1}): {stats[i]:>5.0f}')
    print(f'Losses: {stats[6]:>5.0f}')
    print()
    print(f'Expectation: {mean}')

if SAVE_ESTIMATOR:
    with open('estimator.csv', 'w') as estimator_file:
        for req_guess, num_remain in estimator:
            print(f'{req_guess},{num_remain}', file=estimator_file)

if SAVE_ENTROPY:
    with open('entropy.csv', 'w') as entropy_file:
        for num_guess, (avail, gained) in entropy:
            print(f'{num_guess},{avail},{gained}', file=entropy_file)