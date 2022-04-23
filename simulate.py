from faulthandler import disable
from multiprocessing import cpu_count
from multiprocessing.dummy import freeze_support
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from nerdle_solver.game import NerdleGame
from nerdle_solver import solver
from nerdle_solver.combinations import get_sol_list
import matplotlib.pyplot as plt

PLAYER = solver.AutoNerdlePlayer
SOLUTIONS = get_sol_list()
PARALLEL = True
NAME = 'baseline'
SHOW_BAR = True

def do_iter(solution):
    game = NerdleGame(PLAYER(), solution)
    game.start_game()
    return 6 - game.guess_remaining - 1

if __name__ == "__main__":
    freeze_support()

    # 0-5 is wins, 6 is num losses
    stats = np.zeros((7), dtype=np.int64)

    if PARALLEL:
        results = process_map(do_iter, SOLUTIONS, max_workers=cpu_count(), chunksize=1, disable=not SHOW_BAR)
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


    print(stats)
    print('-------')
    for i in range(6):
        print(f'Win({i+1}): {stats[i]:>5.0f}')
    print(f'Losses: {stats[6]:>5.0f}')

    plt.hist(ENTROPY_STATS, bins=20)
    plt.show()