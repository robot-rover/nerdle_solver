import math
import os

import numpy as np
from nerdle_cuda.context import PythonCluePool
from estimator import ESTIMATOR_PATH
from nerdle_solver.clues import filter_secrets
from nerdle_solver.combinations import COM_ARRAY, COM_LIST, COM_PACKED, SOL_ARRAY, SOL_LIST_INDEXED
from nerdle_solver.convert import array_to_eq, eqs_to_array, pack_array, unpack_array
from nerdle_solver.entropy import generate_entropies

ESTIMATOR_PARAMS = None
if os.path.exists(ESTIMATOR_PATH):
    ESTIMATOR_PARAMS = np.load(ESTIMATOR_PATH)

class AutoNerdlePlayer:
    def __init__(self, pool=None, debug=False):
        self.pool = pool
        self.debug = debug
        self.history = []
        self.is_win = None
        self.possible_secrets = SOL_LIST_INDEXED
        self.estimator = []
        self.entropy = []

    def give_clue(self, guess, clue):
        if self.debug:
            print(f"Clue:  {clue}")
        self.history.append((guess, clue))
        old_len = len(self.possible_secrets)
        self.possible_secrets = filter_secrets(guess, clue, self.possible_secrets, tuples=True)
        new_len = len(self.possible_secrets)
        if old_len > 1:
            self.entropy.append((-math.log2(1/old_len), math.log2(1/new_len)-math.log2(1/old_len)))
        if self.debug:
            print(len(self.possible_secrets), "secrets left")
        if clue != 'gggggggg':
            self.estimator.append(len(self.possible_secrets))

    def get_guess(self, guesses_left):
        if len(self.history) == 0:
            guess = AutoNerdlePlayer.FIRST_GUESS
        else:
            if len(self.possible_secrets) == 1:
                return self.possible_secrets[0][0]
            secret_array = eqs_to_array((tup[0] for tup in self.possible_secrets), slots=8, quantity=len(self.possible_secrets))
            entropies = generate_entropies(COM_ARRAY, secret_array, pool=self.pool, guess_packed=COM_PACKED)
            # Addition Heuristic
            # for secret, index in self.possible_secrets:
            #     entropies[index] += 1/len(self.possible_secrets)
            # best_idx = np.argmax(entropies)

            # Estimator Heuristic
            current_uncertainty = -math.log2(1/len(self.possible_secrets))
            sol_indexes = np.array([index for eq, index in self.possible_secrets])
            p_sol = np.zeros_like(entropies)
            p_sol[sol_indexes] += 1/len(self.possible_secrets)
            x_coord = np.maximum(current_uncertainty - entropies, ESTIMATOR_PARAMS[2])
            e_score = p_sol * 1 + (1 - p_sol) * (ESTIMATOR_PARAMS[1] * np.log(x_coord) + ESTIMATOR_PARAMS[0])
            best_idx = np.argmin(e_score)

            guess = COM_LIST[best_idx]
            if self.debug:
                entropies_list = list(zip(COM_LIST, entropies))
                entropies_list.sort(key=lambda item: item[1])
                for name, entropy in entropies_list[:10]:
                    print(f'{name}: {entropy}')

        if self.debug:
            print(f"Guess: {guess}")

        return guess

    def bad_guess(self, bad_guess):
        raise RuntimeError(f'Bad Guess: {bad_guess}, (hist: {self.history})')

    def win(self, num_guesses):
        if self.debug:
            print(f'Win({num_guesses})')
            print(f'Estimator: {self.estimator}')
        self.is_win = True

    def lose(self, secret):
        if self.debug:
            print(f'Lose()')
        self.is_win = False

    @staticmethod
    def _get_first_guess():
        cache_path = f'first_guess.txt'
        if not os.path.exists(cache_path):
            print("Calculating first guess")
            guess_array = COM_ARRAY
            secret_array = SOL_ARRAY
            with PythonCluePool(((1000,secret_array.shape[0]),)) as pool:
                entropies = generate_entropies(guess_array, secret_array, progress=True, pool=pool)
            print(entropies[8:16])
            best_idx = np.argmax(entropies)
            best = COM_LIST[best_idx]
            with open(cache_path, 'w') as cache_file:
                print(best, file=cache_file)
            return best
        else:
            with open(cache_path, 'r') as cache_file:
                return cache_file.read().strip()

    FIRST_GUESS = _get_first_guess()

