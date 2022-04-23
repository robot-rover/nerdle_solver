import os

import numpy as np
from nerdle_solver.clues import filter_secrets
from nerdle_solver.combinations import get_comb_array, get_sol_array, get_sol_list
from nerdle_solver.convert import array_to_eq, eqs_to_array, pack_array, unpack_array
from nerdle_solver.entropy import generate_entropies



class AutoNerdlePlayer:
    def __init__(self, debug=False):
        self.debug = debug
        self.history = []
        self.is_win = None
        self.possible_secrets = get_sol_list()
        self.guess_array = get_comb_array()

    def give_clue(self, guess, clue):
        if self.debug:
            print(f"Clue:  {clue}")
        self.history.append((guess, clue))
        self.possible_secrets = filter_secrets(guess, clue, self.possible_secrets)
        if self.debug:
            print(len(self.possible_secrets), "secrets left")

    def get_guess(self, guesses_left):
        if len(self.history) == 0:
            guess = AutoNerdlePlayer.get_first_guess()
        else:
            secret_array = eqs_to_array(self.possible_secrets)
            entropies = generate_entropies(self.guess_array, secret_array, in_dict=True)
            for secret in pack_array(secret_array, dtype=np.uint64, ord=15):
                entropies[secret] += 1/len(self.possible_secrets)
            best_idx = max(entropies.items(), key=lambda tup: tup[1])[0]
            guess = array_to_eq(unpack_array(np.array(best_idx, dtype=np.uint64),ord=15))
            entropies_list = list(entropies.items())
            entropies_list.sort(key=lambda item: item[1], reverse=True)
            if self.debug:
                for name_packed, entropy in entropies_list[:10]:
                    eq_str = array_to_eq(unpack_array(np.array(name_packed, dtype=np.uint64),ord=15))
                    print(f'{eq_str}: {-entropy}')

        if self.debug:
            print(f"Guess: {guess}")

        return guess

    def bad_guess(self, bad_guess):
        raise RuntimeError(f'Bad Guess: {bad_guess}, (hist: {self.history})')

    def win(self, num_guesses):
        if self.debug:
            print(f'Win({num_guesses})')
        self.is_win = True

    def lose(self, secret):
        if self.debug:
            print(f'Lose()')
        self.is_win = False

    @staticmethod
    def get_first_guess():
        cache_path = f'first_guess.txt'
        if not os.path.exists(cache_path):
            guess_array = get_comb_array()
            secret_array = get_sol_array()
            entropies = generate_entropies(guess_array, secret_array)
            best_packed = max(entropies, key=lambda tup: tup[1])[0]
            best = array_to_eq(unpack_array(np.array(best_packed, dtype=np.uint64), ord=15))
            with open(cache_path, 'w') as cache_file:
                print(best, file=cache_file)
            return best
        else:
            with open(cache_path, 'r') as cache_file:
                return cache_file.read().strip()

