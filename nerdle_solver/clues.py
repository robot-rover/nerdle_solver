import numpy as np
from termcolor import colored


CLUE_TYPES = 'gby'
GREEN = 0
BLACK = 1
YELLOW = 2


def generate_clue(secret, guess):
    assert len(secret) == len(guess)
    clue = bytearray(GREEN for _ in secret)
    not_green = []
    for idx, (s, g) in enumerate(zip(secret, guess)):
        if s != g:
            clue[idx] = BLACK
            not_green.append(secret[idx])
    for ng in not_green:
        indexes = [idx for idx, gc in enumerate(guess) if gc == ng and clue[idx] == BLACK]
        if len(indexes) > 0:
            clue[indexes[0]] = YELLOW
    return ''.join(CLUE_TYPES[idx] for idx in clue)

def print_color_guess(guess, clue):
    for idx, letter in enumerate(clue):
        if letter == 'g':
            print(colored(guess[idx], 'green'), end='')
        elif letter == 'y':
            print(colored(guess[idx], 'yellow'), end='')
        elif letter == 'b':
            print(colored(guess[idx], 'red'), end='')
        else:
            print(letter, end = '')
    print()


def filter_secrets(guess, clue, possible_secrets, tuples=False):
    remaining_solutions = []
    for possible_secret_item in possible_secrets:
        possible_secret = possible_secret_item[0] if tuples else possible_secret_item
        valid = True
        for index, element in enumerate(clue):
            #check greens
            if element == 'g':
                if possible_secret[index] == guess[index]:
                    continue
                else:
                    valid = False
                    break
            else:
                if possible_secret[index] != guess[index]:
                    continue
                else:
                    valid = False
                    break


        #check yellows
        if valid and generate_clue(possible_secret, guess) == clue:
            remaining_solutions.append(possible_secret_item)

    return remaining_solutions

def bincount2d(array, ord=None):
    """ np.bincounts vectorized """
    # This function makes a 2D array into a 1D array, and adds ord*row_num so that
    # the bins of each row don't overlap, then restores the shape at the end
    if ord is None:
        ord = array.max() + 1
    a_offs = array + np.arange(array.shape[0])[:,None]*ord
    return np.bincount(a_offs.ravel(), minlength=array.shape[0]*ord).reshape(-1,ord)

def generate_cluev(secrets, guess, char_ord=15):
    # secrets[batches, slots]
    assert len(guess) == secrets.shape[1]
    slots = secrets.shape[1]
    # Initialize to Green
    clues = np.zeros_like(secrets)
    not_green = (secrets != guess).astype(np.uint8)
    clues += BLACK * not_green

    # Use +1 and loss of first column to not count chars that are green
    counts = bincount2d((secrets+1) * not_green, ord=char_ord+1)[:,1:]

    for slot_idx in range(slots):
        char = guess[slot_idx]
        each_secrets_idx_count = counts[:, char]
        make_yellow = not_green[:,slot_idx] * (each_secrets_idx_count > 0)
        each_secrets_idx_count -= make_yellow
        clues[:,slot_idx] += (YELLOW - BLACK) * make_yellow

    return clues
