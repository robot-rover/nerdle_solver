CLUE_TYPES = 'gyb'
GREEN = 0
YELLOW = 1
BLACK = 2


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
