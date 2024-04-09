from nerdle_solver.solver import AutoNerdlePlayer
from .combinations import COM_LIST, SOL_LIST
from .game import NerdleGame

print(f'Guesses: {len(COM_LIST)}')
print(f'Secrets: {len(SOL_LIST)}')

game = NerdleGame(AutoNerdlePlayer(debug=True))
game.start_game()
