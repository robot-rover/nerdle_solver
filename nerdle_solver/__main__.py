from nerdle_solver.solver import AutoNerdlePlayer
from .combinations import get_comb_list, get_sol_list
from .game import NerdleGame

game = NerdleGame(AutoNerdlePlayer(debug=True))
game.start_game()
