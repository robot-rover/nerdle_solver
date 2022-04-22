from nerdle_solver.solver import AutoNerdlePlayer
from .combinations import get_comb_list, get_sol_list
from .game import NerdleGame

game = NerdleGame(AutoNerdlePlayer(),'27-4*6=3')
game.start_game()
