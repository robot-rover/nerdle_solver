from random import choice
from nerdle_solver.combinations import get_sol_list, get_comb_list
from nerdle_solver.equation import parse, validate
from nerdle_solver.clues import generate_clue, print_color_guess

class NerdleGame:
    def __init__(self, player=None, secret=None, num_guess=6):
        self.secret = secret if secret is not None else choice(get_sol_list(8))
        self.player = player if player is not None else TerminalNerdlePlayer()
        self.num_guess = num_guess

    def start_game(self):
        guesses_remaining = self.num_guess
        print("Secret is",self.secret)
        while True:
            guesses_remaining = guesses_remaining -1
            while True:
                guess = self.player.get_guess(guesses_remaining)
                if validate(parse(guess)):
                    break
                self.player.bad_guess()
            clue = generate_clue(self.secret, guess)
            self.player.give_clue(guess, clue)

            if guess == str(self.secret):
                self.player.win(6-guesses_remaining)
                break
            elif guesses_remaining == 0:
                self.player.lose(self.secret)
                break


class TerminalNerdlePlayer:
    def __init__(self):
        pass

    def give_clue(self, guess, clue):
        print_color_guess(guess, clue)
        print()

    def get_guess(self, guesses_remaining):
        while True:
            print("Enter a guess:")
            guess = input()
            print("\nYou guessed: " + str(guess))
            print("You have " + str(guesses_remaining) + " guesses remaining")
            return guess

    def bad_guess(self):
        print("Invalid input, try again!\n")

    def win(self, numGuesses):
        print("You Won! It took you " + str(numGuesses) +" guesses")

    def lose(self, secret):
        print("You Lost! The equation was: " + str(secret))
        print("Dumbass")
