from random import choice
from nerdle_solver.combinations import get_sol_list, get_comb_list
from nerdle_solver.equation import parse, validate
from nerdle_solver.clues import generate_clue, print_color_guess

class NerdleGame:
    def __init__(self):
        pass

    def start_game(self, secret = None):
        player = TerminalNerdlePlayer()
        if secret is None:
            #generate a random secret
            secret = choice(get_sol_list(8))
            print("Secret is: " + str(secret) + "\n")
        guesses_remaining = 6
        while True:
            guesses_remaining = guesses_remaining -1
            guess = player.get_guess(guesses_remaining)
            player.give_clue(guess, generate_clue(secret, guess))
            if guess == str(secret):
                player.win(6-guesses_remaining)
                break
            elif guesses_remaining == 0:
                player.lose(secret)
                break


class NerdlePlayer:
    def __init__(self):
        pass

    def give_clue(self, clue):
        pass

    def get_guess(self):
        pass

    def bad_guess(self, bad_guess):
        pass

    def win(self):
        pass
    
    def lose(self):
        pass


class TerminalNerdlePlayer(NerdlePlayer):
    def __init__(self):
        pass

    def give_clue(self, guess, clue):
        print_color_guess(guess, clue)
        print()

    def get_guess(self, guesses_remaining):
        while True:
            print("Enter a guess:")
            guess = input()
            if validate(parse(guess)):
                print("\nYou guessed: " + str(guess))
                print("You have " + str(guesses_remaining) + " guesses remaining")
                return guess
            else:
                print("Invalid input, try again!\n")

    def win(self, numGuesses):
        print("You Won! It took you " + str(numGuesses) +" guesses")
    
    def lose(self, secret):
        print("You Lost! The equation was: " + str(secret))
        print("Dumbass")