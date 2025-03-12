from env import Game
import random
import os

game = Game()
game.print()

while(not game.end_game):
    a = int(input())
    game.step(a)
    os.system('cls' if os.name == 'nt' else 'clear')
    game.print()
    