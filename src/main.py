import pygame

from agent import Agent
from game import Game


if __name__ == "__main__":
    game = Game()
    train = False
    if train:
        env = ShooterEnv(game)
        agent = Agent(env)
        history = agent.train()
        print(history)
    else:

        game.run()
