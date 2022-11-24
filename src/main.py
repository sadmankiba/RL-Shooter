from enum import Enum, auto

from agent import Agent
from game import Game, ShooterEnv
from constants import TRAIN_FPS, PLAY_FPS

TRAIN_ITER = 5000

class Run:
    TRAIN = auto()    
    PLAG = auto()
    PLU = auto()


if __name__ == "__main__":
    run = Run.PLAG

    if run == Run.TRAIN:
        game = Game(TRAIN_FPS, False, False)
        env = ShooterEnv(game)
        agent = Agent(env)
        history = agent.train(TRAIN_ITER)
        print(history)
    elif run == Run.PLAG:
        game = Game(TRAIN_FPS, True, True)
        env = ShooterEnv(game)
        agent = Agent(env)
        agent.play_from_saved(f"1300_11_33")
    else:
        game = Game(PLAY_FPS, True, True)
        while True:
            game.run()
            game.start()
