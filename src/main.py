from agent import Agent
from game import Game, ShooterEnv
from constants import TRAIN_FPS, PLAY_FPS

TRAIN_ITER = 5000



if __name__ == "__main__":
    train = False
    if train:
        game = Game(TRAIN_FPS, False)
        env = ShooterEnv(game)
        agent = Agent(env)
        history = agent.train(TRAIN_ITER)
        print(history)
    else:
        game = Game(PLAY_FPS, True)
        while True:
            game.run()
            game.start()
