from agent import Agent
from game import Game, ShooterEnv

TRAIN_ITER = 5000

if __name__ == "__main__":
    game = Game()
    train = True
    if train:
        env = ShooterEnv(game)
        agent = Agent(env)
        history = agent.train(TRAIN_ITER)
        print(history)
    else:
        game.run()
