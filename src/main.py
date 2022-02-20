from agent import Agent
from game import Game, ShooterEnv

TRAIN_ITER = 5000
TRAIN_FPS = 500
PLAY_FPS = 60

if __name__ == "__main__":
    
    train = False
    if train:
        game = Game(TRAIN_FPS)
        env = ShooterEnv(game)
        agent = Agent(env)
        history = agent.train(TRAIN_ITER)
        print(history)
    else:
        game = Game(PLAY_FPS)
        while True:
            game.run()
            game.start()
