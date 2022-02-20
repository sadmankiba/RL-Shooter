import random 

from game import Game, Action, ShooterEnv

class TestGame:
    def test_run(self, game: Game):
        for _ in range(100):
            a = random.choice([Action.UP, Action.DOWN, Action.NOP])
            game.step(a)
            if not game.running:
                game.start()
        
        assert True

class TestEnv:
    def test_step(self, game: Game):
        env = ShooterEnv(game)
        s_nxt, rew, done = env.step(Action.UP)
        assert rew == 0
        assert done == False 
    


