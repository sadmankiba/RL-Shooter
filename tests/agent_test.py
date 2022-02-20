from game import ShooterEnv, Game
from agent import Agent


class AgentTest:
    def test_train(self, game: Game):
        env = ShooterEnv(game)
        agent = Agent(env)
        history = agent.train(5)
        assert True 