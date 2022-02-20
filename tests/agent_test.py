import pytest
import numpy as np

from game import ShooterEnv, Game, Action, STATE_IMG_H, STATE_IMG_W
from agent import Agent

@pytest.fixture
def agent(game):
    env = ShooterEnv(game)
    return Agent(env)
    

class TestAgent:
    def test_train(self, agent: Agent):
        history = agent.train(5)
        assert history["mean_rew"] == []
    
    def test_action_choice(self, agent: Agent):
        a = agent._choose_action(agent._env.state)
        assert a in list(agent._env.actions)

    def test_model(self, agent: Agent):
        r = np.random.normal(size=(STATE_IMG_H, STATE_IMG_W, 1))
        assert len(r.shape) == 3
        q = agent._model.predict(r[np.newaxis, ...])[0]
        assert q.shape == (len(agent._env.actions), )