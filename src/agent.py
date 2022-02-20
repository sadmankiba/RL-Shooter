from nptyping import NDArray
from constants import UP, DOWN

from game import ShooterEnv

EPSILON_INIT = 1.0
PLAY_STEPS = 10

class ConvModel:
    pass

class ReplayBuffer:
    pass

class Agent:
    def __init__(self, env: ShooterEnv):
        self._env = env
        self._model = ConvModel()
        self._rep = ReplayBuffer()
        self._epsilon = EPSILON_INIT

    def _play_and_record(steps: int):
        s = self._env.state
        pass

    def train(n: int):
        for _ in n:
            self._play_and_record(PLAY_STEPS)


    @classmethod
    def policy(cls, state: NDArray[60, 40]):
        return UP
