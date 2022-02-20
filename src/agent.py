import random

from tensorflow.keras.models import Sequential
from nptyping import NDArray

from constants import UP, DOWN
from game import ShooterEnv

EPSILON_INIT = 1.0
EPSILON_DEC_SCALE = 0.99
PLAY_STEPS = 10
REPLAY_SAMPLE_TRAIN_SIZE = 64
ITER_UPDATE_TARGET_MODEL = 100
ITER_DEC_EPSILON = 100
ITER_UPDATE_HISTORY = 200

def convmodel():
    m = Sequential()
    m.add(Dense(2, input_dim=3, activation="softmax"))
    m.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return m

class ReplayBuffer:
    pass

class Agent:
    def __init__(self, env: ShooterEnv):
        self._env = env
        self._model = convmodel()
        self._tartget_model = convmodel()
        self._rep = ReplayBuffer()
        self._epsilon = EPSILON_INIT
        self.GAMMA = 0.99

    def train(self, n: int):
        history = {
            "mean_rew": [],
            "mean_loss": []
        }
        total_loss = 0
        for i in range(n):
            self._play_and_record(PLAY_STEPS)
            batch = self._rep.sample(REPLAY_SAMPLE_TRAIN_SIZE)
            q_nxt_batch = self._target_model.predict(batch.s_nxt)
            v_nxt_batch = np.max(q_nxt_batch, axis=1)
            q_ref_batch = batch.rew + self.GAMMA * v_nxt_batch * (1 - batch.done)
            q_current = self._model.predict(batch.s)[batch.actions]
            loss = 1 / len(batch.s) * np.sum(q_ref_batch - q_current)**2
            total_loss += loss
            self._model.update(loss)

            if i % ITER_UPDATE_TARGET_MODEL == 0:
                self._target_model.weights = self._model.weights

            if i % ITER_DEC_EPSILON == 0:
                self._epsilon *= EPSILON_DEC_SCALE

            if i % ITER_UPDATE_HISTORY == 0:
                history["mean_rew"].append(self._evaluate(5))
                history["mean_loss"].append(total_loss / ITER_UPDATE_HISTORY)
                total_loss = 0

        return history

    def _play_and_record(self, steps: int):
        s = self._env.state
        for _ in range(steps):
            a = self._choose_action(s)
            s_nxt, rew, done = self._env.step(a)
            self._rep.add(s, a, rew, s_nxt, done)
            s = s_nxt

    def _choose_action(self, s: state):
        ba = self._choose_best_action(s)
        ra = random.choice(self._env.actions)
        return random.sample([ba, ra], p=[1 - epsilon, epsilon])

    def _choose_best_action(self, s: state):
        q = self._model.predict(s)
        return q.argmax()

    def _evaluate(self, n_eps: int) -> float:
        env = self._env.copy()
        s = env.state
        rew_all = []
        for _ in range(n_eps):
            done = False
            rew_eps = 0
            while not done:
                a = self._choose_best_action(s)
                s_nxt, rew, done = self._env.step(a)
                rew_eps += rew

            rew_all.append(rew_eps)

        return sum(rew_all) / n_eps
