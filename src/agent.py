import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.losses import Huber
from nptyping import NDArray

from constants import UP, DOWN
from game import ShooterEnv, T_STATE, T_Action

EPSILON_INIT = 1.0
EPSILON_DEC_SCALE = 0.99
PLAY_STEPS = 10
REPLAY_SAMPLE_TRAIN_SIZE = 64
ITER_UPDATE_TARGET_MODEL = 100
ITER_DEC_EPSILON = 100
ITER_UPDATE_HISTORY = 200


class ReplayBuffer:
    def __init__(self, size: int):
        self._storage = collections.deque([], size)
        self._maxsize = size

    def __len__(self):
        return len(self._storage)

    def add(self, s, a, rew, s_nxt, done):
        self._storage.appendleft((s, a, rew, s_nxt, done))

    def sample(self, batch_size: int) -> dict[str, NDArray]:
        idxes = random.choices(range(len(self._storage)), k=batch_size)

        batch = np.array(self._storage)[idxes]
        def _get_stored(pos) -> list:
            return list(data[pos] for data in batch)

        return {
            "s": np.array( _get_stored(0) ),
            "a": np.array( _get_stored(1) ),
            "rew": np.array( _get_stored(2) ),
            "s_nxt": np.array( _get_stored(3) ),
            "done": np.array( _get_stored(4) ),
        }

class Agent:
    def __init__(self, env: ShooterEnv):
        self._env = env
        self._model = self._convmodel()
        self._tartget_model = self._convmodel()
        self._rep = ReplayBuffer()
        self._epsilon = EPSILON_INIT
        self.GAMMA = 0.99

    def _convmodel(lr: float=0.001):
        m = Sequential()
        m.add(Conv2D(5, (6, 6), activation="relu", kernel_initializer="he_uniform", input_shape=(38, 38, 1)))
        m.add(MaxPool2D((2, 2), strides=2))
        m.add(Conv2D(10, (4, 4), activation="relu", kernel_initializer="he_uniform"))
        m.add(MaxPool2D((2, 2)))
        m.add(Flatten())
        m.add(Dense(2, activation=None))
        m.compile(Adam(lr), loss=tf.keras.losses.Huber())
        return m

    def train(self, n: int):
        history = {
            "mean_rew": [],
            "mean_loss": []
        }
        total_loss = 0
        for i in range(n):
            self._play_and_record(PLAY_STEPS)
            batch = self._rep.sample(REPLAY_SAMPLE_TRAIN_SIZE)
            q_nxt_batch = np.squeeze(self._target_model.predict(self._prep_state_img(batch["s_nxt"])))
            v_nxt_batch = np.max(q_nxt_batch, axis=1)
            q_ref_batch = batch["rew"] + self.GAMMA * v_nxt_batch * (1 - batch["done"])

            total_loss += loss
            with tf.GradientTape() as tape:
                q_current = np.squeeze(self._model.predict(self._prep_state_img(batch["s"])))[:, batch["actions"]]
                assert q_ref_batch.shape == q_current.shape
                loss = Huber()(q_ref_batch, q_current)

            model_gradients = tape.gradient(loss, self._model.trainable_variables)
            self._model.optimizer.apply_gradients(zip(model_gradients, self._model.trainable_variables))

            if i % ITER_UPDATE_TARGET_MODEL == 0:
                self._target_model.set_weights(self._model.get_weights())

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

    def _choose_action(self, s: T_STATE) -> T_Action:
        ba = self._choose_best_action(s)
        ra = random.choice(self._env.actions)
        return random.choices([ba, ra], weights=[1 - epsilon, epsilon], k=1)[0]

    def _choose_best_action(self, s: T_STATE) -> T_Action:
        q = self._model.predict(self._prep_state_img(s))
        return np.argmax(q.flatten())

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

    def _prep_state_img(s: NDArray) -> NDArray:
        s = s[..., np.newaxis]
        return s / 255
