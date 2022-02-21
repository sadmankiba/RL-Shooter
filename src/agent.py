import random
import inspect
import collections
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from nptyping import NDArray

from constants import UP, DOWN
from game import ShooterEnv, T_STATE, T_Action, STATE_IMG_H, STATE_IMG_W, Action, log
from util import FileSave, parent_dir

EPSILON_INIT = 1.0
EPSILON_DEC_SCALE = 0.99
PLAY_STEPS = 32
REPLAY_SAMPLE_TRAIN_SIZE = 32
ITER_UPDATE_TARGET_MODEL = 100
ITER_DEC_EPSILON = 100
ITER_UPDATE_HISTORY = 200
ITER_SAVE_IMG = 200
ITER_SAVE_WEIGHTS = 200
REPLAY_BUFFER_SIZE = 6400
LEARNING_RATE = 0.001


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
            "s": np.array(_get_stored(0)),
            "a": np.array(_get_stored(1)),
            "rew": np.array(_get_stored(2)),
            "s_nxt": np.array(_get_stored(3)),
            "done": np.array(_get_stored(4)),
        }


class Agent:
    def __init__(self, env: ShooterEnv):
        self._env = env
        self._model = self._convmodel(LEARNING_RATE)
        self._target_model = self._convmodel(LEARNING_RATE)
        self._rep = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self._epsilon = EPSILON_INIT
        self.GAMMA = 0.99
        self._model.summary()

    def _convmodel(self, lr:float):
        m = Sequential()
        m.add(
            Conv2D(
                8,
                (6, 6),
                strides=2,
                padding="same",
                activation="relu",
                kernel_initializer="he_uniform",
                input_shape=(STATE_IMG_H, STATE_IMG_W, 1),
            )
        )
        m.add(MaxPool2D((2, 2), strides=2))
        m.add(
            Conv2D(
                16,
                (5, 5),
                activation="relu",
                padding="same",
                kernel_initializer="he_uniform",
            )
        )
        m.add(MaxPool2D((2, 2)))
        m.add(Flatten())
        m.add(Dense(32, activation="relu", kernel_initializer="he_uniform"))
        m.add(Dense(12, activation="relu", kernel_initializer="he_uniform")) 
        m.add(Dense(len(self._env.actions), activation=None))
        m.compile(Adam(lr), loss=tf.keras.losses.Huber())
        return m

    def train(self, n: int):
        history = {"mean_rew": [], "mean_loss": []}
        total_loss = 0
        for i in range(n):
            self._play_and_record(PLAY_STEPS)
            loss = self._update_model_from_batch()
            total_loss += loss

            if i != 0 and i % ITER_UPDATE_TARGET_MODEL == 0:
                self._target_model.set_weights(self._model.get_weights())

            if i != 0 and i % ITER_DEC_EPSILON == 0:
                self._epsilon *= EPSILON_DEC_SCALE

            if i != 0 and i % ITER_UPDATE_HISTORY == 0:
                history["mean_rew"].append(self._evaluate(5))
                history["mean_loss"].append(total_loss / ITER_UPDATE_HISTORY)
                log.info(
                    f"iter: {i}/{n}, games played: {self._env.games_played}"
                    f", mean reward: {history['mean_rew'][-1]}, loss: {history['mean_loss'][-1]}"
                )
                total_loss = 0
                self._plot_metrics(history, i)

            if i != 0 and i % ITER_SAVE_IMG == 0:
                batch = self._rep.sample(10)
                for s, a, rew in zip(batch["s"], batch["a"], batch["rew"]):
                    FileSave.fig_state(s, f"act_{Action.rev(a)}"
                    f"_q_{self._model.predict(self._prep_state_img(s)).tolist()}_iter_{i}")

            if i != 0 and i % ITER_SAVE_WEIGHTS == 0:
                self._target_model.save(
                    f"{parent_dir(inspect.currentframe()).parent}"
                    f"/saved_weights/target_model_{i}_{datetime.now().strftime('%H_%M')}.h5"
                )

            log.info(f"[TRAIN]: iter {i}")

        return history
    
    def _plot_metrics(self, history: dict, i: int):
        plt.subplot(1, 2, 1)
        plt.title("Mean reward per life")
        plt.plot(history["mean_rew"])
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title("TD loss history")
        plt.plot(history["mean_loss"])
        plt.grid()
        FileSave.fig_metrics(f"iter_{i}")

    def play_from_saved(self, i: int):
        self._model = tf.keras.models.load_model(
            f"{parent_dir(inspect.currentframe()).parent}"
            f"/saved_weights/target_model_{i}.h5"
        )
        
        s = self._env.state
        while True:
            a = self._choose_best_action(s)
            s_nxt, _, _ = self._env.step(a)
            s = s_nxt

    def _update_model_from_batch(self):
        batch = self._rep.sample(REPLAY_SAMPLE_TRAIN_SIZE)
        q_nxt_batch = self._target_model.predict(self._prep_state_img(batch["s_nxt"]))

        v_nxt_batch = np.max(q_nxt_batch, axis=1)
        q_ref_batch = batch["rew"] + self.GAMMA * v_nxt_batch * (1 - batch["done"])

        inp_batch = self._prep_state_img(batch["s"])
        a_batch = batch["a"]
        with tf.GradientTape() as tape:
            q_current = self._model(inp_batch)
            one_hot_actions = to_categorical(
                a_batch, len(self._env.actions), dtype=np.float32
            )
            q_a = tf.reduce_sum(tf.multiply(q_current, one_hot_actions), axis=1)
            loss = Huber()(q_ref_batch, q_a)

        model_gradients = tape.gradient(loss, self._model.trainable_variables)
        self._model.optimizer.apply_gradients(
            zip(model_gradients, self._model.trainable_variables)
        )
        return loss

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
        return random.choices(
            [ba, ra], weights=[1 - self._epsilon, self._epsilon], k=1
        )[0]

    def _choose_best_action(self, s: T_STATE) -> T_Action:
        q = self._model.predict(self._prep_state_img(s))[0]
        return np.argmax(q)

    def _evaluate(self, n_eps: int) -> float:
        s = self._env.state
        rew_all = []
        for _ in range(n_eps):
            done = False
            rew_eps = 0
            while not done:
                a = self._choose_best_action(s)
                s_nxt, rew, done = self._env.step(a)
                rew_eps += rew
                s = s_nxt

            rew_all.append(rew_eps)

        return sum(rew_all) / n_eps

    def _prep_state_img(self, s: NDArray) -> NDArray:
        s = s[..., np.newaxis]
        if len(s.shape) == 3:
            s = s[np.newaxis, ...]

        return s / 255
