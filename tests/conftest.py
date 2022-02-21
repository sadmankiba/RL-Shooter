import pytest

from game import Game
from constants import TRAIN_FPS

@pytest.fixture
def game():
    return Game(TRAIN_FPS, False, False)

