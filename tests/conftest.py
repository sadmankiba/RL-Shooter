import pytest

from game import Game
from main import TRAIN_FPS

@pytest.fixture
def game():
    return Game(TRAIN_FPS)

