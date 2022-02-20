import pytest

from game import Game

@pytest.fixture
def game():
    return Game()

