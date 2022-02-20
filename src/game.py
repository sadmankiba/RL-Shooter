import random
from typing import Any, Iterable

import pygame
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
import numpy as np
import cv2
from nptyping import NDArray

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCORE_HEIGHT = 80
FPS = 60
ADDENEMY = pygame.USEREVENT + 1

ENEMY_ADD_TIMER = 2000

PLAYER_SIZE = (75, 25)
ENEMY_SIZE = (20, 10)

BLACK = (0, 0, 0)
YELLOW = (255, 255, 102)

STATE_IMG_H = 38
STATE_IMG_W = 38

T_STATE = NDArray[(STATE_IMG_H, STATE_IMG_W), int]
T_Action = int

class Action:
    UP = 0
    DOWN = 1
    NOP = 2

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.surf = pygame.Surface(PLAYER_SIZE)
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()

    def update(self, a: T_Action) -> None:
        if a == Action.UP:
            self.rect.move_ip(0, -5)
        elif a == Action.DOWN:
            self.rect.move_ip(0, 5)

        if self.rect.top <= SCORE_HEIGHT:
            self.rect.top = SCORE_HEIGHT
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super(Enemy, self).__init__()
        self.surf = pygame.Surface(ENEMY_SIZE)
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect(
            center=(
                random.randint(SCREEN_WIDTH + 20, SCREEN_WIDTH + 100),
                random.randint(SCORE_HEIGHT + 20, SCREEN_HEIGHT),
            )
        )
        self.speed = random.random() + 0.5

    def update(self):
        self.rect.move_ip(-self.speed, 0)


class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.time.set_timer(ADDENEMY, ENEMY_ADD_TIMER)
        self._score_font = pygame.font.SysFont("comicsansms", 20)
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self._clock = pygame.time.Clock()
        self.start()

    def start(self):
        self.running = True
        self._player = Player()
        self._enemies = pygame.sprite.Group()
        self._all_sprites = pygame.sprite.Group()
        self._all_sprites.add(self._player)
        self.score = 0

    def run(self):
        while self.running:
            pressed_keys = pygame.key.get_pressed()
            if pressed_keys[K_UP]:
                a = Action.UP
            elif pressed_keys[K_DOWN]:
                a = Action.DOWN
            else:
                a = Action.NOP

            self.step(a)

    def step(self, a: T_Action):
        self._check_event()
        self._player.update(a)
        self._enemies.update()

        for enemy in self._enemies:
            if enemy.rect.right < 0:
                self.running = False

            if pygame.sprite.collide_rect(enemy, self._player):
                self.score += 1
                enemy.kill()

        self.render()

    def render(self):
        self._screen.fill(BLACK)

        for entity in self._all_sprites:
            self._screen.blit(entity.surf, entity.rect)

        self._render_score()

        pygame.display.flip()
        self._clock.tick(FPS)

    def _check_event(self):
        for event in pygame.event.get():
            if event.key == K_ESCAPE and event.type == KEYDOWN:
                self.running = False

            elif event.type == QUIT:
                self.running = False

            elif event.type == ADDENEMY:
                new_enemy = Enemy()
                self._enemies.add(new_enemy)
                self._all_sprites.add(new_enemy)

    def _render_score(self):
        value = self._score_font.render("Your Score: " + str(self.score), True, YELLOW)
        self._screen.blit(value, [0, 0])


class ShooterEnv:
    def __init__(self, game: Game):
        self._game = game
        self._game.render()
        self.state = self._scr_proc()
        self.actions: Iterable[T_Action] = range(len(Action))
        self._MAX_STEPS = 100000
        self._n_steps = 0

    def _scr_proc() -> T_STATE:
        scr = pygame.surfarray.array3d(pygame.display.get_surface())
        scr = cv2.cvtColor(cv2.resize(scr, (STATE_IMG_H, STATE_IMG_W)), cv2.COLOR_BGR2GRAY)
        scr = cv2.threshold(scr, 1, 255, cv2.THRESH_BINARY)
        return np.array(scr)

    def step(self, a: T_Action) -> tuple[Any, int, bool]:
        prev_scr = self._game.score
        self._game.step(a)
        rew = (self._game.score - prev_scr) / 10
        self._n_steps += 1

        done = False
        if (not self._game.running) or self._n_steps >= self._MAX_STEPS:
            done = True
            self._game.start()
            self._n_steps = 0

        self.state = self._scr_proc()
        return self.state, rew, done
