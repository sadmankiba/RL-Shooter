import random
import inspect
import logging
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
    K_SPACE,
    K_s
)
import numpy as np
import cv2
from nptyping import NDArray

from util import parent_dir

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCORE_HEIGHT = 80
ADDENEMY = pygame.USEREVENT + 1

PLAYER_SIZE = (25, 75)
PLAYER_IMG_SIZE = (75, 75)
ENEMY_SIZE = (25, 50)
ENEMY_IMG_SIZE = (75, 75)
MISSILE_SIZE = (8, 5)
MISSILE_IMG_SIZE = (24, 24)

PLAYER_MOVE_STEP = 20
ENEMY_SPEED = 3
MISSILE_SPEED = 25

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 102)
GRAY = (40, 40, 50)

STATE_IMG_H = 40
STATE_IMG_W = 40

T_STATE = NDArray[(STATE_IMG_H, STATE_IMG_W), int]
T_Action = int

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("model-logger")

class Action:
    UP = 0
    DOWN = 1
    NOP = 2
    SHOOT = 3

    def __len__(self):
        return 4
    
    @classmethod
    def rev(self, a: int):
        if a == 0: 
            return "up"
        elif a == 1: 
            return "down"
        elif a == 2: 
            return "nop"
        elif a == 3: 
            return "shoot"


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        use_img = True
        if use_img:
            self.surf = pygame.image.load(
                f"{parent_dir(inspect.currentframe()).parent}/assets/spaceship.png"
            ).convert_alpha()
        else:
            self.surf = pygame.Surface(PLAYER_SIZE)
            self.surf.fill(WHITE)
        self.rect = self.surf.get_rect(
            center=(
                int(PLAYER_SIZE[0] / 1.5),
                random.randint(
                    int(SCORE_HEIGHT + PLAYER_SIZE[1] / 2),
                    int(SCREEN_HEIGHT - PLAYER_SIZE[1] / 2),
                ),
            )
        )

    def update(self, a: T_Action) -> None:
        if a == Action.UP:
            self.rect.move_ip(0, -PLAYER_MOVE_STEP)
        elif a == Action.DOWN:
            self.rect.move_ip(0, PLAYER_MOVE_STEP)

        if self.rect.top <= SCORE_HEIGHT:
            self.rect.top = SCORE_HEIGHT
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT


class Missile(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int):
        super(Missile, self).__init__()
        use_img = False
        if use_img:
            self.surf = pygame.image.load(
                f"{parent_dir(inspect.currentframe()).parent}/assets/missile.png"
            ).convert_alpha()
        else:
            self.surf = pygame.Surface(MISSILE_SIZE)
            self.surf.fill(GRAY)
        self.rect = self.surf.get_rect(center=(x, y))
        self.speed = MISSILE_SPEED

    def update(self):
        self.rect.move_ip(self.speed, 0)


class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super(Enemy, self).__init__()
        use_img = True
        if use_img:
            self.surf = pygame.image.load(
                f"{parent_dir(inspect.currentframe()).parent}/assets/spaceship_left.png"
            ).convert_alpha()
        else:
            self.surf = pygame.Surface(ENEMY_SIZE)
            self.surf.fill(WHITE)
        
        self.rect = self.surf.get_rect(
            center=(
                random.randint(SCREEN_WIDTH + 20, SCREEN_WIDTH + 100),
                random.randint(SCORE_HEIGHT + 40, SCREEN_HEIGHT - 40),
            )
        )
        self.speed = ENEMY_SPEED

    def update(self):
        self.rect.move_ip(-self.speed, 0)


class Game:
    def __init__(self, fps: int, display: bool) -> None:
        pygame.init()
        self._fps = fps
        pygame.time.set_timer(ADDENEMY, fps * 20)
        self._display = display
        self._score_font = pygame.font.SysFont("comicsansms", 20)
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self._clock = pygame.time.Clock()
        self.start()

    def start(self):
        self.running = True
        self._player = Player()
        self._enemies = pygame.sprite.Group()
        self._missiles = pygame.sprite.Group()
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
            elif pressed_keys[K_s]:
                a = Action.SHOOT
            else:
                a = Action.NOP

            self.step(a)

    def step(self, a: T_Action):
        self._check_event()
        if a == Action.SHOOT:
            m = Missile(self._player.rect.centerx, self._player.rect.centery)
            self._missiles.add(m)
            self._all_sprites.add(m)
        elif a == Action.UP or a == Action.DOWN:
            self._player.update(a)
        
        self._enemies.update()
        self._missiles.update()

        for enemy in self._enemies:
            if pygame.sprite.collide_rect(enemy, self._player) or enemy.rect.right < 0:
                self.running = False

            if missile := pygame.sprite.spritecollideany(enemy, self._missiles):
                self.score += 1
                enemy.kill()
                missile.kill()

        self.render()

    def render(self):
        self._screen.fill(BLACK)

        for entity in self._all_sprites:
            self._screen.blit(entity.surf, entity.rect)

        self._render_score()

        if self._display:
            pygame.display.flip()
        
        self._clock.tick(self._fps)

    def _check_event(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                self.running = False

            elif event.type == QUIT:
                self.running = False

            elif event.type == ADDENEMY:
                new_enemy = Enemy()
                self._enemies.add(new_enemy)
                self._all_sprites.add(new_enemy)

    def _check_click(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                return True
        
        return False

    def _render_score(self):
        value = self._score_font.render("Your Score: " + str(self.score), True, YELLOW)
        self._screen.blit(value, [0, 0])


class ShooterEnv:
    def __init__(self, game: Game):
        self._game = game
        self._game.render()
        self.state = self._scr_proc()
        self.actions: Iterable[T_Action] = range(len(Action()))
        self._MAX_STEPS = 100000
        self._n_steps = 0
        self.games_played = 0

    def _scr_proc(self) -> T_STATE:
        scr = pygame.surfarray.array3d(pygame.display.get_surface())
        scr = np.transpose(scr, axes=[1, 0, 2])
        scr = scr[SCORE_HEIGHT:, :, :]
        scr = cv2.cvtColor(
            cv2.resize(scr, (STATE_IMG_H, STATE_IMG_W)), cv2.COLOR_BGR2GRAY
        )
        _, scr = cv2.threshold(scr, 100, 255, cv2.THRESH_BINARY)
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
            self.games_played += 1
            self._n_steps = 0

        self.state = self._scr_proc()
        # log.debug(f"[ENV-STEP]: act {a}, s_nxt {self.state.tolist()}, rew {rew}, done {done}")
        
        return self.state, rew, done
