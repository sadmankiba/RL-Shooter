import random

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

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCORE_HEIGHT = 80
ADDENEMY = pygame.USEREVENT + 1

ENEMY_ADD_TIMER = 2000

PLAYER_SIZE = (75, 25)
ENEMY_SIZE = (20, 10)

BLACK = (0, 0, 0)
YELLOW = (255, 255, 102)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.surf = pygame.Surface(PLAYER_SIZE)
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect()
    
    def update(self, pressed_keys):
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -5)
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0, 5)
        if pressed_keys[K_LEFT]:
            self.rect.move_ip(-5, 0)
        if pressed_keys[K_RIGHT]:
            self.rect.move_ip(5, 0)

        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH
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
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        self._running = True
        self._player = Player()
        self._enemies = pygame.sprite.Group()
        self._all_sprites = pygame.sprite.Group()
        self._all_sprites.add(self._player)
        self._score = 0

        pygame.time.set_timer(ADDENEMY, ENEMY_ADD_TIMER)
        
    def run(self):
        while self._running:
            self._check_event()
            pressed_keys = pygame.key.get_pressed()

            self._player.update(pressed_keys)
            self._enemies.update()

            self._screen.fill(BLACK)

            for entity in self._all_sprites:
                self._screen.blit(entity.surf, entity.rect)

            for enemy in self._enemies:
                if enemy.rect.right < 0: 
                    self._running = False

                if pygame.sprite.collide_rect(enemy, self._player):
                    self._score += 1
                    enemy.kill()

            self._update_score()
            pygame.display.flip()

            clock.tick(60)

    def _check_event(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self._running = False

            elif event.type == QUIT:
                self._running = False
            
            elif event.type == ADDENEMY:
                new_enemy = Enemy()
                self._enemies.add(new_enemy)
                self._all_sprites.add(new_enemy)

    def _update_score(self):
        value = score_font.render("Your Score: " + str(self._score), True, YELLOW)
        self._screen.blit(value, [0, 0])

if __name__ == "__main__":
    pygame.init()
    score_font = pygame.font.SysFont("comicsansms", 20)
    
    clock = pygame.time.Clock()
    
    game = Game() 
    game.run()