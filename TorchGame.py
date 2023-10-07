import pygame
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BACKGROUND_COLOR = (32, 44, 57)

COLOR_GREEN = (68, 175, 105)
COLOR_BLUE  = (25, 133, 161)

# get a color morph between green and blue
MORPH_GREENBLUE = lambda x: tuple((1-x)*np.array(COLOR_GREEN) + x*np.array(COLOR_BLUE))

# CIRCLE_COLOR = (68, 175, 105)    # Green
CIRCLE_COLOR = (25, 133, 161)  # Blue
CIRCLE_RADIUS = 20
CIRCLE_MASS = 1

# Initialize pygame
pygame.init()
pygame.display.set_caption('TorchGame')

class Object():
    def __init__(self, x=None, y=None, color=None, radius=None):
        self.x = x if x is not None else np.random.randint(0, SCREEN_WIDTH)
        self.y = y if y is not None else np.random.randint(0, SCREEN_HEIGHT)
        self.color = color
        self.radius = radius
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

class Game(nn.Module):

    def __init__(self, 
            N=2,
        ):
        super().__init__()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.circles = [
            Object(color=MORPH_GREENBLUE(i/(N-1)), 
            radius=CIRCLE_RADIUS) for i in range(N)
        ]

        Z = torch.zeros((N, 2))
        Z = torch.abs(torch.rand_like(Z)) * (SCREEN_WIDTH + SCREEN_HEIGHT)/2
        self.Z = torch.nn.parameter.Parameter(Z)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-6)
    
    def sync_Z(self):
        # updates the circles coordinates based on Z tensor
        for i, c in enumerate(self.circles):
            c.x = self.Z[i, 0].item()
            c.y = self.Z[i, 1].item()

    def forward(self):
        self.physics()

    def physics(self):
    
        # updates the Z tensor based on the potential energy defined by the pairwise distances
        with torch.autograd.set_detect_anomaly(True):
            def V_lj(R):
                eps=1e-6
                scale = 0.01
                R *= scale
                return 4 * (1/(R**12+eps) - 1/(R**6+eps))
            def V_rep(R):
                # simple repulsive potential
                eps=1e-6
                scale = 0.01
                R *= scale
                return 1/(R**2+eps)

            X, Y = self.Z.T
            # R = torch.sqrt((X - X.T)**2 + (Y - Y.T)**2)
            XX = (X[:, None] - X[None, :])**2
            YY = (Y[:, None] - Y[None, :])**2
            R = torch.sqrt(XX + YY)

            # print(R.shape, R)
            # R = R - 2*CIRCLE_RADIUS
            # R = torch.clamp(R, min=0)
            
            # Calculate the potential energy using a simple lennard-jones potential
            # U = V_lj(R)  # (N, N)
            U = V_rep(R)  # (N, N)

            # update the positions of the circles based on the Z tensor
            # self.Z.data -= 0.01 * self.Z.grad  # gradient descent
            U = (1-torch.eye(len(U))) * U

            self.optimizer.zero_grad()
            U.sum().backward()
            self.optimizer.step()
            print('POTENTIAL ENERGY', U.sum())

            # update the positions of the circles based on the Z tensor
            self.sync_Z()        

            # if torch.any(torch.isnan(self.Z)):
            #     print("NaNs in Z!", U, self.Z, R, X, Y)
            #     raise Exception("NaNs in Z!")
        
    def draw(self):
        # Drawing
        self.screen.fill(BACKGROUND_COLOR)
        [
            c.draw(self.screen) for c in self.circles
        ]
        pygame.display.flip()

game = Game()
# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            exit()

    game.physics()
    game.draw()
