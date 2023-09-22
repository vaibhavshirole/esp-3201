# -*- coding: utf-8 -*-

import pygame
import sys
import argparse
import time

from pygame.locals import *
from agent import Agent
from maze import Maze
from search import search

filename = 'map/single/tinyMaze.txt'
maze = Maze(filename)
print(maze)

statesExplored = maze.getStatesExplored()
print('statesExplored:', statesExplored)

objectives = maze.getObjectives()
print('objectives:', objectives)

state = maze.getStart()
print('state:', state)

# this will +1 to maze.statesExplored
neighbors = maze.getNeighbors(state[0], state[1])
print('neighbors:', neighbors)

print('statesExplored:', maze.getStatesExplored())

for neighbor in neighbors:
    print(neighbor, maze.isWall(neighbor[0], neighbor[1]))
    print(maze.getNeighbors(neighbor[0], neighbor[1]))
    print('statesExplored:', maze.getStatesExplored())


