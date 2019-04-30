#!/usr/bin/env python3
# equivalent to bot version 13 & 15

#####################################################################################################################
#
# Change History
#
# Date                  Desc
# ----                  ----
# 2019-03-13 00:00      Initial version
#
#####################################################################################################################


# Todo
# 1.

#########
# Setup #
#########
import hlt
from hlt import constants, commands
from hlt.positionals import Direction, Position
import random
import logging
import numpy as np, argparse, time
from simple_dqn import *
import keras

parser = argparse.ArgumentParser()
# General
parser.add_argument("--RANDOM_SEED", default=-1, type=int, help="random seed, -1 = no need to set the seed")

args = parser.parse_args()

if args.RANDOM_SEED != -1:
    random.seed(args.RANDOM_SEED)
    np.random.seed(args.RANDOM_SEED)

game = hlt.Game()
game.ready("LouisBot")


###################
# Custom function #
###################
def set_halite_map():
    halite_map = np.zeros([game.game_map.height, game.game_map.width])
    for w in range(game.game_map.width):
        for h in range(game.game_map.height):
            halite_map[h, w] = game_map[Position(w, h)].halite_amount
    return halite_map


def set_ship_map():
    # enemy ship marked as -1
    # my ship marked as 1
    ship_map = np.zeros([game.game_map.height, game.game_map.width])
    for w in range(game.game_map.width):
        for h in range(game.game_map.height):
            if game_map[Position(w, h)].ship:
                if game_map[Position(w, h)].ship.owner == me.id:
                    ship_map[h, w] = 1
                else:
                    ship_map[h, w] = -1
    return ship_map


##################
# Main game loop #
##################
q_network = keras.models.load_model('q_network')
target_network = keras.models.load_model('target_network')

replay = []

greedy_epsilon = 0.1
greedy_epsilon_decay = 0.9999


while True:
    # Get the latest game state.
    # Extract player metadata and the updated map metadata convenience.
    game.update_frame()
    me = game.me
    game_map = game.game_map
    command_queue = []

    if game.turn_number == 1:
        # init sequence of raw_state and processed_state
        raw_state_seq =

    for ship in me.get_ships():

        state = preprocess(raw_state)

        # Select actions from epsilon-greedy policy
        a = e_greedy(state, greedy_epsilon * greedy_epsilon_decay ** t)


    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

