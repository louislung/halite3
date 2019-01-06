#!/usr/bin/env python3
# equivalent to bot version 3

# Change History
# Date                  Desc
# ----                  ----
# 2018-12-31 00:00      initial version


# Import the Halite SDK, which will let you interact with the game.
import hlt
from hlt import constants
from hlt.positionals import Direction
import random
import logging

# This game object contains the initial game state.
game = hlt.Game()
# Respond with your name.
game.ready("MyPythonBot")
ship_status = {}


def safe_move_check(shipid, targetpos):
    safe = True
    for ship in me.get_ships():
        if shipid == ship.id: continue
        if targetpos == ship.position: safe = False
        if ship.id in command.keys():
            if targetpos == command[ship.id]['targetpos']: safe = False

    return safe


while True:
    # Get the latest game state.
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map

    # A command queue holds all the commands you will run this turn.
    command_queue = []
    command = {}

    for ship in me.get_ships():
        logging.info("Ship {} has {} halite.".format(ship.id, ship.halite_amount))
        command[ship.id] = {'ship': ship, 'targetpos': ship.position}

        if ship.id not in ship_status:
            ship_status[ship.id] = "exploring"

        if ship_status[ship.id] == "returning":
            if ship.position == me.shipyard.position:
                ship_status[ship.id] = "exploring"
            else:
                move = game_map.naive_navigate(ship, me.shipyard.position)
                if not safe_move_check(ship.id, ship.position.directional_offset(move)):
                    move = 'o'
                    continue
                command_queue.append(ship.move(move))
                command[ship.id]['targetpos'] = ship.position.directional_offset(move)
                continue
        elif ship.halite_amount >= constants.MAX_HALITE / 1.2:
            ship_status[ship.id] = "returning"
            move = game_map.naive_navigate(ship, me.shipyard.position)
            command_queue.append(ship.move(move))
            command[ship.id]['targetpos'] = ship.position.directional_offset(move)
            continue

        # exploring
        # For each of your ships, move randomly if the ship is on a low halite location or the ship is full.
        #   Else, collect halite.
        ship_moved = False
        for direction in [Direction.North, Direction.South, Direction.East, Direction.West]:
            surrounding = ship.position.directional_offset(direction)
            if game_map[surrounding].halite_amount > game_map[ship.position].halite_amount * 1.5:
                if not safe_move_check(ship.id, surrounding): continue
                command_queue.append(ship.move(direction))
                command[ship.id]['targetpos'] = ship.position.directional_offset(direction)
                ship_moved = True
                break

        if ship_moved:
            continue

        if game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full:
            for i in range(4):
                choice = random.choice([Direction.North, Direction.South, Direction.East, Direction.West])
                if not safe_move_check(ship.id, ship.position.directional_offset(choice)): continue
                command_queue.append(ship.move(choice))
                command[ship.id]['targetpos'] = ship.position.directional_offset(choice)
                ship_moved = True
                break
            if not ship_moved:
                command_queue.append(ship.stay_still())
        else:
            command_queue.append(ship.stay_still())

    if me.halite_amount >= 1000 and not game_map[me.shipyard].is_occupied and game.turn_number < 300 and len(me.get_ships()) < 10:
        command_queue.append(me.shipyard.spawn())

    # If you're on the first turn and have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though.
    # if game.turn_number <= 1 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
    #     command_queue.append(game.me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
