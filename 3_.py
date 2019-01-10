#!/usr/bin/env python3
# equivalent to bot version 6

#####################################################################################################################
#
# Change History
#
# Date                  Desc
# ----                  ----
# 2018-12-31 00:00      initial version
# 2018-12-31 18:00      refactor code
#                       introduced "Expected Halite" (discounted by round), similar to time value of money
#                       all ships will return to shipyard near end of the game
# 2019-01-01 18:00      Revamp safe_move_check, leverage mark_unsafe method of hlt
#                       Exploring will try to move farther away from shipyard (measured by manhattan distance)
#                       Improve returning (will try to move around if block by enemy's ship)
#                       New function: exec_instruction, set_instruction, naive_navigate_wo_mark_unsafe, gen_random_direction_list
#
#####################################################################################################################


#########
# Setup #
#########
import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import random
import logging
import numpy as np, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--MAX_SHIP_ON_MAP", default=20, type=int, help="max no. of ship allowed on map")
parser.add_argument("--MAX_SPAWN_SHIP_TURN", default=0.75, type=float, help="stop spawn ship after this turn, range from 0 to 1")
parser.add_argument("--HALITE_DISCOUNT_RATIO", default=1.1, type=float, help="discount ratio to calculate expected halite collected by a ship")
parser.add_argument("--MAX_EXPECTED_HALITE_ROUND", default=2, type=int, help="max future round in calculating expected halite")
parser.add_argument("--MIN_HALITE_TO_STAY", default=20, type=int, help="min halite for a ship to stay and collect")
parser.add_argument("--MAX_HALITE_RETURN", default=950, type=int, help="a ship will return if collected more than this number")
parser.add_argument("--MOVE_BACK_PROB", default=0.1, type=float, help="prob that an exploring ship will move closer to shipyard")
parser.add_argument("--MOVE_AROUND_WHEN_BLOCK_IN_RETURN", default=1, type=int, help="move around the ship if blocked by enemy ship during return")
parser.add_argument("--RANDOM_SEED", default=-1, type=int, help="random seed, -1 = no need to set the seed")
args = parser.parse_args()

if args.RANDOM_SEED != -1:
    random.seed(args.RANDOM_SEED)
    np.random.seed(args.RANDOM_SEED)

game = hlt.Game()
game.ready("LouisBot")

class custom_constants:
    def __init__(self):
        self.MAX_SHIP_ON_MAP = args.MAX_SHIP_ON_MAP
        self.MAX_SPAWN_SHIP_TURN = int(constants.MAX_TURNS * args.MAX_SPAWN_SHIP_TURN)
        self.HALITE_DISCOUNT_RATIO = args.HALITE_DISCOUNT_RATIO
        self.MAX_EXPECTED_HALITE_ROUND = args.MAX_EXPECTED_HALITE_ROUND
        self.MIN_HALITE_TO_STAY = args.MIN_HALITE_TO_STAY
        self.MAX_HALITE_RETURN = args.MAX_HALITE_RETURN
        self.MOVE_BACK_PROB = args.MOVE_BACK_PROB
        self.MOVE_AROUND_WHEN_BLOCK_IN_RETURN = args.MOVE_AROUND_WHEN_BLOCK_IN_RETURN

cust_constants = custom_constants()

ship_status = {}
previous_halite_amount = 99999999
analysis = {
    'ship_collected_halite': {},
    'ship_existed_turn': {},
}

###################
# Custom function #
###################
def normalize_directional_offset(position, move):
    return game_map.normalize(position.directional_offset(move))


def naive_navigate_wo_mark_unsafe(ship, position):
    # original game_map.naive_navigate will mark the position as unsafe
    # this function will do the same thing without marking unsafe
    move = game_map.naive_navigate(ship, position)
    if move != Direction.Still:
        game_map[ship.position.directional_offset(move)].ship = None
    return move


def gen_random_direction_list():
    # to keep tuple inside np array
    # https://stackoverflow.com/questions/47389447/how-convert-a-list-of-tupes-to-a-numpy-array-of-tuples
    direction_list = np.empty(4, dtype=object)
    direction_list[:] = [Direction.North, Direction.South, Direction.East, Direction.West]
    random.shuffle(direction_list)
    return direction_list


def safe_move_check(id, position, against='all'):
    #
    # check if our own ship is currently on or will move to targetpos
    #
    # logging.info('[safe_move_check]\tid={}\tposition={}\tmap ship={}'.format(id,position,game_map[position].ship))
    if me.get_ship(id).position == position: return True

    safe = not game_map[position].is_occupied

    if against == 'all' or safe:
        return safe
    elif against == 'own':
        occupied_by_me = me.has_ship(game_map[position].ship.id)
        if occupied_by_me:
            return False
        else:
            return True
    else:
        logging.warning('[safe_move_check] should not arrive this step')
        return False


def expected_value(ship, target_position):
    # assume targetpos just 1 move ahead than ship's position at this moment
    # expected value to move to that position and collect halite in next five round

    # assert game_map[target_position].halite_amount > 0
    halite_map_temp = halite_map.copy()
    ship_halite_amount = ship.halite_amount
    ship_position = ship.position
    expected_max_round = min(cust_constants.MAX_EXPECTED_HALITE_ROUND, constants.MAX_TURNS - game.turn_number)
    expected = 0

    round = 0
    while round < expected_max_round:
        move_cost = int(halite_map_temp[position_to_tuple(ship_position)] / constants.MOVE_COST_RATIO)
        stay_reward = min(int(halite_map_temp[position_to_tuple(ship_position)] / constants.EXTRACT_RATIO), constants.MAX_HALITE - ship_halite_amount)

        if ship_halite_amount >= constants.MAX_HALITE:
            # Move ship and end loop if ship reached MAX_HALITE
            expected -= move_cost / (cust_constants.HALITE_DISCOUNT_RATIO ** round)
            ship_halite_amount -= move_cost
            break

        if ship_position != target_position:
            # Move ship if not at target position
            # todo: logic is problemmatic
            ship_position = normalize_directional_offset(ship_position, game_map.get_unsafe_moves(ship_position, target_position)[0])
            expected -= move_cost / (cust_constants.HALITE_DISCOUNT_RATIO ** round)
            ship_halite_amount -= move_cost
        else:
            if stay_reward == 0:
                # No more halite to collect at target position
                expected -= move_cost / (cust_constants.HALITE_DISCOUNT_RATIO ** round)
                ship_halite_amount -= move_cost
                break
            # Collect halite if at target position
            expected += stay_reward / (cust_constants.HALITE_DISCOUNT_RATIO ** round)
            ship_halite_amount += stay_reward
            halite_map_temp[position_to_tuple(ship_position)] -= stay_reward

        round += 1

    return expected


def exploring(ship):
    direction_list = gen_random_direction_list()

    # move to the place where it give better expected halite
    for direction in direction_list:
        target_position = normalize_directional_offset(ship.position, direction)
        if safe_move_check(ship.id, target_position) and me.shipyard.position != target_position \
                and expected_value(ship, target_position) > expected_value(ship, ship.position):
            command_ship(ship, 'move', direction)
            return

    # random move to a safe direction if current halite is in low state else stay and collect
    # 2019-01-01: ship will move farther away from shipyard (with little prob move closer to shipyard)
    if game_map[ship.position].halite_amount < cust_constants.MIN_HALITE_TO_STAY:
        if me.shipyard.position == ship.position:
            farther_away = np.array([True]*4)
        else:
            current_distance = game_map.calculate_distance(ship.position, me.shipyard.position)
            farther_away = np.array(list(map(lambda d: game_map.calculate_distance(normalize_directional_offset(ship.position, d), me.shipyard.position) > current_distance, direction_list)))
            # set a direction closer to shipyard to True with a probability
            if np.random.rand() <= cust_constants.MOVE_BACK_PROB:
                farther_away[(~farther_away).nonzero()[0][0]] = True

        for direction in direction_list[farther_away]:
            target_position = normalize_directional_offset(ship.position, direction)
            if safe_move_check(ship.id, target_position) and me.shipyard.position != target_position:
                command_ship(ship, 'move', direction)
                return

    else:
        command_ship(ship, 'move', Direction.Still)
        return

    # None of the logic work above, so stay still
    command_ship(ship, 'move', Direction.Still)
    return


def returning(ship):
    # move to shipyard, stay still if no safe move
    move = naive_navigate_wo_mark_unsafe(ship, me.shipyard.position)

    if move != Direction.Still and not safe_move_check(ship.id, normalize_directional_offset(ship.position, move)):
        move = Direction.Still

    # update on 2019-01-01: move around ship if block by enemy for 1 turn
    if move == Direction.Still and ship.id in previous_ship_data.keys() and previous_ship_data[ship.id]['originalpos'] == ship.position:
        # didn't move in last round and not going to move, check if block by others
        unsafe_move = game_map.get_unsafe_moves(ship.position, me.shipyard.position)[0]
        if not safe_move_check(ship.id, normalize_directional_offset(ship.position, unsafe_move), 'own'):
            # block by own ship, continue to stay still
            pass
        else:
            # block by enemy ship, move other direction
            for direction in gen_random_direction_list():
                if ship.position.directional_offset(direction) == previous_ship_data[ship.id]['originalpos']: continue
                if safe_move_check(ship.id, normalize_directional_offset(ship.position, direction)):
                    move = direction
                    set_instruction(game.turn_number + 1, 'mark_unsafe', {'position': ship.position, 'ship': ship})
                    break

    command_ship(ship, 'move', move)
    return


def returning_and_end(ship):
    # move to shipyard, stay still if no safe move
    if ship.position == me.shipyard.position:
        command_ship(ship, 'move', Direction.Still)
    elif normalize_directional_offset(ship.position, game_map.get_unsafe_moves(ship.position, me.shipyard.position)[0]) == me.shipyard.position:
        move = game_map.get_unsafe_moves(ship.position, me.shipyard.position)[0]
        command_ship(ship, 'move', move)
    else:
        returning(ship)
    return


def spawn_ship():
    if me.halite_amount >= constants.SHIP_COST \
            and not game_map[me.shipyard].is_occupied \
            and game.turn_number <= cust_constants.MAX_SPAWN_SHIP_TURN \
            and len(me.get_ships()) < cust_constants.MAX_SHIP_ON_MAP:
        command_queue.append(me.shipyard.spawn())
    return


def command_ship(ship, action, move):
    # send command of a ship to command_quene
    if action == 'move':
        # stay still if not enough cost to move
        move = Direction.Still if ship.halite_amount < get_move_cost(ship.position) else move
        target_position = normalize_directional_offset(ship.position, move)

        # update game map for naive_navigate
        if move != Direction.Still:
            game_map[target_position].mark_unsafe(ship)
            game_map[ship.position].ship = None
        # update ship_data
        ship_data[ship.id]['targetpos'] = target_position

        command = ship.move(move)

    command_queue.append(command)
    ship_data[ship.id]['commanded'] = True
    return


def get_move_cost(position):
    # get move cost of any position
    if game_map[position].halite_amount == 0:
        return 0
    elif game_map[position].halite_amount <= constants.MOVE_COST_RATIO:
        return 1
    else:
        return int(game_map[position].halite_amount / constants.MOVE_COST_RATIO)


def set_halite_map():
    halite_map = np.zeros([game.game_map.width, game.game_map.height])
    for w in range(game.game_map.width):
        for h in range(game.game_map.height):
            halite_map[w,h] = game_map[Position(w, h)].halite_amount
    return halite_map


def position_to_tuple(position):
    return (position.x, position.y)


def set_instruction(exec_turn, action, instruction):
    instruction['action'] = action
    if exec_turn not in instruction_queue.keys():
        instruction_queue[exec_turn] = [instruction]
    else:
        instruction_queue[exec_turn].append(instruction)
    return

def exec_instruction():
    if game.turn_number not in instruction_queue.keys(): return

    for instruction in instruction_queue[game.turn_number]:
        if instruction['action'] == 'mark_unsafe':
            game_map[instruction['position']].mark_unsafe(instruction['ship'])

    return

##################
# Main game loop #
##################
instruction_queue = {}
while True:
    # Get the latest game state.
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map
    halite_map = set_halite_map()

    # A command queue holds all the commands you will run this turn.
    command_queue = []
    ship_data = {}

    # set instuction from previous round
    exec_instruction()

    for ship in me.get_ships():
        ship_data[ship.id] = {'ship': ship, 'originalpos': ship.position, 'targetpos': ship.position, 'commanded': None}

        #
        # Set ship status
        #

        # Set ship to exploring if status not found
        if ship.id not in ship_status:
            ship_status[ship.id] = "exploring"

        distance = game_map.calculate_distance(ship.position, me.shipyard.position)
        if constants.MAX_TURNS - game.turn_number - 1 <= distance + int(len(me.get_ships())/2) <= constants.MAX_TURNS - game.turn_number + 1:
            ship_status[ship.id] = "returning_and_end"

        # For ship which is returning in last round
        if ship_status[ship.id] == "returning":
            # Set to exploring if returned
            if ship.position == me.shipyard.position:
                ship_status[ship.id] = "exploring"
        # For ship which is exploring
        elif ship_status[ship.id] == "exploring":
            if ship.halite_amount > cust_constants.MAX_HALITE_RETURN:
                ship_status[ship.id] = "returning"

        #
        # Send command according to status
        #
        if ship_status[ship.id] == "exploring":
            exploring(ship)
        elif ship_status[ship.id] == "returning":
            returning(ship)
        elif ship_status[ship.id] == "returning_and_end":
            returning_and_end(ship)
        else:
            logging.error('A ship without status is found, set to stay still')
            command_ship(ship, 'move', Direction.Still)

    spawn_ship()

    # Collect stat for analysis
    for ship in me.get_ships():
        if ship.id not in analysis['ship_existed_turn'].keys():
            analysis['ship_existed_turn'][ship.id] = 0
            analysis['ship_collected_halite'][ship.id] = 0
        else:
            analysis['ship_existed_turn'][ship.id] += 1

        if ship.position == me.shipyard.position and me.halite_amount > previous_halite_amount:
            analysis['ship_collected_halite'][ship.id] += me.halite_amount - previous_halite_amount

    # Log stat for analysis
    if game.turn_number == constants.MAX_TURNS:
        logging.info('max turn reached')

        for k in analysis['ship_existed_turn'].keys():
            logging.info('ship id={}\texisted turn={}\tcollected halite={}\taverage halite collected per turn={}'.format(
                k, analysis['ship_existed_turn'][k], analysis['ship_collected_halite'][k], analysis['ship_collected_halite'][k] / (analysis['ship_existed_turn'][k] + 0.01)))

        halite_left = 0
        for ship in me.get_ships():
            halite_left += ship.halite_amount
        logging.info('For ships left on map, no. of ships={}, halite in ships={}'.format(len(me.get_ships()), halite_left))


    # Record value
    previous_halite_amount = me.halite_amount
    previous_ship_data = ship_data.copy()

    # Send your moves back to the game environment, ending this turn.
    logging.debug(command_queue)
    logging.debug(ship_status)
    game.end_turn(command_queue)
