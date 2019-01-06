#!/usr/bin/env python3
# equivalent to bot version 7

#####################################################################################################################
#
# Change History
#
# Date                  Desc
# ----                  ----
# 2018-12-31 00:00      Initial version
# 2018-12-31 18:00      Refactor code
#                       Introduced "Expected Halite" (discounted by round), similar to time value of money
#                       All ships will return to shipyard near end of the game
# 2019-01-01 18:00      Revamp safe_move_check, leverage mark_unsafe method of hlt
#                       Exploring will try to move farther away from shipyard (measured by manhattan distance)
#                       Improve returning (will try to move around if block by enemy's ship)
#                       New function: exec_instruction, set_instruction, naive_navigate_wo_mark_unsafe, gen_random_direction_list
# 2019-01-03 16:32      Implemented new exploring function (new_exploring, exploring_next_turns, get_optimize_naive_move, new_expected_value, distance_map_from_position)
#                       The new EXPLORE mechanism is as follows:
#                           For distance d range from 0 to MAX_EXPECTED_HALITE_ROUND:
#                               Found the cell (distance = d and not occupied) with max halite
#                               Calculate expected gain (get_optimize_naive_move) if move to that cell
#                               Calculate expected gain if stay in the cell and collect halite, sum with gain above
#                           Order the possible cell to move by expected gain (descending order)
#                               if there exist a naive move to that cell, do it, if not, try next cell
#                       Returning ship will use get_optimize_naive_move to find path with lower cost
#                       Avoided blocking ship at shipyard, and will try to move around enemy ship during return
#                       Stop spawning ship if cells that can be collected (>= MIN_HALITE_TO_STAY) is <= ship number
#                       Updated default value of MAX_SHIP_ON_MAP=40, HALITE_DISCOUNT_RATIO=1.5, MAX_EXPECTED_HALITE_ROUND=8
#####################################################################################################################


# Todo
# 1. refine calculation of expected value for inspired ship
# 2. a smarter way to determine spawn ship or not, expected gain of spawning a ship should > ship cost
# 3. in log, print how birth and death round of a ship, print sum of how many ships collided
# 4. improve return and end, now ships always concentrated as a straight line

#########
# Setup #
#########
import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import random
import logging
import numpy as np, argparse, time

parser = argparse.ArgumentParser()
parser.add_argument("--MAX_SHIP_ON_MAP", default=40, type=int, help="max no. of ship allowed on map")
parser.add_argument("--MAX_SPAWN_SHIP_TURN", default=0.75, type=float, help="stop spawn ship after this turn, range from 0 to 1")
parser.add_argument("--HALITE_DISCOUNT_RATIO", default=1.5, type=float, help="discount ratio to calculate expected halite collected by a ship")
parser.add_argument("--MAX_EXPECTED_HALITE_ROUND", default=8, type=int, help="max future round in calculating expected halite")
parser.add_argument("--MIN_HALITE_TO_STAY", default=20, type=int, help="min halite for a ship to stay and collect")
parser.add_argument("--MAX_HALITE_RETURN", default=950, type=int, help="a ship will return if collected more than this number")
parser.add_argument("--MOVE_BACK_PROB", default=0.1, type=float, help="prob that an naive exploring ship will move closer to shipyard")
parser.add_argument("--MOVE_AROUND_WHEN_BLOCK_IN_RETURN", default=1, type=int, help="move around the ship if blocked by enemy ship during return")
parser.add_argument("--RANDOM_SEED", default=-1, type=int, help="random seed, -1 = no need to set the seed")
args = parser.parse_args()

if args.RANDOM_SEED != -1:
    random.seed(args.RANDOM_SEED)

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
row_array = np.array([range(game.game_map.height), ] * game.game_map.width).transpose()
col_array = np.array([range(game.game_map.width), ] * game.game_map.height)
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
    # this function will do the same thing as naive_navigate without marking unsafe
    move = game_map.naive_navigate(ship, position)
    if move != Direction.Still:
        game_map[ship.position.directional_offset(move)].ship = None
    return move


def get_optimize_naive_move(source, destination, turns=5, safe=True, return_cost=False, discount_factor=1):
    # return a naive move (must be closer towards destination) that have minimum expected cost in next X turns
    if source == destination:
        if return_cost:
            return Direction.Still, 0
        else:
            return Direction.Still

    naive_move = []
    cost = []

    turns = min(game_map.calculate_distance(source, destination), turns)

    for direction in game_map.get_unsafe_moves(source, destination):
        target_position = source.directional_offset(direction)
        if safe and game_map[target_position].is_occupied:
            continue
        naive_move.append(direction)
        cost.append(get_move_cost(source))

    if turns > 1:
        for idx, direction in enumerate(naive_move):
            m, c = get_optimize_naive_move(source.directional_offset(direction), destination, turns=turns-1, safe=False, return_cost=True)
            cost[idx] += c / (discount_factor ** (turns-1))

    if len(naive_move) == 0:
        naive_move=[Direction.Still]
        cost=[0]

    if return_cost:
        return naive_move[np.argmin(cost)], np.min(cost)
    else:
        return naive_move[np.argmin(cost)]


def ensure_shipyard_not_blocked(ship):
    i = 0

    # no worries if shipyard is not occupied
    if game_map[me.shipyard.position].is_occupied:
        i+=1
    else:
        return

    # if shipyard occupied, make sure there is one exit left
    for p in me.shipyard.position.get_surrounding_cardinals():
        if game_map[p].is_occupied:
            i+=1
        else:
            free = p

    if i==5:
        logging.error('Shipyard blocked !!!')
    elif i==4:
        game_map[free].mark_unsafe(ship)

    return


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
    # expected value to move to that position and collect halite

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


def new_exploring(ship):
    # logging.debug('[new_exploring] started ship id={}, pos={}'.format(ship.id, ship.position))
    # start_time = time.time()
    # Check if the ship is being surrounded
    i = 0
    for surrounding in ship.position.get_surrounding_cardinals():
        if game_map[surrounding].is_occupied: i+=1
    if i == 4:
        # end with no command as surrounded
        # logging.debug('[new_exploring] ended no command as surrounded, in {} s'.format(time.time() - start_time))
        return

    # Explore using expected gains
    for p in exploring_next_turns(ship.position, ship.halite_amount, cust_constants.MAX_EXPECTED_HALITE_ROUND):
        move = get_optimize_naive_move(ship.position, p)
        if (move == Direction.Still and p != ship.position) or (p == me.shipyard.position):
            continue
        else:
            command_ship(ship, 'move', move)
            # logging.debug('[new_exploring] ended expected gains move={}, in {} s'.format(move, time.time() - start_time))
            return

    # Fail to find any moves above, fall back to naive algorithm
    if game_map[ship.position].halite_amount < cust_constants.MIN_HALITE_TO_STAY:
        move = move_farther_from_shipyard(ship)
        command_ship(ship, 'move', move)
        # logging.debug('[new_exploring] ended naive algo, in {} s'.format(time.time() - start_time))
        return


def move_farther_from_shipyard(ship):
    # logging.debug('[move_farther_from_shipyard] started ship id={}'.format(ship.id))
    direction_list = gen_random_direction_list()
    if me.shipyard.position == ship.position:
        farther_away = np.array([True] * 4)
    else:
        current_distance = game_map.calculate_distance(ship.position, me.shipyard.position)
        farther_away = np.array(list(map(
            lambda d: game_map.calculate_distance(normalize_directional_offset(ship.position, d), me.shipyard.position) > current_distance, direction_list)))
        # set a direction closer to shipyard to True with a probability
        if np.random.rand() <= cust_constants.MOVE_BACK_PROB:
            farther_away[(~farther_away).nonzero()[0][0]] = True

    for direction in direction_list[farther_away]:
        target_position = normalize_directional_offset(ship.position, direction)
        if safe_move_check(ship.id, target_position) and me.shipyard.position != target_position:
            # logging.debug('[move_farther_from_shipyard] ended')
            return direction

    # logging.debug('[move_farther_from_shipyard] ended')
    return Direction.Still


def returning(ship):
    # move to shipyard, stay still if no safe move
    move = get_optimize_naive_move(ship.position, me.shipyard.position)

    # update on 2019-01-02: below is useless as get_optimize_naive_move should return safe move now
    # if move != Direction.Still and not safe_move_check(ship.id, normalize_directional_offset(ship.position, move)):
    #     move = Direction.Still

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
    places_left_to_explore = (halite_map >= cust_constants.MIN_HALITE_TO_STAY).sum()

    if me.halite_amount >= constants.SHIP_COST \
            and not game_map[me.shipyard.position].is_occupied \
            and game.turn_number <= cust_constants.MAX_SPAWN_SHIP_TURN \
            and len(me.get_ships()) < cust_constants.MAX_SHIP_ON_MAP\
            and places_left_to_explore > len(me.get_ships()):
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

    if target_position == me.shipyard.position or target_position in me.shipyard.position.get_surrounding_cardinals():
        ensure_shipyard_not_blocked(ship)

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
    halite_map = np.zeros([game.game_map.height, game.game_map.width])
    for w in range(game.game_map.width):
        for h in range(game.game_map.height):
            halite_map[h, w] = game_map[Position(w, h)].halite_amount
    return halite_map


def position_to_tuple(position):
    return (position.x, position.y)


def distance_map_from_position(position, safe=True, turns=5):
    # get distance map, position requires turn >= turns will mark as 0
    distance_map = abs(row_array - position.y) + abs(col_array - position.x)
    distance_map = (distance_map <= turns)*distance_map

    # occcupied position within turns mark as 0
    if safe:
        for p in zip(*distance_map.nonzero()):
            if game_map[Position(*p[::-1])].is_occupied:
                distance_map[p] = 0

    return distance_map


def exploring_next_turns(source, ship_halite, explore_distance=5):
    # logging.debug('[exploring_next_turns] started source={}'.format(source))
    # start_time = time.time()
    pos = np.array([])
    pos_expected_value = np.array([])

    # get distance map in numpy 2-d array
    distance_map = distance_map_from_position(source, safe=True, turns=explore_distance)

    for d in range(explore_distance+1):
        if d==0:
            masked_halite_map = np.zeros([game.game_map.height, game.game_map.width])
            masked_halite_map[source.y, source.x] = 1
            masked_halite_map = masked_halite_map * halite_map
        else:
            masked_halite_map = (distance_map == d) * halite_map

        if masked_halite_map.max() == 0: continue

        # logging.debug('[exploring_next_turns] masked halite map.nonzero={}'.format(masked_halite_map.nonzero()))
        #logging.debug('[exploring_next_turns] in distance={}, max halite={}'.format(d, masked_halite_map.max()))
        for p in zip(*(masked_halite_map == masked_halite_map.max()).nonzero()):
            #logging.debug('[exploring_next_turns] source={}'.format(source))
            e = new_expected_value(source, Position(*p[::-1]), ship_halite)
            if e > 0:
                pos = np.append(pos, Position(*p[::-1]))
                pos_expected_value = np.append(pos_expected_value, e)

    # logging.debug('[exploring_next_turns] pos = {}'.format(pos))
    # logging.debug('[exploring_next_turns] pos expected val = {}'.format(pos_expected_value))
    # logging.debug('[exploring_next_turns] ended in {} s'.format(time.time()-start_time))
    return pos[(-pos_expected_value).argsort()]


def new_expected_value(source, destination, ship_halite):
    # logging.debug('[new_expected_value] started')
    # start_time = time.time()
    # will explore the expected value for a ship to move to that destination and collect til full
    expected_value = 0

    # logging.debug('[new_expected_value] distance={}, source={}, destination={}'.format(game_map.calculate_distance(source, destination), source, destination))
    m, c = get_optimize_naive_move(source, destination, turns=9999, safe=False, return_cost=True, discount_factor=cust_constants.HALITE_DISCOUNT_RATIO)

    expected_value += -c
    used_turns = game_map.calculate_distance(source, destination)
    destination_halite = game_map[destination].halite_amount

    while ship_halite < constants.MAX_HALITE:
        # logging.debug('[new_expected_value] loop destination_halite={}, ship_halie={}, expected_value={}'.format(destination_halite, ship_halite, expected_value))
        if destination_halite < cust_constants.MIN_HALITE_TO_STAY: break
        stay_reward = min(int(destination_halite / constants.EXTRACT_RATIO), constants.MAX_HALITE - ship_halite)
        destination_halite -= stay_reward
        ship_halite += stay_reward
        used_turns += 1
        expected_value += stay_reward / (cust_constants.HALITE_DISCOUNT_RATIO ** used_turns)

    # logging.debug('[new_expected_value] ended in {} s'.format(time.time()-start_time))
    return expected_value


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
    for ship in me.get_ships():
        if ship_status[ship.id] == "exploring":
            new_exploring(ship)

    for ship in me.get_ships():
        if ship_status[ship.id] == "returning":
            returning(ship)

    for ship in me.get_ships():
        if ship_status[ship.id] == "returning_and_end":
            returning_and_end(ship)
        # else:
        #     logging.error('A ship without status is found, set to stay still')
        #     command_ship(ship, 'move', Direction.Still)

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
