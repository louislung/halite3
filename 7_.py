#!/usr/bin/env python3
# equivalent to bot version 12

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
# 2019-01-06 19:00      Ship with low halite no need to return at the end
#                       Set ship status to exploit when few places >= MIN_HALITE_TO_STAY within farthest place it can go
#                       Spawn ship will calculate the expected return now, this gives better results in a small map / 4v4 gamp
# 2019-01-09 02:00      Updated distance_map_from_position to accounts for wrap around
#                       Fixed RANDOM_SEED (added np.random.seed)
#                       Will convert to dropoff now
# 2019-01-13 01:00      Updated default value of MAX_SHIP_ON_MAP=200, MAX_SPAWN_SHIP_TURN=0.9
#                       Added parameter MAX_MAKE_DROPOFF_TURN=0.9, MAKE_DROPOFF_DENSITY_QUANTILE=0.9, MAKE_DROPOFF_GAIN_COST_RATIO=3, SPAWN_GAIN_COST_RATIO=2
#                       Deprecated PLACES_LEFT_FOR_EXPLOIT
#                       Added make dropoff logic (convert_to_dropoff_conditions_check):
#                           Calculate density map using discounted mean
#                           If ship reached a cell with density >= MAKE_DROPOFF_DENSITY_QUANTILE quantile
#                               and gain (sum of halite of nearby cells without ship) >= cost * MAKE_DROPOFF_GAIN_COST_RATIO
#                           Make dropoff
#                       Updated set to exploit logic
#                           1. if cells >= MIN_HALITE_TO_STAY less than total ships on map Or
#                           2. 0.75 quantile of halite that ship can explore < MIN_HALITE_TO_STAY
#                       Updated spawn ship logic, gain >= cost * _gain_cost_ratio (set to 2 for now)
#                       Rename function
#                           move_farther_from_shipyard -> move_farther_from_dropoff
#                           ensure_shipyard_not_blocked -> ensure_dropoff_not_blocked
#                       Ship will explore using gaussian filtered map
#                       Ship will not move to cell which has enemy ship nearby while exploring
# 2019-01-13 23:00      Take inspiration into account in expected halite calculation
#                       In 2p game, will try to keep ship number 5 more than enemy (ignore expected gain and cost)
#                       Added collision mode, will actively collide enemy ship
#                       Apply gaussian filter to inspired halite map for exploring ships
# 2019-01-16 23:00      Added paramter COLLISION_2P, COLLISION_4P, COLLISION_FRIENDLY_SHIP_DISTANCE, COLLISION_FRIENDLY_SHIP_RATIO, COLLISION_HALITE_GAIN_COST_RATIO
#                       Turn collision mode off in 4P
#####################################################################################################################


# Todo
# 1. refine calculation of expected value for inspired ship
# 2. a smarter way to determine spawn ship or not, expected gain of spawning a ship should > ship cost
# 3. in log, print how birth and death round of a ship, print sum of how many ships collided
# 4. improve return and end, now ships always concentrated as a straight line
# 5. at early game state, set max_halite to 300 so ship will return and spawn more ship early
# 6. should save halite to make dropoff?
# 7. ship should move deeper into halite dense region to create dropoff

#########
# Setup #
#########
import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import random
import logging
import numpy as np, argparse, time
from scipy.ndimage.filters import generic_filter, gaussian_filter, convolve
from scipy.stats import iqr, kurtosis

parser = argparse.ArgumentParser()
# General
parser.add_argument("--RANDOM_SEED", default=-1, type=int, help="random seed, -1 = no need to set the seed")
parser.add_argument("--HALITE_DISCOUNT_RATIO", default=1.5, type=float, help="discount ratio to calculate expected halite collected by a ship")

# Explore related
parser.add_argument("--MAX_EXPECTED_HALITE_ROUND", default=8, type=int, help="max future round in calculating expected halite")
parser.add_argument("--MIN_HALITE_TO_STAY", default=50, type=int, help="min halite for a ship to stay and collect")
parser.add_argument("--MIN_HALITE_QUANTILE_TO_STAY", default=0.5, type=int, help="override MIN_HALITE_TO_STAY if this is lower")
parser.add_argument("--MOVE_BACK_PROB", default=0., type=float, help="prob that an naive exploring ship will move closer to shipyard")
parser.add_argument("--HALITE_GAUSSIAN_DISTANCE", default=8, type=float, help="distance to generate gauissan filtered map")
parser.add_argument("--USE_INSPIRED_HALITE_MAP", default=1, type=int, help="apply gaussian filer on inspired halite map")

# Return related
parser.add_argument("--MAX_HALITE_RETURN", default=950, type=int, help="a ship will return if collected more than this number")
parser.add_argument("--MOVE_AROUND_WHEN_BLOCK_IN_RETURN", default=1, type=int, help="move around the ship if blocked by enemy ship during return")

# Spawn ship related
parser.add_argument("--MAX_SHIP_ON_MAP", default=200, type=int, help="max no. of ship allowed on map")
parser.add_argument("--MAX_SPAWN_SHIP_TURN", default=0.9, type=float, help="stop spawn ship after this turn, range from 0 to 1")
parser.add_argument("--SPAWN_GAIN_COST_RATIO", default=2, type=float, help="make dropoff if sum of halite around > cost by this factor")
parser.add_argument("--MIN_HALITE_QUANTILE_TO_SPAWN", default=50, type=int, help="halite amount to check against and decide spawn or not")
parser.add_argument("--MIN_QUANTILE_TO_SPAWN", default=0.5, type=float, help="quantile of halite to check against and decide spawn or not")

# Dropoff related
parser.add_argument("--HALITE_DENSITY_DISTANCE", default=3, type=int, help="distance used to generate density map")
parser.add_argument("--MAX_MAKE_DROPOFF_TURN", default=0.9, type=float, help="stop make dropoff after this turn, range from 0 to 1")
parser.add_argument("--MAKE_DROPOFF_DENSITY_QUANTILE", default=0.9, type=float, help="density quantile threshold to consider making a dropoff")
parser.add_argument("--MAKE_DROPOFF_GAIN_COST_RATIO", default=3, type=float, help="make dropoff if sum of halite around > cost by this factor")
parser.add_argument("--DROPOFF_GAIN_DISTANCE", default=4, type=float, help="sum of halite within this distance will be considered as gain")
parser.add_argument("--DROPOFF_MIN_DISTANCE", default=8, type=float, help="create dropoff only if no other dropoff within this distance")

# Collision related
parser.add_argument("--COLLISION_2P", default=1, type=int, help="enable collision in 2p")
parser.add_argument("--COLLISION_4P", default=0, type=int, help="enable collision in 4p")
parser.add_argument("--COLLISION_FRIENDLY_SHIP_DISTANCE", default=2, type=float, help="distance to check friendly ship ratio before collision")
parser.add_argument("--COLLISION_FRIENDLY_SHIP_RATIO", default=1, type=float, help="ratio of friendly ship over enemy ship")
parser.add_argument("--COLLISION_HALITE_GAIN_COST_RATIO", default=1.5, type=float, help="collide if gain more than cost by this ratio")

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
        self.MAX_MAKE_DROPOFF_TURN = int(constants.MAX_TURNS * args.MAX_MAKE_DROPOFF_TURN)
        self.HALITE_DISCOUNT_RATIO = args.HALITE_DISCOUNT_RATIO
        self.MAX_EXPECTED_HALITE_ROUND = args.MAX_EXPECTED_HALITE_ROUND
        self.MIN_HALITE_TO_STAY = args.MIN_HALITE_TO_STAY
        self.MAX_HALITE_RETURN = args.MAX_HALITE_RETURN
        self.MOVE_BACK_PROB = args.MOVE_BACK_PROB
        self.MOVE_AROUND_WHEN_BLOCK_IN_RETURN = args.MOVE_AROUND_WHEN_BLOCK_IN_RETURN
        self.MAKE_DROPOFF_GAIN_COST_RATIO = args.MAKE_DROPOFF_GAIN_COST_RATIO
        self.SPAWN_GAIN_COST_RATIO = args.SPAWN_GAIN_COST_RATIO
        self.MAKE_DROPOFF_DENSITY_QUANTILE = args.MAKE_DROPOFF_DENSITY_QUANTILE
        self.MIN_HALITE_QUANTILE_TO_SPAWN = args.MIN_HALITE_QUANTILE_TO_SPAWN
        self.MIN_QUANTILE_TO_SPAWN = args.MIN_QUANTILE_TO_SPAWN
        self.HALITE_DENSITY_DISTANCE = args.HALITE_DENSITY_DISTANCE
        self.DROPOFF_GAIN_DISTANCE = args.DROPOFF_GAIN_DISTANCE
        self.DROPOFF_MIN_DISTANCE = args.DROPOFF_MIN_DISTANCE
        self.HALITE_GAUSSIAN_DISTANCE = args.HALITE_GAUSSIAN_DISTANCE
        self.MIN_HALITE_QUANTILE_TO_STAY = args.MIN_HALITE_QUANTILE_TO_STAY
        self.USE_INSPIRED_HALITE_MAP = args.USE_INSPIRED_HALITE_MAP
        self.COLLISION_2P = args.COLLISION_2P
        self.COLLISION_4P = args.COLLISION_4P
        self.COLLISION = 0
        if len(game.players) == 2 and args.COLLISION_2P:
            self.COLLISION = 1
        elif len(game.players) == 4 and args.COLLISION_4P:
            self.COLLISION = 1
        self.COLLISION_FRIENDLY_SHIP_DISTANCE = args.COLLISION_FRIENDLY_SHIP_DISTANCE
        self.COLLISION_FRIENDLY_SHIP_RATIO = args.COLLISION_FRIENDLY_SHIP_RATIO
        self.COLLISION_HALITE_GAIN_COST_RATIO = args.COLLISION_HALITE_GAIN_COST_RATIO

cust_constants = custom_constants()

ship_status = {}
previous_halite_amount = 9999999999
save_halite_for_dropoff = False
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


def get_optimize_naive_move(source, destination, turns=5, safe=True, return_cost=False, discount_factor=1, safe_against_own=True):
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
        elif safe_against_own and game_map[target_position].ship is not None and game_map[target_position].ship.owner == me.id:
            continue
        naive_move.append(direction)
        cost.append(get_move_cost(source))

    if turns > 1:
        for idx, direction in enumerate(naive_move):
            m, c = get_optimize_naive_move(source.directional_offset(direction), destination, turns=turns-1, safe=False, return_cost=True, safe_against_own=False)
            cost[idx] += c / (discount_factor ** (turns-1))

    if len(naive_move) == 0:
        naive_move=[Direction.Still]
        cost=[0]

    if return_cost:
        return naive_move[np.argmin(cost)], np.min(cost)
    else:
        return naive_move[np.argmin(cost)]


def ensure_dropoff_not_blocked(ship, dropoff):
    i = 0

    # no worries if dropoff is not occupied
    if game_map[dropoff].is_occupied:
        i+=1
    else:
        return

    # if dropoff occupied, make sure there is one exit left
    for p in dropoff.get_surrounding_cardinals():
        if game_map[p].is_occupied:
            i+=1
        else:
            free = p

    if i==5:
        logging.error('Dropoff blocked !!!')
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


def check_enemy_ship_nearby(source):
    # return True is there is a enemy ship around
    # todo: update to use ship_map instead
    for p in source.get_surrounding_cardinals():
        if game_map[p].ship is not None and game_map[p].ship.owner != me.id:
            return True
    return False


def collide_with_enemy(ship, friendly_ship_distance = 2, friendly_ship_ratio = 1/3, halite_gain_cost_ratio = 1.1):

    if not check_enemy_ship_nearby(ship.position):
        # no enemy to collide with
        return False

    distance_map = distance_map_from_position(ship.position, turns=friendly_ship_distance, safe=False)
    friendly_ship_nearby = ((ship_map == 1) * (distance_map > 0)).sum()
    enemy_ship_nearby = ((ship_map == -1) * (distance_map > 0)).sum()

    if friendly_ship_nearby == 0:
        # none of my ship nearby
        return False

    if friendly_ship_ratio >=1 and friendly_ship_nearby / enemy_ship_nearby >=  friendly_ship_ratio:
        # enemy ship out numbered my ship
        return False
    if friendly_ship_ratio < 1 and friendly_ship_nearby / enemy_ship_nearby <=  friendly_ship_ratio:
        # enemy ship out numbered my ship
        return False

    for p in ship.position.get_surrounding_cardinals():
        if game_map[p].ship is not None and game_map[p].ship.owner != me.id:
            _pos_halite = game_map[p].halite_amount
            _enemy_halite = game_map[p].ship.halite_amount
            if _enemy_halite + _pos_halite > ship.halite_amount * halite_gain_cost_ratio:
                move = game_map.get_unsafe_moves(ship.position, p)[0]
                command_ship(ship, 'move', move)
                return True


def new_exploring(ship, min_halite_to_stay = cust_constants.MIN_HALITE_TO_STAY):
    # logging.debug('[new_exploring] started ship id={}, pos={}'.format(ship.id, ship.position))
    # start_time = time.time()

    if cust_constants.COLLISION:
        collided = collide_with_enemy(ship, friendly_ship_distance=cust_constants.COLLISION_FRIENDLY_SHIP_DISTANCE,
                                      friendly_ship_ratio=cust_constants.COLLISION_FRIENDLY_SHIP_RATIO,
                                      halite_gain_cost_ratio=cust_constants.COLLISION_HALITE_GAIN_COST_RATIO)
        if collided: return

    if cust_constants.COLLISION and int(0.3 * constants.MAX_TURNS) <= game.turn_number and ship.halite_amount <= 400:
        # may move close to enemy ship
        _collision = True
    else:
        _collision = False

    # Check if the ship is being surrounded
    i = 0
    for surrounding in ship.position.get_surrounding_cardinals():
        if game_map[surrounding].is_occupied: i+=1
    if i == 4:
        # end with no command as surrounded
        # logging.debug('[new_exploring] ended no command as surrounded, in {} s'.format(time.time() - start_time))
        return

    # Explore using expected gains
    for p in exploring_next_turns(ship.position, ship.halite_amount, cust_constants.MAX_EXPECTED_HALITE_ROUND, min_halite_to_stay):
        move = get_optimize_naive_move(ship.position, p, safe=not(_collision))
        if (move == Direction.Still and p != ship.position) or (p == ship_data[ship.id]['closest_dropoff_position']):
            continue
        elif not _collision and check_enemy_ship_nearby(normalize_directional_offset(ship.position, move)):
            continue
        else:
            command_ship(ship, 'move', move)
            # logging.debug('[new_exploring] ended expected gains move={}, in {} s'.format(move, time.time() - start_time))
            return

    # Explore using density
    distance_map = distance_map_from_position(ship.position, safe=False, turns=1)
    target = np.unravel_index(((distance_map == 1) * halite_gaussian_map).argmax(), halite_gaussian_map.shape)
    move = get_optimize_naive_move(ship.position, Position(target[1],target[0]))
    if move != Direction.Still \
            and not check_enemy_ship_nearby(normalize_directional_offset(ship.position, move)) \
            and not normalize_directional_offset(ship.position, move) == ship_data[ship.id]['closest_dropoff_position']:
        command_ship(ship, 'move', move)
        return

    # Fail to find any moves above, fall back to naive algorithm
    if game_map[ship.position].halite_amount < min_halite_to_stay:
        move = move_farther_from_dropoff(ship)
        command_ship(ship, 'move', move)
        # logging.debug('[new_exploring] ended naive algo, in {} s'.format(time.time() - start_time))
        return


def move_farther_from_dropoff(ship):
    # logging.debug('[move_farther_from_shipyard] started ship id={}'.format(ship.id))
    direction_list = gen_random_direction_list()
    # dropoff = me.shipyard.position
    dropoff = ship_data[ship.id]['closest_dropoff_position']
    if dropoff == ship.position:
        farther_away = np.array([True] * 4)
    else:
        current_distance = game_map.calculate_distance(ship.position, dropoff)
        farther_away = np.array(list(map(
            lambda d: game_map.calculate_distance(normalize_directional_offset(ship.position, d), dropoff) > current_distance, direction_list)))
        # set a direction closer to shipyard to True with a probability
        if np.random.rand() <= cust_constants.MOVE_BACK_PROB:
            farther_away[(~farther_away).nonzero()[0][0]] = True

    for direction in direction_list[farther_away]:
        target_position = normalize_directional_offset(ship.position, direction)
        if safe_move_check(ship.id, target_position) and dropoff != target_position and not check_enemy_ship_nearby(target_position):
            # logging.debug('[move_farther_from_shipyard] ended')
            return direction

    # logging.debug('[move_farther_from_shipyard] ended')
    return Direction.Still


def returning(ship):
    # move to shipyard, stay still if no safe move
    move = get_optimize_naive_move(ship.position, ship_data[ship.id]['closest_dropoff_position'])

    # update on 2019-01-01: move around ship if block by enemy for 1 turn
    if move == Direction.Still and ship.id in previous_ship_data.keys() and previous_ship_data[ship.id]['originalpos'] == ship.position:
        # didn't move in last round and not going to move, check if block by others
        unsafe_move = game_map.get_unsafe_moves(ship.position, ship_data[ship.id]['closest_dropoff_position'])[0]
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
    if ship.position == ship_data[ship.id]['closest_dropoff_position']:
        command_ship(ship, 'move', Direction.Still)
    elif normalize_directional_offset(ship.position, game_map.get_unsafe_moves(ship.position, ship_data[ship.id]['closest_dropoff_position'])[0]) == ship_data[ship.id]['closest_dropoff_position']:
        move = game_map.get_unsafe_moves(ship.position, ship_data[ship.id]['closest_dropoff_position'])[0]
        command_ship(ship, 'move', move)
    else:
        returning(ship)
    return


def spawn_ship():
    # logging.debug('[spawn_ship] started')
    # start_time = time.time()
    places_left_to_explore = (halite_map >= cust_constants.MIN_HALITE_TO_STAY).sum()

    # divided by 2 as need to return to shipyard
    # divieded by collect_turn as assume the ship move then collect for collect_turn for each distance
    collect_turn = 1
    explore_distance = int(int((constants.MAX_TURNS - game.turn_number -1) / 2) / (1+collect_turn))
    _cost = (constants.SHIP_COST + constants.DROPOFF_COST) if save_halite_for_dropoff else constants.SHIP_COST

    # do not spawn if not enough cost
    if me_halite_left < _cost:
        logging.info('[spawn_ship] stopped spawn ship coz not enough halite={}, cost={}'.format(me_halite_left, _cost))
        return

    # do not spawn if shipyard occupied
    if game_map[me.shipyard.position].is_occupied:
        logging.info('[spawn_ship] stopped spawn ship coz shipyard occupied')
        return

    # do not spawn if not enough places left to explore
    if places_left_to_explore <= total_ships: #updated on 2019-01-06, compared with total_ships
        logging.info('[spawn_ship] stopped spawn ship coz places_left_to_explore={}, total_ships={}'.format(places_left_to_explore, total_ships))
        return

    # do not spawn if my ships > MAX_SHIP_ON_MAP
    if len(me.get_ships()) > cust_constants.MAX_SHIP_ON_MAP:
        logging.info('[spawn_ship] stopped spawn ship coz reached MAX_SHIP_ON_MAP'.format(len(me.get_ships())))
        return

    # do not spawn if turn > MAX_SPAWN_SHIP_TURN
    if game.turn_number > cust_constants.MAX_SPAWN_SHIP_TURN:
        logging.info('[spawn_ship] stopped spawn ship coz reached MAX_SPAWN_SHIP_TURN')
        return

    if np.quantile(halite_map, cust_constants.MIN_QUANTILE_TO_SPAWN) <= cust_constants.MIN_HALITE_QUANTILE_TO_SPAWN:
        logging.info('[spawn_ship] stopped spawn ship coz halite at {} quantile={}'.format(cust_constants.MIN_QUANTILE_TO_SPAWN, np.quantile(halite_map, cust_constants.MIN_QUANTILE_TO_SPAWN)))
        return

    # if 2p game, maintains ship number >= enemy + 5
    if len(game.players) == 2 and len(me.get_ships()) - (total_ships - len(me.get_ships())) <= 5:
        logging.info('[spawn_ship] spawn in 2p game!')
        command_queue.append(me.shipyard.spawn())
        # logging.debug('[spawn_ship] ened time={} s'.format(time.time() - start_time))
        return

    # do not spawn if not enough to explore full map and expect gain is too low
    if explore_distance > game_map.width: pass
    else:
        distance_map = distance_map_from_position(me.shipyard.position, safe=True, turns=explore_distance)
        expected_halite = 0
        d=0

        for d in range(1,explore_distance+1):
            max_halite = ((distance_map == d) * halite_map).max()
            if max_halite < cust_constants.MIN_HALITE_TO_STAY: continue
            expected_halite += max_halite / constants.EXTRACT_RATIO + max_halite / (constants.EXTRACT_RATIO**2) + max_halite / (constants.EXTRACT_RATIO**3)
            if expected_halite > constants.SHIP_COST * cust_constants.SPAWN_GAIN_COST_RATIO: break

        logging.info('[spawn_ship] explore_distance={}, expected_halite={}, d={}'.format(explore_distance, expected_halite, d))

        if expected_halite <= constants.SHIP_COST * cust_constants.SPAWN_GAIN_COST_RATIO:
            return

    logging.info('[spawn_ship] spawn')
    command_queue.append(me.shipyard.spawn())
    # logging.debug('[spawn_ship] ened time={} s'.format(time.time() - start_time))
    return


def command_ship(ship, action, move=None):
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

        if target_position == ship_data[ship.id]['closest_dropoff_position']:
            ship_data[ship.id]['moving_to_dropoff'] = True

        if target_position == ship_data[ship.id]['closest_dropoff_position'] or target_position in ship_data[ship.id]['closest_dropoff_position'].get_surrounding_cardinals():
            ensure_dropoff_not_blocked(ship, ship_data[ship.id]['closest_dropoff_position'])

        command = ship.move(move)

    elif action == 'convert_dropoff':
        command = ship.make_dropoff()

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


def get_convert_dropoff_cost(ship):
    # return the cost for a ship to convert into dropoff at its current position
    cost = constants.DROPOFF_COST
    cost -= ship.halite_amount
    cost -= game_map[ship.position].halite_amount
    return cost


def get_extract_halite(halite, inspired=False):
    if halite == 0: return 0
    _extract_ratio = constants.INSPIRED_EXTRACT_RATIO if inspired else constants.EXTRACT_RATIO
    return max(1, int(halite / _extract_ratio))


def set_halite_map():
    halite_map = np.zeros([game.game_map.height, game.game_map.width])
    for w in range(game.game_map.width):
        for h in range(game.game_map.height):
            halite_map[h, w] = game_map[Position(w, h)].halite_amount
    return halite_map


def set_inspired_halite_map(halite_map, ship_map, ship_threshold=4, bonus_multiplier=3, bonus_distance=4):
    # return a halite map taking inspiration into account
    # cell with >= X enemy ship around will multiple halite by a factor

    def _check(buffer, threshold, multiplier):
        return multiplier if (buffer == -1).sum() >= threshold else 1

    row_array = np.array([range(bonus_distance * 2 + 1), ] * (bonus_distance * 2 + 1)).transpose()
    col_array = np.array([range(bonus_distance * 2 + 1), ] * (bonus_distance * 2 + 1))
    distance_map = abs(row_array - bonus_distance) + abs(col_array - bonus_distance)
    weights = (distance_map <= bonus_distance) * 1.
    multiplier_map = generic_filter(ship_map, _check, footprint=weights, mode='wrap', extra_arguments=(ship_threshold, bonus_multiplier))
    return multiplier_map * halite_map


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


def position_to_tuple(position):
    return (position.x, position.y)


def distance_map_from_position(position, safe=True, turns=5):
    # get distance map, position requires turn >= turns will mark as 0
    # distance_map = abs(row_array - position.y) + abs(col_array - position.x)
    distance_map = np.minimum(abs(row_array - position.y), game_map.height - abs(row_array - position.y)) + \
                   np.minimum(abs(col_array - position.x), game_map.width - abs(col_array - position.x))

    # # Needs special handling since edges are connected
    # # for row
    # to_be_replace = int(game_map.height / 2. - min(position.y, game_map.height - position.y - 1) - 1)
    # if to_be_replace > 0:
    #     if position.y < game_map.height / 2.:
    #         distance_map[-to_be_replace:, :] = np.flip(
    #             distance_map[-to_be_replace - to_be_replace - 1:-to_be_replace - 1, :], 0)
    #     else:
    #         distance_map[:to_be_replace, :] = np.flip(
    #             distance_map[to_be_replace + 1:to_be_replace + to_be_replace + 1, :], 0)
    # # for col
    # to_be_replace = int(game_map.width / 2. - min(position.x, game_map.width - position.x - 1) - 1)
    # if to_be_replace > 0:
    #     if position.x < game_map.width / 2.:
    #         distance_map[:, -to_be_replace:] = np.flip(
    #             distance_map[:, -to_be_replace - to_be_replace - 1:-to_be_replace - 1], 1)
    #     else:
    #         distance_map[:, :to_be_replace] = np.flip(
    #             distance_map[:, to_be_replace + 1:to_be_replace + to_be_replace + 1], 1)

    distance_map = (distance_map <= turns) * distance_map

    # occcupied position within turns mark as 0
    if safe:
        for p in zip(*distance_map.nonzero()):
            if game_map[Position(*p[::-1])].is_occupied:
                distance_map[p] = 0

    return distance_map


def exploring_next_turns(source, ship_halite, explore_distance=5, min_halite_to_stay=cust_constants.MIN_HALITE_TO_STAY, safe=True):
    # logging.debug('[exploring_next_turns] started source={}'.format(source))
    # start_time = time.time()
    pos = np.array([])
    pos_expected_value = np.array([])

    # get distance map from source position
    distance_map = distance_map_from_position(source, safe=safe, turns=explore_distance)

    # loop over every distance from source, from 0 to explore_distance
    for d in range(explore_distance+1):
        if d==0:
            masked_halite_map = np.zeros([game_map.height, game_map.width])
            masked_halite_map[source.y, source.x] = 1
            masked_halite_map = masked_halite_map * halite_map
        else:
            masked_halite_map = (distance_map == d) * halite_map

        if masked_halite_map.max() == 0: continue

        # get expected value for position which has the max halite
        for p in zip(*(masked_halite_map == masked_halite_map.max()).nonzero()):
            e = new_expected_value(source, Position(*p[::-1]), ship_halite, min_halite_to_stay)
            if e > 0:
                pos = np.append(pos, Position(*p[::-1]))
                pos_expected_value = np.append(pos_expected_value, e)

    # logging.debug('[exploring_next_turns] ended in {} s'.format(time.time()-start_time))
    return pos[(-pos_expected_value).argsort()]


def new_expected_value(source, destination, ship_halite, min_halite_to_stay=cust_constants.MIN_HALITE_TO_STAY):
    # logging.debug('[new_expected_value] started')
    # start_time = time.time()
    # will explore the expected value for a ship to move to that destination and collect til full
    expected_value = 0
    inspired = check_inspired(destination)

    # logging.debug('[new_expected_value] distance={}, source={}, destination={}'.format(game_map.calculate_distance(source, destination), source, destination))
    m, c = get_optimize_naive_move(source, destination, turns=9999, safe=False, return_cost=True, discount_factor=cust_constants.HALITE_DISCOUNT_RATIO)

    expected_value += -c
    used_turns = game_map.calculate_distance(source, destination)
    destination_halite = game_map[destination].halite_amount

    while ship_halite < cust_constants.MAX_HALITE_RETURN:
        # logging.debug('[new_expected_value] loop destination_halite={}, ship_halie={}, expected_value={}'.format(destination_halite, ship_halite, expected_value))
        if destination_halite < min_halite_to_stay: break
        stay_reward = min(get_extract_halite(destination_halite, inspired), constants.MAX_HALITE - ship_halite)
        destination_halite -= stay_reward
        if inspired: stay_reward = min(stay_reward * (1 + constants.INSPIRED_BONUS_MULTIPLIER), constants.MAX_HALITE - ship_halite)
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


def get_closest_dropoff_position(source, turns=None):
    dropoffs = list(me.get_dropoffs())
    dropoffs.append(me.shipyard)
    distance = []
    for dropoff in dropoffs:
        distance.append(game_map.calculate_distance(source, dropoff.position))

    if turns is not None and np.min(distance) > turns:
        return None

    return dropoffs[np.argmin(distance)].position


def gen_density_map(any_map, distance=3, discount=1.1, stat='mean'):
    row_array = np.array([range(distance * 2 + 1), ] * (distance * 2 + 1)).transpose()
    col_array = np.array([range(distance * 2 + 1), ] * (distance * 2 + 1))
    distance_map = abs(row_array - distance) + abs(col_array - distance)
    weights = (distance_map <= distance) * (float(discount) ** -distance_map)
    if stat == 'mean':
        weights = weights / (distance_map <= distance).sum()
        density_map = convolve(any_map, weights, mode='wrap')
    else:
        density_map = generic_filter(any_map, calculate_density, footprint=weights, mode='wrap', extra_arguments=(weights, stat))
    return density_map


def calculate_density(buffer, weights = None, stat='mean'):
    # this function should be used with scipy.ndimage.filters.generic_fliter
    if weights is not None:
        weights = weights.ravel()
        weights = weights[weights != 0]
    else:
        weights = np.ones(buffer.shape)

    _func = {
        'mean': np.mean,
        'var': np.var,
        'std': np.std,
        'max': np.max,
        'min': np.min,
        'median': np.median,
        'sum': np.sum,
        'iqr': iqr,
        '75quantile': lambda x: np.quantile(x, 0.75)
    }

    assert weights.shape == buffer.shape
    assert stat in _func.keys()
    density = _func[stat](buffer * weights)
    return density


def convert_to_dropoff_conditions_check(ship):
    _gain_cost_ratio = cust_constants.MAKE_DROPOFF_GAIN_COST_RATIO
    _cost = get_convert_dropoff_cost(ship)

    # check MAX_MAKE_DROPOFF_TURN
    if game.turn_number > cust_constants.MAX_MAKE_DROPOFF_TURN:
        logging.debug('[convert_to_dropoff_conditions_check] reached MAX_MAKE_DROPOFF_TURN')
        return 0

    if make_dropoff:
        logging.debug('[convert_to_dropoff_conditions_check] shipid={} other ship will make dropoff'.format(ship.id))
        return 0

    # not convert if not reach dense halite region
    _halite_density = halite_density_map[ship.position.y, ship.position.x]
    _halite_density_q = np.quantile(halite_density_map, cust_constants.MAKE_DROPOFF_DENSITY_QUANTILE)
    if _halite_density < _halite_density_q:
        logging.debug('[convert_to_dropoff_conditions_check] shipid={} low halite density={}, quantile={}'.format(ship.id, _halite_density, _halite_density_q))
        return 0

    # not convert if gain <= cost * ratio (halite sum within nearby region)
    _gain = ((distance_map_from_position(ship.position, safe=True, turns=cust_constants.DROPOFF_GAIN_DISTANCE) > 0) * halite_map).sum()
    if _gain <=  _cost * _gain_cost_ratio:
        logging.debug('[convert_to_dropoff_conditions_check] shipid={} low gain={}'.format(ship.id, _gain))
        return 0

    # check if nearby has no other dropoff
    if get_closest_dropoff_position(ship.position, turns=cust_constants.DROPOFF_MIN_DISTANCE) is not None:
        logging.debug('[convert_to_dropoff_conditions_check] shipid={} nearby has dropoff'.format(ship.id))
        return 0

    # check enough cost
    if me_halite_left < _cost:
        logging.debug('[convert_to_dropoff_conditions_check] shipid={} not enough halite'.format(ship.id))
        # not make dropoff only because of not enough halite, should save halite
        return -1

    # friendly ship density?

    # enemy ship density?

    # going to make drop off, log stat
    logging.info('Make dropoff, ship id={}, cost={}, gain={}, halite density={}, halite density 90% quantil={}'.format(
        ship.id, _cost, _gain, _halite_density, _halite_density_q
    ))

    return 1


def check_inspired(source):
    # return True if ship is inspired at source position
    distance_map = distance_map_from_position(source, safe=False, turns=game_map.width)
    inspired = (((distance_map <= constants.INSPIRATION_RADIUS) * ship_map) == -1).sum() >= constants.INSPIRATION_SHIP_COUNT
    return inspired


def reject_outliers(data, min_q=0.05, max_q=0.95):
    q05 = np.quantile(data, min_q)
    q95 = np.quantile(data, max_q)
    data = data[data >= q05]
    data = data[data <= q95]
    return data


##################
# Main game loop #
##################
instruction_queue = {}
while True:
    overall_start_time = time.time()
    start_time = time.time()
    game.update_frame()
    me = game.me
    game_map = game.game_map
    logging.debug('extract from game time={} s'.format(time.time() - start_time))

    start_time = time.time()
    halite_map = set_halite_map()
    logging.debug('set halite map time={} s'.format(time.time() - start_time))

    start_time = time.time()
    ship_map = set_ship_map()
    logging.debug('set ship map time={} s'.format(time.time() - start_time))

    start_time = time.time()
    halite_density_map = gen_density_map(halite_map, distance=cust_constants.HALITE_DENSITY_DISTANCE, discount=cust_constants.HALITE_DISCOUNT_RATIO, stat='mean')
    logging.debug('set halite density map time={} s'.format(time.time() - start_time))

    start_time = time.time()
    inspired_halite_map = set_inspired_halite_map(halite_map, ship_map, constants.INSPIRATION_SHIP_COUNT, 1 + constants.INSPIRED_BONUS_MULTIPLIER, constants.INSPIRED_EXTRACT_RATIO)
    logging.debug('set inspired_halite_map time={} s'.format(time.time() - start_time))

    start_time = time.time()
    if cust_constants.USE_INSPIRED_HALITE_MAP:
        halite_gaussian_map = gaussian_filter(inspired_halite_map, cust_constants.HALITE_GAUSSIAN_DISTANCE)
    else:
        halite_gaussian_map = gaussian_filter(halite_map, cust_constants.HALITE_GAUSSIAN_DISTANCE)
    logging.debug('set halite gaussian_filter map time={} s'.format(time.time() - start_time))

    start_time = time.time()
    me_halite_left = me.halite_amount

    # if game.turn_number <= 20:
    #     cust_constants.MAX_HALITE_RETURN = 700
    # else:
    #     cust_constants.MAX_HALITE_RETURN = args.MAX_HALITE_RETURN

    # A command queue holds all the commands you will run this turn.
    command_queue = []
    ship_data = {}
    make_dropoff = False
    total_ships = 0
    remaining_turns = constants.MAX_TURNS - game.turn_number
    for i in game.players:
        total_ships += len(game.players[i].get_ships())

    logging.info('Kurtosis={}'.format(kurtosis(reject_outliers(halite_map, 0.05, 0.95), None)))
    for q in [0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1]:
        logging.info('{}: {}'.format(q,np.quantile(halite_map,q)))

    # set instuction from previous round
    exec_instruction()
    logging.debug('prepare time={} s'.format(time.time()-start_time))
    start_time = time.time()
    for ship in me.get_ships():
        dropoff_position = get_closest_dropoff_position(ship.position)
        ship_data[ship.id] = {'ship': ship, 'originalpos': ship.position, 'targetpos': ship.position, 'commanded': None,
                              'closest_dropoff_position': dropoff_position, 'halite_amount': ship.halite_amount, 'moving_to_dropoff': False}

        #
        # Set ship status
        #

        # Convert ship to dropoff
        _convert_dropoff_signal = convert_to_dropoff_conditions_check(ship)
        if _convert_dropoff_signal == 1:
            ship_status[ship.id] = "convert_dropoff"
            me_halite_left -= get_convert_dropoff_cost(ship)
            make_dropoff = True
            save_halite_for_dropoff = False
            continue
        # elif _convert_dropoff_signal == -1 and len(me.get_ships()) >= 20 + len(list(me.get_dropoffs())) * 10:
        #     save_halite_for_dropoff = True

        # Set ship to exploring if status not found
        if ship.id not in ship_status:
            ship_status[ship.id] = "exploring"

        distance = game_map.calculate_distance(ship.position, ship_data[ship.id]['closest_dropoff_position'])
        if constants.MAX_TURNS - game.turn_number - 1 <= distance + int(len(me.get_ships())/(2+len(me.get_dropoffs()))) <= constants.MAX_TURNS - game.turn_number + 1 \
            and ship.halite_amount > game_map.calculate_distance(ship.position, ship_data[ship.id]['closest_dropoff_position']):
            ship_status[ship.id] = "returning_and_end"

        # For ship which is returning in last round
        if ship_status[ship.id] == "returning":
            # Set to exploring if returned
            if ship.position == ship_data[ship.id]['closest_dropoff_position']:
                ship_status[ship.id] = "exploring"
        # For ship which is exploring
        elif ship_status[ship.id] in ["exploring", "exploit"]:
            if ship.halite_amount > cust_constants.MAX_HALITE_RETURN:
                ship_status[ship.id] = "returning"

        if ship_status[ship.id] == "exploring":
            # if map has no much halite left
            if (halite_map >= cust_constants.MIN_HALITE_TO_STAY).sum() <= total_ships:
                ship_status[ship.id] = "exploit"

            # To decide set a ship to exploit or not based on cells left to explore around the ship
            # explore_distance = int(int((constants.MAX_TURNS - game.turn_number - game_map.calculate_distance(ship.position, ship_data[ship.id]['closest_dropoff_position'])) / 2) / (1 + 2))
            explore_distance = int((remaining_turns - 1.1 * game_map.calculate_distance(ship.position, ship_data[ship.id]['closest_dropoff_position'])) / 2)
            if explore_distance >= game_map.width: continue

            distance_map = distance_map_from_position(ship.position, False, explore_distance)
            _halite_map = (distance_map > 0) * halite_map
            _halite_map.astype(float)
            _halite_map[_halite_map==0] = np.nan
            # places_left_to_explore = (((distance_map > 0) * halite_map) >= cust_constants.MIN_HALITE_TO_STAY).sum()

            if np.nanquantile(_halite_map, 0.75) < cust_constants.MIN_HALITE_TO_STAY:
                logging.info('set to exploit ship id={}, 75 quantile={}'.format(ship.id, np.nanquantile(_halite_map, 0.75)))
                ship_status[ship.id] = "exploit"

    logging.debug('set ship status time={} s'.format(time.time() - start_time))
    start_time = time.time()
    #
    # Send command according to status
    #
    for ship in me.get_ships():
        if ship_status[ship.id] == "convert_dropoff":
            command_ship(ship,'convert_dropoff')
    logging.debug('convert dropoff time={} s'.format(time.time() - start_time))
    start_time = time.time()
    _i = 1

    for ship in me.get_ships():
        if ship_status[ship.id] == "exploring":
            _i += 1
            new_exploring(ship, min(cust_constants.MIN_HALITE_TO_STAY, np.quantile(halite_map, cust_constants.MIN_HALITE_QUANTILE_TO_STAY)))
        elif ship_status[ship.id] == "exploit":
            _i += 1
            new_exploring(ship, 1)
    logging.debug('exploring ship={}, time={} s, avg={} s'.format(_i, round(time.time() - start_time,4), round((time.time() - start_time) / _i, 4)))
    start_time = time.time()

    for ship in me.get_ships():
        if ship_status[ship.id] == "returning":
            returning(ship)
    logging.debug('returning time={} s'.format(time.time() - start_time))
    start_time = time.time()

    for ship in me.get_ships():
        if ship_status[ship.id] == "returning_and_end":
            returning_and_end(ship)
        # else:
        #     logging.error('A ship without status is found, set to stay still')
        #     command_ship(ship, 'move', Direction.Still)
    logging.debug('returning_and_end time={} s'.format(time.time() - start_time))
    start_time = time.time()

    spawn_ship()
    logging.debug('spawn ship time={} s'.format(time.time() - start_time))
    start_time = time.time()

    # Collect stat for analysis
    for ship in me.get_ships():
        if ship.id not in analysis['ship_existed_turn'].keys():
            analysis['ship_existed_turn'][ship.id] = 0
            analysis['ship_collected_halite'][ship.id] = 0
        else:
            analysis['ship_existed_turn'][ship.id] += 1

        if ship_data[ship.id]['moving_to_dropoff']:
            analysis['ship_collected_halite'][ship.id] += ship_data[ship.id]['halite_amount']

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
    logging.debug('collect stat time={} s'.format(time.time() - start_time))
    logging.debug('OVERALL time={} s'.format(time.time() - overall_start_time))
    # Send your moves back to the game environment, ending this turn.
    logging.debug(command_queue)
    logging.debug(ship_status)
    game.end_turn(command_queue)
