# My bot to the [Halite III Competition](https://halite.io/) 

### Quick start
1. Make sure python version = 3.6.7 is installed
2. Download python started kit [here](https://halite.io/learn-programming-challenge/downloads)
3. Put the scipts of this repo under the same folder of the starter kit
4. Modify **run_game.sh** (inside starter kit) as below and run it

```shell
# Play with different parameters, setting RANDOM_SEED so it is reproducible
a="python3 5_.py --RANDOM_SEED 1"
b="python3 5_.py --RANDOM_SEED 1 --MIN_HALITE_TO_STAY 30"
c="python3 5_.py --RANDOM_SEED 1 --MIN_HALITE_TO_STAY 40"
d="python3 5_.py --RANDOM_SEED 1 --MIN_HALITE_TO_STAY 50"

./halite --replay-directory replays/ -vvv "$a" "$b" "$c" "$d"
```

### Version details
| Script  | Rating | Ranking | Description |
| --- | --- | --- | --- |
| 5_.py  | 67.58 | 250 | <ul><li>Ships with low halite do not return to shipyard at the end of game</li><li>Set ship status to exploit when few places >= MIN_HALITE_TO_STAY within farthest place it can go</li><li>Decide spawn ship or not based on expected return (i.e. if not much halite left to explore), this gives better results in a small map or 4v4 game</li></ul> |
| 4_.py  | 65.56 | 322 | <ul><li>Implemented new exploring function (new_exploring, exploring_next_turns, get_optimize_naive_move, new_expected_value, distance_map_from_position)</li><li>New exploring mechanism (details under below sections)</li><li>Returning ship will use get_optimize_naive_move to find path with lower cost</li><li>Avoided blocking ship at shipyard, and will try to move around enemy ship during return</li><li>Stop spawning ship if cells that can be collected (>= MIN_HALITE_TO_STAY) is <= ship number</li><li>Updated default value of MAX_SHIP_ON_MAP=40, HALITE_DISCOUNT_RATIO=1.5, MAX_EXPECTED_HALITE_ROUND=8</li></ul> |
| 3_.py  | 57.55 | 627 | <ul><li>Revamp safe_move_check, leverage mark_unsafe method of hlt</li><li>Exploring will try to move farther away from shipyard (measured by manhattan distance)</li><li>Improve returning (will try to move around if block by enemy's ship)</li><li>New function: exec_instruction, set_instruction, naive_navigate_wo_mark_unsafe, gen_random_direction_list</li></ul> |
| 2_.py  | 54.14 | 775 | <ul><li>Refactor code</li><li>Ships will move based on "Expected Halite" (discounted by round). </li><li>Ships will return to shipyard near end of the game</li></ul> |
| 1_.py  | 44.7 | 1369 | Based on starter code provided on halite website |

### Explore mechanism (implemented from 4_.py)
**Apply logic below to ships 1 by 1**
  
For distance d range from 0 to MAX_EXPECTED_HALITE_ROUND away from the ship:  
* Found the cell (distance = d and not occupied) with max halite
* Calculate expected cost if move to that cell  
* Calculate expected gain if stay in that cell and collect halite, minus the cost above  
* Order the possible cells to move by expected gain (descending order)  

For all possible cells from above:
* If there exists a naive move to that cell, do it, if not, try next possible cell             