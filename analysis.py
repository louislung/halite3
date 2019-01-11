import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import generic_filter

def gen_game_map(width, factor = 2):
    # higher factor means more cells have high halite
    _min = np.exp(-499/100)
    _max = np.exp(-1/100)
    outer = np.clip(np.random.rand(int(width / 4), int(width / 1)) * _max * 1 / factor, _min, _max)
    inner = np.clip(np.random.rand(int(width / 4), int(width / 1)) * _max, _min, _max)
    m = np.concatenate([outer, inner])
    m = np.concatenate([m, np.flip(m, 0)])
    m = -np.log(m) * 100
    m = m.astype(int)
    m = m + np.flip(m,1)
    return m


def gen_density_map(halite_map, distance, discount=1.1, stat='mean'):
    row_array = np.array([range(distance * 2 + 1), ] * (distance * 2 + 1)).transpose()
    col_array = np.array([range(distance * 2 + 1), ] * (distance * 2 + 1))
    distance_map = abs(row_array - distance) + abs(col_array - distance)
    weights = (distance_map <= distance) * (float(discount) ** -distance_map)

    density_map = generic_filter(halite_map, calculate_density, footprint=weights, mode='wrap', extra_arguments=(weights, stat))
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
    }

    assert weights.shape == buffer.shape
    assert stat in _func.keys()
    density = _func[stat](buffer * weights)
    return density


def gen_distance_map(x, y, width, turns=9999):
    row_array = np.array([range(width), ] * width).transpose()
    col_array = np.array([range(width), ] * width)

    distance_map = abs(row_array - y) + abs(col_array - x)

    # Needs special handling since edges are connected
    # for row
    to_be_replace = int(width / 2. - min(y, width - y - 1) - 1)
    if to_be_replace > 0:
        if y < width / 2.:
            distance_map[-to_be_replace:, :] = np.flip(distance_map[-to_be_replace - to_be_replace - 1:-to_be_replace - 1, :], 0)
        else:
            distance_map[:to_be_replace, :] = np.flip(distance_map[to_be_replace + 1:to_be_replace + to_be_replace + 1, :], 0)
    # for col
    to_be_replace = int(width / 2. - min(x, width - x - 1) - 1)
    if to_be_replace > 0:
        if x < width / 2.:
            distance_map[:, -to_be_replace:] = np.flip(distance_map[:, -to_be_replace - to_be_replace - 1:-to_be_replace - 1], 1)
        else:
            distance_map[:, :to_be_replace] = np.flip(distance_map[:, to_be_replace + 1:to_be_replace + to_be_replace + 1], 1)

    distance_map = (distance_map <= turns) * distance_map

    return distance_map


# Generate a halite map
halite_map = gen_game_map(64, 1.3)
halite_map = np.random.rand(64,64)
# halite_map[30,30] = 10
for _ in ['1','101','201','301','401']:
    halite_map = np.load('halite_map_turn_{}.npy'.format(_))
    plt.imshow(halite_map, cmap='Blues_r', interpolation='none')
    plt.colorbar()
    plt.title('halite map turn = {}'.format(_))
    plt.show()

    # Then generate a density map see if it helps to find places with more high value cells
    for i in range(1,4,2):
        density_map = gen_density_map(halite_map, i)
        plt.imshow(density_map, cmap='Blues_r', interpolation='nearest')
        plt.colorbar()
        plt.title('density map with distance = {}'.format(i))
        plt.show()
