import numpy as np
import matplotlib.pyplot as plt

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


def gen_density_map(halite_map, distance=8, discount=1.1):
    assert halite_map.shape[0] == halite_map.shape[1]

    density_map = np.zeros((halite_map.shape[0], halite_map.shape[1]))

    for y in range(halite_map.shape[0]):
        for x in range(halite_map.shape[1]):
            distance_map = gen_distance_map(x, y, halite_map.shape[0])
            density_map[y, x] = ((distance_map <= distance) * halite_map * (float(discount) ** -distance_map)).mean()

    return density_map


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
plt.imshow(halite_map, cmap='Blues_r', interpolation='none')
plt.colorbar()
plt.title('halite map')
plt.show()

# Then generate a density map see if it helps to find places with more high value cells
for i in range(1,4):
    density_map = gen_density_map(halite_map, i)
    plt.imshow(density_map, cmap='Blues_r', interpolation='nearest')
    plt.colorbar()
    plt.title('density map with distance = {}'.format(i))
    plt.show()