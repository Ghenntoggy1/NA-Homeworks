import random
import math
import numpy as np
from sklearn import preprocessing
from PIL import Image

SCALE = 50
MAP_SIZE = (100, 100)

def interpolate(a, b, x):
    ft = x * math.pi
    f = (1 - math.cos(ft)) * 0.5
    return a * (1 - f) + b * f

class PerlinNoise:
    def __init__(self, seed=None):
        self.seed = seed
        self.permutation_table = list(range(256))
        if seed is not None:
            random.seed(seed)
            random.shuffle(self.permutation_table)
            self.permutation_table *= 2

    def noise(self, x, y, z):
        if self.seed is not None:
            x += self.seed
            y += self.seed
            z += self.seed

        xi = int(x) & 255
        yi = int(y) & 255
        zi = int(z) & 255

        xf = x - int(x)
        yf = y - int(y)
        zf = z - int(z)

        u = fade(xf)
        v = fade(yf)
        w = fade(zf)

        a = self.permutation_table[xi] + yi
        aa = self.permutation_table[a] + zi
        ab = self.permutation_table[a + 1] + zi
        b = self.permutation_table[xi + 1] + yi
        ba = self.permutation_table[b] + zi
        bb = self.permutation_table[b + 1] + zi

        grad_aa = self._grad(self.permutation_table[aa], xf, yf, zf)
        grad_ba = self._grad(self.permutation_table[ba], xf - 1, yf, zf)
        grad_ab = self._grad(self.permutation_table[ab], xf, yf - 1, zf)
        grad_bb = self._grad(self.permutation_table[bb], xf - 1, yf - 1, zf)
        grad_aa1 = self._grad(self.permutation_table[aa + 1], xf, yf, zf - 1)
        grad_ba1 = self._grad(self.permutation_table[ba + 1], xf - 1, yf, zf - 1)
        grad_ab1 = self._grad(self.permutation_table[ab + 1], xf, yf - 1, zf - 1)
        grad_bb1 = self._grad(self.permutation_table[bb + 1], xf - 1, yf - 1, zf - 1)

        x1 = interpolate(grad_aa, grad_ba, u)
        x2 = interpolate(grad_ab, grad_bb, u)
        y1 = interpolate(x1, x2, v)

        x3 = interpolate(grad_aa1, grad_ba1, u)
        x4 = interpolate(grad_ab1, grad_bb1, u)
        y2 = interpolate(x3, x4, v)

        return (interpolate(y1, y2, w) + 1) / 2

    def _grad(self, hash, x, y, z):
        h = hash & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        return (-u if (h & 1) == 0 else u) + (-v if (h & 2) == 0 else v)

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def update_point(coords, seed):
    perlin_noise = PerlinNoise(seed)
    return perlin_noise.noise(coords[0]/SCALE, coords[1]/SCALE, 0)

def generate_heightmap(map_size):
    seed = int(random.random()*1000)
    minimum = 0
    maximum = 0
    heightmap = np.zeros(map_size)

    for x in range(map_size[0]):
        for y in range(map_size[1]):
            new_value = update_point((x, y), seed)
            heightmap[x][y] = new_value
    return preprocessing.normalize(heightmap)

def rgb_norm(world):
    world_min = np.min(world)
    world_max = np.max(world)
    norm = lambda x: (x-world_min/(world_max - world_min))*255
    return np.vectorize(norm)

def prep_world(world):
    norm = rgb_norm(world)
    world = norm(world)
    return world

map_size = (256, 256)
Image.fromarray(prep_world(generate_heightmap(map_size))).show()