# cython: language_level=3
from __future__ import print_function
import numpy as np


def feature_extraction(int[:, :, :] image):

    image = np.ascontiguousarray(image)
    cdef int height, width
    height, width = image.shape[0], image.shape[1]
    for y in range(height):
        for x in range(width):
            image[y][x][0] = int(image[y][x][0] / 255)
            image[y][x][1] = int(image[y][x][1] / 255)
            image[y][x][2] = int(image[y][x][2] / 255)

    return image
