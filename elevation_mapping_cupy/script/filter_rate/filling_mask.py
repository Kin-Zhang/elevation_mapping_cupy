import os
import threading
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage.interpolation as csp
from ruamel.yaml import YAML

from filter_rate.filling_kernel import fillingKernel


class FillingMask(object):
    def __init__(self, dim):
        # Configs
        self.resolution = 0.04
        self.map_dim = dim # 8m ==> 200

        # Buffers
        self.thread_lock = threading.Lock()
        self.center = cp.array([0.0, 0.0], dtype=cp.float32)
        self.filled_mask = cp.zeros((1, self.map_dim, self.map_dim), dtype=cp.float32)

        # CuPy Kernels
        self.initializeKernels()

    def initializeKernels(self):
        self.min_sensor_distance_bk = 0.1#float(cfg['kernels']['min_sensor_distance_bk'])
        self.min_sensor_distance_fn = 0.1#float(cfg['kernels']['min_sensor_distance_fn'])
        self.max_sensor_distance = 12.0 #float(cfg['kernels']['min_sensor_distance_fn'])
        self.max_height_range = 5#float(cfg['kernels']['max_height_range'])

        self.filling_kernel = fillingKernel(
            self.resolution, 
            self.map_dim, 
            self.map_dim,
            self.min_sensor_distance_bk, 
            self.min_sensor_distance_fn,
            self.max_sensor_distance, 
            self.max_height_range
        )

    def clearMaks(self):
        print("Clear mask")
        with self.thread_lock:
            self.filled_mask *= 0

    def updateMapCenter(self, center):
        """
        center: 2D vector that contains the new map center position in the world frame
        """
        center = cp.asarray(center)
        delta_pos = center - self.center
        delta_cell = cp.around(delta_pos / self.resolution)
        self.center += delta_cell * self.resolution
        self._shiftMap(-delta_cell)

    def _shiftMap(self, delta_cell):
        shift_fn = csp.shift
        with self.thread_lock:
            self.filled_mask[0] = shift_fn(self.filled_mask[0], delta_cell, order=0, cval=0.0)

    def step(self, points, R, t):
        # Points to cuda and remove nan values
        points = cp.asarray(points)
        points = points[~cp.isnan(points).any(axis=1)]

        # Apply kernel
        with self.thread_lock:
            self.filling_kernel(
                points, float(self.center[0]), float(self.center[1]), cp.asarray(R), cp.asarray(t),    # In params
                self.filled_mask,                                                      # Out params
                size=(points.shape[0])
            )

        return self.filled_mask.get()
            
    def getMapCenter(self):
        return self.center.get()