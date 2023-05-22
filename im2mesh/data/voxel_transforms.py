#!/usr/bin/env python
# coding=utf-8

import numpy as np
import copy

class CropPlaneVoxel(object):
    def __init__(self,):
        self._index_matrix_list = [
            np.linspace(-1, 1, 32, dtype=np.float32)[:, np.newaxis, np.newaxis],
            np.linspace(-1, 1, 32, dtype=np.float32)[np.newaxis, :, np.newaxis],
            np.linspace(-1, 1, 32, dtype=np.float32)[np.newaxis, np.newaxis, :],
        ]
    def __call__(self, data):
        # data is a dict
        # data['voxels'] is a numpy array
        # data['inputs'] is a numpy array
        data_out = data.copy()
        if 'voxels' in data_out:
            data_out['voxels'] = self._crop_plane(data_out['voxels'])
        data_out['inputs'] = self._crop_plane(data_out['inputs'])
        return data_out
    def _crop_plane(self, data):
        # data is a numpy array
        # data.shape = (32, 32, 32)
        assert data.shape == (32, 32, 32)
        a, b, c = np.random.rand(3)
        index1, index2, index3 = self._index_matrix_list
        v = a*index1 + b*index2 + c*index3
        assert v.shape == (32, 32, 32)
        mask = (v > 0)
        data_out = data * mask
        return data_out


class CropFixedPlaneVoxel(object):
    def __init__(self,):
        self._index_matrix_list = [
            np.linspace(-1, 1, 32, dtype=np.float32)[:, np.newaxis, np.newaxis],
            np.linspace(-1, 1, 32, dtype=np.float32)[np.newaxis, :, np.newaxis],
            np.linspace(-1, 1, 32, dtype=np.float32)[np.newaxis, np.newaxis, :],
        ]
    def __call__(self, data):
        # data is a dict
        # data['voxels'] is a numpy array
        # data['inputs'] is a numpy array
        data_out = data.copy()
        if 'voxels' in data_out:
            data_out['voxels'] = self._crop_plane(data_out['voxels'])
        data_out['inputs'] = self._crop_plane(data_out['inputs'])
        return data_out
    def _crop_plane(self, data):
        # data is a numpy array
        # data.shape = (32, 32, 32)
        assert data.shape == (32, 32, 32)
        a, b, c = np.random.randint(3, size=[3,])
        while abs(a)+abs(b)+abs(c) == 0:
            a, b, c = np.random.randint(3, size=[3,])
        index1, index2, index3 = self._index_matrix_list
        v = a*index1 + b*index2 + c*index3
        assert v.shape == (32, 32, 32)
        mask = (v > 0)
        data_out = data * mask
        return data_out

class CropOctantVoxel(object):
    def __init__(self):
        pass
    def __call__(self, data):
        # data is a dict
        # data['voxels'] is a numpy array
        # data['inputs'] is a numpy array
        data_out = data.copy()
        if 'voxels' in data_out:
            data_out['voxels'] = self._crop_octant(data_out['voxels'])
        data_out['inputs'] = self._crop_octant(data_out['inputs'])
        return data_out
    def _crop_octant(self, data):
        # data is a numpy array
        # data.shape = (32, 32, 32)
        assert data.shape == (32, 32, 32)
        crop_idx = np.random.randint(8)
        sign1, sign2, sign3 = crop_idx//4, crop_idx%4//2, crop_idx%2
        slice1 = [slice(0, 16), slice(16, 32)][int(sign1)]
        slice2 = [slice(0, 16), slice(16, 32)][int(sign2)]
        slice3 = [slice(0, 16), slice(16, 32)][int(sign3)]
        data_out = copy.deepcopy(data)
        data_out[slice1, slice2, slice3] = 0
        return data_out

