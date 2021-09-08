#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from parl.utils import logger

__all__ = ['ReplayMemory']


class BCReplayMemory(object):
    def __init__(self, max_size, obs_dim, act_dim,ref_obs_dim):
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs = np.zeros((max_size,obs_dim), dtype='float32')
        self.ref_obs = np.zeros((max_size,ref_obs_dim), dtype='float32')
        self._curr_size = 0
        self._curr_pos = 0

    def sample_batch(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)

        obs = self.obs[batch_idx]
        ref_obs = self.ref_obs[batch_idx]
        return obs, ref_obs

    def make_index(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        return batch_idx

    def sample_batch_by_index(self, batch_idx):
        obs = self.obs[batch_idx]
        ref_obs = self.ref_obs[batch_idx]
        return obs, ref_obs

    def append(self, obs, ref_obs):
        if self._curr_size < self.max_size:
            self._curr_size += 1
        self.obs[self._curr_pos] = obs
        self.ref_obs[self._curr_pos] = ref_obs
        self._curr_pos = (self._curr_pos + 1) % self.max_size

    def size(self):
        return self._curr_size

    def __len__(self):
        return self._curr_size

    def save(self, pathname):
        other = np.array([self._curr_size, self._curr_pos], dtype=np.int32)
        np.savez(
            pathname,
            obs=self.obs,
            ref_obs=self.ref_obs,
            other=other)

    def load(self, pathname):
        data = np.load(pathname)
        other = data['other']
        if int(other[0]) > self.max_size:
            logger.warn('loading from a bigger size rpm!')
        self._curr_size = min(int(other[0]), self.max_size)
        self._curr_pos = min(int(other[1]), self.max_size - 1)

        self.obs[:self._curr_size] = data['obs'][:self._curr_size]
        self.ref_obs[:self._curr_size] = data['ref_obs'][:self._curr_size]
        logger.info("[load rpm]memory loade from {}".format(pathname))