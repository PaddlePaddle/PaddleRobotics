#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr


class BoW(object):
    """
    Bag of Words representation for inference only.

    Args:
        dict_size (int): size of the dictionary.
        emb_dim (int): output dimension of the embedding vector for tokens.
        bow_dim (int): the representation vector dimension from Bag of Words.
    """

    def __init__(self, dict_size, emb_dim=128, bow_dim=128):
        self.dict_size = dict_size
        self.emb_dim = emb_dim
        self.bow_dim = bow_dim

    def build(self, feed_vars):
        seq = feed_vars['seq']
        emb = fluid.layers.embedding(
            seq, size=[self.dict_size, self.emb_dim], is_sparse=True,
            param_attr=ParamAttr(name='emb'))
        pool = fluid.layers.sequence_pool(emb, pool_type='sum')
        soft = fluid.layers.softsign(pool)
        bow = fluid.layers.fc(
            soft,
            size=self.bow_dim,
            act=None,
            name='fc',
            param_attr=ParamAttr(name='fc.w'),
            bias_attr=ParamAttr(name='fc.b'))
        return bow

    def infer(self, main_program=None):
        if main_program is None:
            test_program = fluid.default_main_program().clone(for_test=True)
        else:
            test_program = main_program.clone(for_test=True)

        with fluid.program_guard(test_program):
            sentence = fluid.layers.data(name='seq', shape=[1],
                                         dtype='int64', lod_level=1)
            feed_vars = {'seq': sentence}
            bow = self.build(feed_vars)

        test_program = test_program.clone(for_test=True)
        return test_program, [bow]
