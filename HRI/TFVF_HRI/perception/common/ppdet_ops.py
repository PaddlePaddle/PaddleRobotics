# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay


def ConvNorm(input,
             num_filters,
             filter_size,
             stride=1,
             groups=1,
             norm_decay=0.,
             norm_type='affine_channel',
             norm_groups=32,
             dilation=1,
             lr_scale=1,
             freeze_norm=False,
             act=None,
             norm_name=None,
             initializer=None,
             name=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=((filter_size - 1) // 2) * dilation,
        dilation=dilation,
        groups=groups,
        act=None,
        param_attr=ParamAttr(
            name=name + "_weights",
            initializer=initializer,
            learning_rate=lr_scale),
        bias_attr=False,
        name=name + '.conv2d.output.1')

    norm_lr = 0. if freeze_norm else 1.
    pattr = ParamAttr(
        name=norm_name + '_scale',
        learning_rate=norm_lr * lr_scale,
        regularizer=L2Decay(norm_decay))
    battr = ParamAttr(
        name=norm_name + '_offset',
        learning_rate=norm_lr * lr_scale,
        regularizer=L2Decay(norm_decay))

    if norm_type in ['bn', 'sync_bn']:
        global_stats = True if freeze_norm else False
        out = fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=norm_name + '.output.1',
            param_attr=pattr,
            bias_attr=battr,
            moving_mean_name=norm_name + '_mean',
            moving_variance_name=norm_name + '_variance',
            use_global_stats=global_stats)
        scale = fluid.framework._get_var(pattr.name)
        bias = fluid.framework._get_var(battr.name)
    elif norm_type == 'gn':
        out = fluid.layers.group_norm(
            input=conv,
            act=act,
            name=norm_name + '.output.1',
            groups=norm_groups,
            param_attr=pattr,
            bias_attr=battr)
        scale = fluid.framework._get_var(pattr.name)
        bias = fluid.framework._get_var(battr.name)
    elif norm_type == 'affine_channel':
        scale = fluid.layers.create_parameter(
            shape=[conv.shape[1]],
            dtype=conv.dtype,
            attr=pattr,
            default_initializer=fluid.initializer.Constant(1.))
        bias = fluid.layers.create_parameter(
            shape=[conv.shape[1]],
            dtype=conv.dtype,
            attr=battr,
            default_initializer=fluid.initializer.Constant(0.))
        out = fluid.layers.affine_channel(
            x=conv, scale=scale, bias=bias, act=act)
    if freeze_norm:
        scale.stop_gradient = True
        bias.stop_gradient = True
    return out


class RoIAlign(object):
    """
    RoI alias pooling with encapsulated config.
    Args:
        pooled_height (int): the height of the output feature map
        pooled_width (int): the width of the output feature map
        spatial_scale (float): scale to convert RoI coordinates into sampling coordinates
        sampling_ratio (int): the number of sampling in the gride
    """

    def __init__(self,
                 pooled_height,
                 pooled_width,
                 spatial_scale,
                 sampling_ratio):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def __call__(self, input, rois_input):
        roi_out = fluid.layers.roi_align(
            input=input,
            rois_input=rois_input,
            pooled_height=self.pooled_height,
            pooled_width=self.pooled_width,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio)
        return roi_out


class FPNRoIAlign(object):
    """
    RoI align pooling for FPN feature maps
    Args:
        sampling_ratio (int): number of sampling points
        min_level (int): lowest level of FPN layer
        max_level (int): highest level of FPN layer
        canconical_level (int): the canconical FPN feature map level
        canonical_size (int): the canconical FPN feature map size
        box_resolution (int): box resolution
        mask_resolution (int): mask roi resolution
    """

    def __init__(self,
                 sampling_ratio=0,
                 min_level=2,
                 max_level=5,
                 canconical_level=4,
                 canonical_size=224,
                 box_resolution=7,
                 mask_resolution=14):
        self.sampling_ratio = sampling_ratio
        self.min_level = min_level
        self.max_level = max_level
        self.canconical_level = canconical_level
        self.canonical_size = canonical_size
        self.box_resolution = box_resolution
        self.mask_resolution = mask_resolution

    def __call__(self, head_inputs, rois, spatial_scale, is_mask=False):
        """
        Adopt RoI align onto several level of feature maps to get RoI features.
        Distribute RoIs to different levels by area and get a list of RoI
        features by distributed RoIs and their corresponding feature maps.

        Returns:
            roi_feat(Variable): RoI features with shape of [M, C, R, R],
                where M is the number of RoIs and R is RoI resolution

        """
        k_min = self.min_level
        k_max = self.max_level
        num_roi_lvls = k_max - k_min + 1
        name_list = list(head_inputs.keys())
        input_name_list = name_list[-num_roi_lvls:]
        spatial_scale = spatial_scale[-num_roi_lvls:]
        rois_dist, restore_index = fluid.layers.distribute_fpn_proposals(
            rois, k_min, k_max, self.canconical_level, self.canonical_size)
        # rois_dist is in ascend order
        roi_out_list = []
        resolution = is_mask and self.mask_resolution or self.box_resolution
        for lvl in range(num_roi_lvls):
            name_index = num_roi_lvls - lvl - 1
            rois_input = rois_dist[lvl]
            head_input = head_inputs[input_name_list[name_index]]
            sc = spatial_scale[name_index]
            roi_out = fluid.layers.roi_align(
                input=head_input,
                rois=rois_input,
                pooled_height=resolution,
                pooled_width=resolution,
                spatial_scale=sc,
                sampling_ratio=self.sampling_ratio)
            roi_out_list.append(roi_out)
        roi_feat_shuffle = fluid.layers.concat(roi_out_list)
        roi_feat_ = fluid.layers.gather(roi_feat_shuffle, restore_index)
        roi_feat = fluid.layers.lod_reset(roi_feat_, rois)

        return roi_feat
