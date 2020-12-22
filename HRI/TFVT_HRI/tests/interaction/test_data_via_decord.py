import pytest
import time
import numpy as np
import multiprocessing as mp
from paddle import fluid

from interaction.common.data_via_decord import XiaoduHiDecordLoader

YOLOv4_MODEL = 'tools/yolov4_paddle/inference_model'
Dataset_pkl = 'data/decord.pkl'

mp.set_start_method('spawn')


def test_decord_dataloader():
    bs, tgt_seq_len, visual_token_dim, nframes = 8, 200, 562, 10
    inst_fm_shape = [512, 5, 5]
    inst_cls_dim, inst_pos_dim = 80, 50
    inst_fm = fluid.data(
        'inst_fm', [-1, tgt_seq_len] + inst_fm_shape, dtype='float32')
    inst_cls = fluid.data(
        'inst_cls', [-1, tgt_seq_len, inst_cls_dim], dtype='float32')
    inst_pos_emb = fluid.data(
        'inst_pos_emb', [-1, tgt_seq_len, inst_pos_dim], dtype='float32')
    visual_tokens = fluid.data(
        'visual_tokens', [-1, tgt_seq_len, visual_token_dim], dtype='float32')
    frame_ids = fluid.data(
        'frame_ids', [-1, tgt_seq_len], dtype='int64')
    padding_mask = fluid.data(
        'padding_mask', [-1, tgt_seq_len], dtype='float32')
    act_ids = fluid.data('act_ids', [-1, nframes], dtype='int64')
    has_act = fluid.data('has_act', [-1, nframes], dtype='float32')
    is_obj = fluid.data('is_obj', [-1, tgt_seq_len], dtype='float32')
    feed_list = [inst_fm, inst_cls, inst_pos_emb, visual_tokens,
                 frame_ids, padding_mask, act_ids, has_act, is_obj]

    # places = fluid.cpu_places()
    places = [fluid.CUDAPlace(0)]
    dataloader = XiaoduHiDecordLoader(
        feed_list, places, YOLOv4_MODEL, Dataset_pkl,
        decord_readers=8, yolov4_detectors=2, post_workers=4,
        batch_size=bs, detector_gpus=[1])
    dataloader.start_workers()

    t_sum, t1 = 0, time.time()
    idx = 1
    for batch in dataloader():
        t2 = time.time()
        t_sum += t2 - t1
        t1 = t2
        print('batch:', idx, ' on avg 1 batch take {} sec'.format(t_sum / idx))

        idx += 1

    print('next epoch...')
    idx = 1
    for batch in dataloader():
        idx += 1
        print(idx)

    dataloader.stop_workers()
