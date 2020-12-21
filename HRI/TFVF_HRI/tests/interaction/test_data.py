import pytest
import os
import time
import pickle
import numpy as np
import multiprocessing as mp
from paddle import fluid

from interaction.common.utils import stable_anno_hash, timestamp_to_ms, \
    ms_to_timestamp
from interaction.common.data import DataWorkerV2, XiaoduHiDataloaderv2, \
    SalutationClsDataloader

YOLOv4_MODEL = 'tools/yolov4_paddle/inference_model'
Video_dir = 'data/xiaodu_clips_v2'
Train_data_pkl = 'data/xiaoduHi_train_v2.pkl'
Test_data_pkl = 'data/xiaoduHi_test_v2.pkl'

Full_neg_dir = 'jetson/log_v3'

Salutation_dataset_dir = 'data/salutation_v2/test'
Salutation_dataset_txt = 'data/salutation_v2/test.txt'

mp.set_start_method('spawn')


def _add_delta_time(anno, dt):
    # NOTE: a helper function to modify the timestamp of anno
    anno_copy = {k: v for k, v in anno.items()}
    t = timestamp_to_ms(anno_copy['Time'])
    anno_copy['Time'] = ms_to_timestamp(t + dt)
    return anno_copy


def test_data_worker(save_cache=True):
    # NOTE: for anno_list[6744, 12082],  anno detector has slicing bug!
    # offset = 6736
    offset = 0
    # offset = 3250 * 3
    anno_lst = []
    for pkl in [Train_data_pkl, Test_data_pkl]:
        with open(pkl, 'rb') as f:
            anno_lst.extend(pickle.load(f))

    total_anno = len(anno_lst)
    anno_lst = anno_lst[offset:]
    print('Total:', total_anno)

    manager = mp.Manager()
    data_queue = manager.Queue()
    worker = DataWorkerV2(
        manager, YOLOv4_MODEL, Video_dir,
        [_add_delta_time(a, 200) for a in anno_lst],
        data_queue, gpu=1, use_frames_first=False, read_cache=False)
    worker.start()
    worker.next_epoch()

    idx = offset
    while idx < total_anno:
        print(idx)
        data = data_queue.get()
        if save_cache:
            cache_file = os.path.join(Video_dir, '{}_{}_cache.pkl'.format(
                anno_lst[idx]['VideoID'], stable_anno_hash(anno_lst[idx])))
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print('Saved {}'.format(cache_file))

        # Check instance with full zero feature
        for inst_id, padding_mask in enumerate(list(data['padding_mask'])):
            if padding_mask == 0:
                continue
            assert not np.all(data['visual_tokens'][inst_id, 50:] == 0)

        idx += 1

    worker.stop()


def test_data_worker_full_neg():
    neg_frames_dir = []
    for path in os.listdir(Full_neg_dir):
        path = os.path.join(Full_neg_dir, path)
        if os.path.isdir(path):
            neg_frames_dir.append(path)

    size, anno_lst = 100, []
    for i in range(min(size, len(neg_frames_dir))):
        null_anno = {'WAE_id': 0, 'VideoType': 'neg_frames'}
        null_anno['Path'] = neg_frames_dir[i]
        anno_lst.append(null_anno)

    total_anno = len(anno_lst)
    print('Total:', total_anno)

    manager = mp.Manager()
    data_queue = manager.Queue()
    worker = DataWorkerV2(
        manager, YOLOv4_MODEL, Video_dir, anno_lst, data_queue, gpu=1)
    worker.start()
    worker.next_epoch()

    count = 0
    while count < total_anno:
        data = data_queue.get()
        import ipdb; ipdb.set_trace()
        count += 1
        print(count)

    worker.stop()


def test_dataloader_v2():
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
    dataloader = XiaoduHiDataloaderv2(
        feed_list, places, YOLOv4_MODEL, Video_dir, Train_data_pkl,
        batch_size=bs, num_workers=8, worker_gpus=[1])
    dataloader.start_workers()
    # shrink dataset size
    # dataloader.pos_anno_lst = dataloader.pos_anno_lst[:100]

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


def test_salutation_cls_dataloader():
    bs = 8
    fm_shape = [512, 5, 5]

    fm = fluid.data('fm', [-1] + fm_shape, dtype='float32')
    tree_mask = fluid.data('tree_mask', [-1, 3], dtype='float32')
    root_cls = fluid.data('root_cls', [-1, 1], dtype='int64')
    left_cls = fluid.data('left_cls', [-1, 1], dtype='int64')
    right_cls = fluid.data('right_cls', [-1, 1], dtype='int64')
    feed_list = [fm, tree_mask, root_cls, left_cls, right_cls]

    # places = fluid.cpu_places()
    places = [fluid.CUDAPlace(0)]
    dataloader = SalutationClsDataloader(
        feed_list,
        places,
        Salutation_dataset_dir,
        Salutation_dataset_txt,
        batch_size=bs,
        rebalance=True)

    t_sum, t1 = 0, time.time()
    idx = 1
    for batch in dataloader():
        t2 = time.time()
        t_sum += t2 - t1
        t1 = t2
        print('batch:', idx, ' on avg 1 batch take {} sec'.format(t_sum / idx))

        idx += 1
